import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14, fine_tune=False):
        super(Encoder, self).__init__()

        self.enc_image_size = encoded_image_size

        resnet_weights = torchvision.models.ResNet101_Weights.DEFAULT

        resnet = torchvision.models.resnet101(weights=resnet_weights)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )

        self.fine_tune(fine_tune)

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """

        out = self.resnet(images)
        out = self.adaptive_pool(out)

        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=False):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """

        for p in self.resnet.parameters():
            p.requires_grad = False

        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention network
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """

        super(Attention, self).__init__()
        self.encoder_attn = nn.Linear(
            encoder_dim, attention_dim
        )  # linear layer to transform encoded image
        self.decoder_attn = nn.Linear(
            decoder_dim, attention_dim
        )  # linear layer to transform decoder's output

        self.full_attn = nn.Linear(
            attention_dim, 1
        )  # linear layer to calculate values to be softmax-ed

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)

        :return: attention weighted encoding, weights
        """

        attn1 = self.encoder_attn(encoder_out)

        attn2 = self.decoder_attn(decoder_hidden)

        attn = self.full_attn(self.relu(attn1 + attn2.unsqueeze(1))).squeeze(2)

        alpha = self.softmax(attn)
        attention_weigthed_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weigthed_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder
    """

    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        encoder_dim=2048,
        dropout=0.5,
    ):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropoutL dropout
        """

        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim,
        )

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )
        self.dropout = nn.Dropout(p=self.dropout)

        self.lstm = nn.LSTM(embed_dim + encoder_dim, decoder_dim, batch_first=True)

        self.fc = nn.Linear(decoder_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        """
        Initialized some parameters with values from the uniform distribution,
        for easier convergence.
        """

        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings
        :param embeddings: pre-trained embeddings
        """

        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """

        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lenghts: caption_lengths, a tensor of dimension (batch_size, 1)

        :returns: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # Flatten image
        num_pixels = encoder_out.size(1)
        
        
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)
        
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Initialize LSTM state
        h_t = torch.zeros(batch_size, self.decoder_dim).to(encoder_out.device)
        c_t = torch.zeros(batch_size, self.decoder_dim).to(encoder_out.device)

        outputs = torch.zeros(
            batch_size, max(decode_lengths), vocab_size
        ).to(encoder_out.device)
        alphas = torch.zeros(
            batch_size, max(decode_lengths), num_pixels
        ).to(encoder_out.device)


        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])  # noqa: E741

            context_vector, alpha = self.attention(
                encoder_out[:batch_size_t], h_t[:batch_size_t]
            )

            lstm_input = torch.cat(
                (embeddings[:batch_size_t, t], context_vector), dim=1
            )

            temp, (h_t_new, c_t_new) = self.lstm(
                lstm_input.unsqueeze(1),
                (h_t[:batch_size_t].unsqueeze(0), c_t[:batch_size_t].unsqueeze(0)),
            )
            h_t = h_t.clone()  # Create a copy to avoid in-place modification
            h_t[:batch_size_t] = h_t_new.squeeze(0)
            c_t = c_t.clone()  # Create a copy to avoid in-place modification
            c_t[:batch_size_t] = c_t_new.squeeze(0)



            output = self.fc(h_t[:batch_size_t])

            outputs[:batch_size_t, t, :] = output
            alphas[:batch_size_t, t, :] = alpha

        return outputs, encoded_captions, decode_lengths, alphas, sort_ind


class ImageLingo(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super(ImageLingo, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions, caption_lengths):
        encoder_out = self.encoder(images)
        outputs = self.decoder(encoder_out, captions, caption_lengths)
        return outputs


if __name__ == "__main__":
    from datasets import CaptionDataset
    import json

    word_map_file = "data/Data/WORDMAP_flickr8k_4_cap_per_img_4_min_word_freq.json"
    with open(word_map_file, "r") as j:
        word_map = json.load(j)

    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)
    print("vocab_size:", vocab_size)

    test_dataset = CaptionDataset(
        data_folder="data/Data",
        data_name="flickr8k_4_cap_per_img_4_min_word_freq",
        split="TEST",
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True
    )
    images, captions, caption_lengths, _ = next(iter(test_loader))

    encoder = Encoder()
    decoder = DecoderWithAttention( attention_dim=512, embed_dim=512, decoder_dim=512, vocab_size=vocab_size, encoder_dim=2048, dropout=0.5)
    outputs, encoded_captions, decode_lengths, alphas, sort_ind = decoder(encoder(images), captions, caption_lengths)
    print(outputs.size())
    print(alphas.size())

    model = ImageLingo(encoder, decoder)

    # torch.jit.trace( model, (images, captions, caption_lengths) )
