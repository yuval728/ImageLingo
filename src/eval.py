import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
from datasets import CaptionDataset
from utils import *  # noqa: F403
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import json
import warnings
import mlflow
import argparse

warnings.filterwarnings("ignore")


def evaluate(beam_size, data_folder, data_name, checkpoint, word_map_file, device):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    
    # Load word map (word2ix)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Load model checkpoint
    checkpoint = torch.load(checkpoint, map_location=device)

    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()

    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=0, pin_memory=True
    )

    references = list()
    hypotheses = list()

    for i, (image, caps, caplens, allcaps) in enumerate(tqdm(loader, desc='Evaluating at beam size ' + str(beam_size))):
        k = beam_size
        image = image.to(device)
        encoder_out = encoder(image)
        encoder_dim = encoder_out.size(3)
        encoder_out = encoder_out.view(1, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1).to(device)
        seqs_alpha = torch.ones(k, 1, num_pixels).to(image.device)

        complete_seqs = list()
        complete_seqs_scores = list()
        complete_seqs_alpha = list()

        step = 1
        h = torch.zeros(k, decoder.decoder_dim).to(image.device)
        c = torch.zeros(k, decoder.decoder_dim).to(image.device)

        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)
            awe, alpha = decoder.attention(encoder_out, h)
            alpha = alpha.view(-1, num_pixels)

            lstm_input = torch.cat([embeddings, awe], dim=1)
            lstm_output, (h_new, c_new) = decoder.lstm(
                lstm_input.unsqueeze(1), (h.unsqueeze(0), c.unsqueeze(0))
            )
            h = h_new.squeeze(0)
            c = c_new.squeeze(0)

            scores = decoder.fc(h)
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

            prev_word_inds = (top_k_words / vocab_size).long()
            next_word_inds = top_k_words % vocab_size

            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)

            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())

            k -= len(complete_inds)
            if k == 0:
                break

            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            if step > 50:
                break

            step += 1

        if len(complete_seqs_scores) == 0:
            continue

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}], img_caps)
        )

        references.append(img_captions)
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

    bleu4 = corpus_bleu(references, hypotheses)
    return bleu4


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate an image captioning model.')
    parser.add_argument('--beam_size', type=int, default=1, help='beam size for generating captions')
    parser.add_argument('--data_folder', type=str, default='data/Data', help='folder containing the data')
    parser.add_argument('--data_name', type=str, default='flickr8k_4_cap_per_img_4_min_word_freq', help='name of the dataset')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/BEST_checkpoint_flickr8k_4_cap_per_img_4_min_word_freq.pth.tar', help='path to model checkpoint')
    parser.add_argument('--word_map', type=str, default='data/Data/WORDMAP_flickr8k_4_cap_per_img_4_min_word_freq.json', help='path to word map JSON')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device to run on (default: cuda if available)')
    parser.add_argument('--mlflow_experiment', type=str, default='ImageLingo', help='MLflow experiment name')
    parser.add_argument('--mlflow_tracking_uri', type=str, default='http://localhost:5000', help='MLflow tracking URI')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID to resume or continue tracking')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    mlflow.set_experiment(args.mlflow_experiment)
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    with mlflow.start_run(run_id=args.run_id, nested=True):
        mlflow.log_param("Beam Size", args.beam_size)
        bleu = evaluate(args.beam_size, args.data_folder, args.data_name, args.checkpoint, args.word_map, args.device)
        mlflow.log_metric("BLEU-4", bleu)

    print("\nBLEU-4 score @ beam size of %d is %.4f." % (args.beam_size, bleu))
