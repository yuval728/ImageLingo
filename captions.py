import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path, transform):
    image = Image.open(image_path)
    # image = image.resize([224, 224], Image.LANCZOS)
    image = transform(image).unsqueeze(0)
    return image.to(device)

def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    
    :return: caption, weights for visualization
    """
    
    k=beam_size
    vocab_size=len(word_map)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    image = load_image(image_path, transform)
    
    encoder_out = encoder(image)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)
    
    encoder_out = encoder_out.view(1,-1,encoder_dim)
    num_pixels = encoder_out.size(1)
    
    encoder_out = encoder_out.expand(k,num_pixels, encoder_dim)
    
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)
    
    seqs = k_prev_words
    
    top_k_scores = torch.zeros(k,1).to(device)
    
    seqs_alpha=torch.ones(k, 1, enc_image_size, enc_image_size).to(device)
    
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()
    
    step=1
    
    h, c =decoder.init_hidden_state(encoder_out)
    
    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        
        embeddings = decoder.embedding(k_prev_words).squeeze(1)
        
        awe, alpha = decoder.attention(encoder_out, h)
        
        alpha = alpha.view(-1, enc_image_size, enc_image_size)
        
        gate = decoder.sigmoid(decoder.f_beta(h))
        
        awe = gate*awe
        
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h,c))
        
        scores = decoder.fc(h)
        scores = F.log_softmax(scores, dim=1)
        
        scores = top_k_scores.expand_as(scores)+scores
        
        if step==1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
            
        prev_word_inds = top_k_words//vocab_size #(top_k_words / vocab_size).long()
        next_word_inds = top_k_words % vocab_size
        
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)
        
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word!=word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds)))-set(incomplete_inds))
        
        if len(complete_inds)>0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        
        k-=len(complete_inds)
        
        if k==0:
            break
        
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
        
        if step>50:
            break
        
        step+=1
        
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]
    
    # print('seq:', seq)
    # print('alphas:', alphas)
    return seq, alphas

def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.
    
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    
    image = Image.open(image_path)
    image = image.resize([14*24, 14*24], Image.LANCZOS)
    
    words = [rev_word_map[ind] for ind in seq]
    
    for t in range(len(words)):
        if t>50:
            break
        plt.subplot(int(np.ceil(len(words)/5.)), 5, t+1)
        
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        
        current_alpha = alphas[t,:].detach().numpy()
        
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.reshape(14,14), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.reshape(14,14), [14*24, 14*24])
        
        if t==0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        
        # plt.set_cmap(cm.get_cmap('jet'))
        plt.axis('off')

    plt.show()
    
def main(args):
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()
    
    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    
    rev_word_map = {v: k for k, v in word_map.items()}
    
    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, args.image, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)
    
    # Visualize caption and attention of best sequence
    visualize_att(args.image, seq, alphas, rev_word_map)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')
    
    parser.add_argument('--image', '-i', help='path to image', required=True)
    parser.add_argument('--checkpoint', '-c', help='path to checkpoint', required=True)
    parser.add_argument('--word_map', '-wm', help='path to word map JSON', required=True)
    parser.add_argument('--beam_size', '-b', default=3, type=int, help='beam size for beam search')
    
    args = parser.parse_args()
    main(args)
    