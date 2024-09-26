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
# import argparse
warnings.filterwarnings("ignore")


data_folder = 'data/Data'
data_name = 'flickr8k_4_cap_per_img_4_min_word_freq'

checkpoint = 'checkpoints/BEST_checkpoint_flickr8k_4_cap_per_img_4_min_word_freq.pth.tar'

word_map_file = 'data/Data/WORDMAP_flickr8k_4_cap_per_img_4_min_word_freq.json'

device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True

checkpoint = torch.load(checkpoint, weights_only=False)

decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()

encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

with open(word_map_file, 'r') as j:
    word_map = json.load(j)

rev_word_map = {v:k for k, v in word_map.items()}
vocab_size = len(word_map)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size = 1, shuffle=True, num_workers=0, pin_memory=True
    )
    
    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!
    
    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    
    references = list()
    hypotheses = list()
    
    for i, (image, caps, caplens, allcaps) in enumerate(tqdm(loader, desc='Evaluating at beam size' + str(beam_size))):
        
        k = beam_size
        
        image=image.to(device)
        
        encoder_out = encoder(image)
        # enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        
        encoder_out = encoder_out.view(1,-1,encoder_dim)
        num_pixels = encoder_out.size(1)
        
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
        
        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)
        
        # Tensor to store top k sequences; now they're just <start>
        seqs =k_prev_words
        
        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k,1).to(device)
        
        complete_seqs = list()
        complete_seqs_scores = list()
        
        step = 1
        h,c = decoder.init_hidden_state(encoder_out)
        
        while True:
            
            embeddings = decoder.embedding(k_prev_words).squeeze(1)
    
            awe, _ = decoder.attention(encoder_out, h)
            
            gate = decoder.sigmoid(decoder.f_beta(h))
            awe = gate * awe
            
            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
            scores = decoder.fc(h)
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores
            
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
                
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k,0, True, True)
                
            # Convert unrolled indices to actual indices of scores
            prev_word_inds = (top_k_words / vocab_size).long()
            next_word_inds = top_k_words % vocab_size
            
            
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            
            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word!=word_map['<end>']]
            
            
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            
            if len(complete_inds)>0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            
            k-=len(complete_inds)
            
            if k==0:
                break
            
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores=top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words=next_word_inds[incomplete_inds].unsqueeze(1)
            
            if step>50:
                break
            
            step+=1
        
        # print(f"Complete Sequences: {complete_seqs}")
        # print(f"Complete Sequence Scores: {complete_seqs_scores}")
        
        if len(complete_seqs_scores) == 0:
            continue  # or handle this case appropriately
        
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'],word_map['<pad>']}], img_caps)
        )
        
        references.append(img_captions)
        
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])   
        
        assert len(references)==len(hypotheses)
        
    bleu4 = corpus_bleu(references,hypotheses)
    
    return bleu4         

if __name__ == '__main__':
    beam_size = 1
    
    mlflow.set_experiment("ImageLingo")
    mlflow.set_tracking_uri("http://localhost:5000")
    run_id = input("Enter the run ID: ")
    with mlflow.start_run( run_id=run_id, nested=True):
        mlflow.log_param("Beam Size", beam_size)
        bleu = evaluate(beam_size)
        mlflow.log_metric("BLEU-4", bleu)
        
        
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, bleu))