import torch
import json
from ts.torch_handler.base_handler import BaseHandler
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os
import io

class LingoHandler(BaseHandler):
    def __init__(self):
        super(LingoHandler, self).__init__()
        self.initialized = False
        
    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        
        
        model_path = os.path.join(model_dir, "encoder.pt")
        self.encoder = torch.jit.load(model_path, map_location=torch.device('cpu'))
        self.encoder.eval()
        
        model_path = os.path.join(model_dir, "decoder.pt")
        self.decoder = torch.load(model_path, map_location=torch.device('cpu'))
        self.decoder.eval()
        
        word_map_path = os.path.join(model_dir, "WORDMAP_flickr8k_4_cap_per_img_4_min_word_freq.json")
        with open(word_map_path, "r") as j:
            self.word_map = json.load(j)
        self.rev_word_map = {v: k for k, v in self.word_map.items()}
        
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.initialized = True
        
    def preprocess(self, data):
        image = data[0].get("data") or data[0].get("body")
        image = Image.open(io.BytesIO(image)).convert("RGB")
        image = self.transforms(image).unsqueeze(0)
        return image
    
    def inference(self, image):
        
        encoder_out = self.encoder(image)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        encoder_out = encoder_out.view(1, -1, encoder_dim)
        
        k=1
        encoder_out = encoder_out.expand(k, encoder_out.size(1), encoder_out.size(2))
        k_prev_words = torch.LongTensor([[self.word_map['<start>']]] * k)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1)
        
        complete_seqs = list()
        complete_seqs_scores = list()
        
        step = 1
        h, c = self.decoder.init_hidden_state(encoder_out)
        
        while True:
            embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
            awe, _ = self.decoder.attention(encoder_out, h)
            gate = self.decoder.sigmoid(self.decoder.f_beta(h))
            awe = gate * awe
            h, c = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
            scores = self.decoder.fc(h)
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores
            
            if step == 1:
                top_k_scores, top_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_words = scores.view(-1).topk(k, 0, True, True)
                
            prev_word_inds = (top_words / len(self.word_map)).long()
            next_word_inds = top_words % len(self.word_map)
            
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != self.word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
                
            k -= len(complete_inds)
            
            if k == 0 or step > 50:
                break
            
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            
            step += 1
            
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        sentence = [self.rev_word_map[ind] for ind in seq]
        return sentence
    
    def postprocess(self, data):
        return [data]

# _service = LingoHandler()            

        
        
        
        