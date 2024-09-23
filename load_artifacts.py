import mlflow.pytorch
import torch
import torch.jit
import argparse
import os
import shutil

def load_model(model_name, model_version):
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pytorch.load_model(model_uri)
    return model

def save_jit_model(model, model_name, save_path):
    model.eval()
    traced_model = torch.jit.script(model)
    traced_model.save(save_path)
    return traced_model

def load_jit_model(model_path):
    model = torch.jit.load(model_path, map_location=torch.device('cpu'))
    return model

def predict_jit_model(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    return output

def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to JIT")
    parser.add_argument("--base_path", type=str, help="Base path to save the JIT model", default="Artifacts")
    parser.add_argument("--word_map_path", type=str, help="Path to the word map JSON file", default="Data/WORDMAP_flickr8k_4_cap_per_img_4_min_word_freq.json")
    parser.add_argument("--encoder_model_name", type=str, help="Name of the encoder model to load", default="encoder")
    parser.add_argument("--encoder_model_version", type=int, help="Version of the encoder model to load", default=1)
    parser.add_argument("--encoder_save_name", type=str, help="Name to save the JIT model",  default="encoder.pt")
    parser.add_argument("--decoder_model_name", type=str, help="Name of the decoder model to load", default="decoder")
    parser.add_argument("--decoder_model_version", type=int, help="Version of the decoder model to load",  default=1)
    parser.add_argument("--decoder_save_name", type=str, help="Name to save the JIT model",  default="decoder.pt")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.base_path, exist_ok=True)
    
    shutil.copy(args.word_map_path, args.base_path)
    print(f"Word map copied to {args.base_path}")
     
    encoder_model = load_model(args.encoder_model_name, args.encoder_model_version)
    encoder_traced_model = save_jit_model(encoder_model, args.encoder_model_name, os.path.join(args.base_path, args.encoder_save_name))
    print(f"Encoder model saved at {os.path.join(args.base_path, args.encoder_save_name)}")
    
    decoder_model = load_model(args.decoder_model_name, args.decoder_model_version)
    torch.save(decoder_model, os.path.join(args.base_path, args.decoder_save_name))
    print(f"Decoder model saved at {os.path.join(args.base_path, args.encoder_save_name)}")
    
    
    
    
    