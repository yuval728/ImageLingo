import argparse
import os
from utils import create_input_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create input files for training, validation, and test data')
    
    parser.add_argument('--dataset', default='coco', help='dataset name, e.g. coco', choices=['coco', 'flickr8k', 'flickr30k'])
    parser.add_argument('--karpathy_json_path', default='data/dataset_coco.json', help='path to Karpathy JSON file')
    parser.add_argument('--image_folder', default='data/images/', help='path to folder with images')
    parser.add_argument('--captions_per_image', type=int, default=5, help='number of captions to sample per image')
    parser.add_argument('--min_word_freq', type=int, default=5, help='minimum word frequency')
    parser.add_argument('--output_folder', default='data/', help='folder to save files')
    parser.add_argument('--max_len', type=int, default=100, help='maximum caption length')
    
    args = parser.parse_args()
    print(args)
    
    create_input_files(args.karpathy_json_path, args.image_folder, args.output_folder, args.captions_per_image, args.min_word_freq, args.max_len)
    print('Input files created')