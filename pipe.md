### Create input files:
    python create_input_files.py --dataset flickr8k --karpathy_json_path ../data/dataset_flickr8k.json --image_folder ../data/Images --output_folder ../data/Data 

### Train the model:
    python src/train.py --data_folder data/Data --checkpoint checkpoints/checkpoint_flickr8k_4_cap_per_img_4_min_word_freq.pth.tar --save_dir checkpoints 

### Evaluate the model:
    python src/eval.py   

### Register the model:
    <!-- mlflow models add -m checkpoints/checkpoint_flickr8k_4_cap_per_img_4_min_word_freq.pth.tar -n flickr8k -->
    python src/model_register.py --run_id a3b39b26b120470eb299bc8273b7988e  --model_type encoder --model_name encoder --best_artifact 
    python src/model_register.py --run_id a3b39b26b120470eb299bc8273b7988e  --model_type decoder --model_name decoder --best_artifact 

### load the artifacts:
    python src/load_artifacts.py --base_path artifacts --word_map_path data/Data/WORDMAP_flickr8k_4_cap_per_img_4_min_word_freq.json --encoder_model_version 8 --decoder_model_version 4  

### Create the model-archive or MAR:
    torch-model-archiver --model-name image_lingo --version 1.1 --serialized-file artifacts/encoder.pt --handler src/lingo_handler.py --extra-files "artifacts/WORDMAP_flickr8k_4_cap_per_img_4_min_word_freq.json,artifacts/decoder.pt,src/models.py" --export-path model_store -f 

### Build the docker image:
    docker build -t image_lingo:1.1 .
    docker build -t image_lingo .  

### Run the docker image:
    docker run -p 8080:8080 -p 8081:8081 -p 8082:8082 image_lingo 

### Test the docker image:
    curl http://localhost:8080/ping
    curl http://localhost:8081/models
    curl http://localhost:8081/models/image_lingo
    curl http://localhost:8081/models/image_lingo/versions
    curl http://localhost:8081/models/image_lingo/versions/1.1
    curl -X POST http://localhost:8080/predictions/image_lingo -T data/Images/1000268201_693b08cb0e.jpg