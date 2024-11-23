# ImageLingo

ImageLingo is an image captioning project that uses deep learning to generate captions for images. The project is built using PyTorch and includes training, evaluation, and deployment components.

## Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Docker
- DVC
- MLflow

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Yuval728/imagelingo.git
    cd imagelingo
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up DVC:
    ```sh
    dvc pull
    ```

## Explanation and Approach

### Overview

ImageLingo is designed to generate descriptive captions for images using a deep learning model. The project leverages a combination of Convolutional Neural Networks (CNNs) for image feature extraction and Recurrent Neural Networks (RNNs) for sequence generation.

### Data Preparation

The dataset used for training the model is the Flickr8k dataset, which contains 8,000 images each with five different captions. The data preparation involves the following steps:

1. **Image Preprocessing**: Images are resized and normalized to ensure consistency.
2. **Caption Tokenization**: Captions are tokenized and converted into sequences of word indices.
3. **Vocabulary Creation**: A vocabulary is created based on the frequency of words in the captions.

### Model Architecture

The model consists of two main components:

1. **Encoder (CNN)**: A pre-trained CNN (such as ResNet) is used to extract features from the images. The final convolutional layer's output is used as the image representation.
2. **Decoder (RNN)**: An RNN (such as LSTM) is used to generate captions based on the image features. The decoder is trained to predict the next word in the sequence given the previous words and the image features.

### Training

The training process involves optimizing the model to minimize the difference between the generated captions and the actual captions. The following steps are performed:

1. **Forward Pass**: The image is passed through the encoder to obtain the image features. The decoder then generates a caption based on these features.
2. **Loss Calculation**: The loss is calculated based on the difference between the generated caption and the actual caption.
3. **Backward Pass**: The gradients are computed and the model parameters are updated to minimize the loss.

### Evaluation

The model is evaluated using standard metrics such as BLEU score, which measures the similarity between the generated captions and the actual captions. The evaluation process involves:

1. **Generating Captions**: The model generates captions for the test images.
2. **Calculating Metrics**: The generated captions are compared with the actual captions using metrics like BLEU score.

### Deployment

The trained model is deployed using Docker and TorchServe. The deployment process involves:

1. **Creating Model Archive**: The model is packaged into a model archive file (MAR) using Torch Model Archiver.
2. **Building Docker Image**: A Docker image is created with the necessary dependencies and the model archive.
3. **Running Docker Container**: The Docker container is run to serve the model and provide an API for generating captions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.