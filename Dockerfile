# Base image for TorchServe
FROM pytorch/torchserve:latest-cpu

WORKDIR /home/model-server

COPY model_store/image_lingo.mar /home/model-server/model_store/
COPY lingo_handler.py /home/model-server/

# Install dependencies
COPY requirements.txt  /home/model-server/

# RUN pip install --no-cache-dir -r /home/model-server/requirements.txt

# Expose the port
EXPOSE 8080 8081 8082

# Start the model server
CMD ["torchserve", "--start", "--model-store", "model_store", "--models", "image_lingo.mar", "--dt"]

