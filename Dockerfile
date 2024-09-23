# Base image for TorchServe
FROM pytorch/torchserve:latest-cpu

WORKDIR /home/model-server

COPY model_store/image_lingo.mar /home/model-server/model_store/

# Install dependencies
COPY requirements.txt  /home/model-server/

RUN pip install --no-cache-dir -r /home/model-server/requirements.txt

# Copy the code
# COPY . /home/model-server/

# Expose the port
EXPOSE 8080 8081 8082

# Start the model server
CMD ["torchserve", "--start", "--model-store", "model_store", "--models", "image_lingo.mar", "--dt"]

