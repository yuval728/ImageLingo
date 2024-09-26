import requests

# Inference URL for your deployed model
url = 'http://127.0.0.1:8080/predictions/image_lingo'

# Path to the image you want to caption
image_path = 'test_images/temp.jpg'

# Open the image file
with open(image_path, 'rb') as f:
    image_data = f.read()

# Send a POST request to the model
response = requests.post(url, data=image_data)

# Output the result
print('Caption:', response.text)
