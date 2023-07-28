# app.py
from flask import Flask, request, jsonify
import torch
from torchvision.transforms import ToTensor
from PIL import Image

# Load the saved model
# (The code for CustomResNet and predict_class function from the previous code snippet)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    image_file = request.files.get('image')
    image_url = request.form.get('image_url')

    if image_file:
        # If the image is sent as a file, process it
        image = Image.open(image_file)
    elif image_url:
        # If the image is sent as a URL, download and process it
        # You may need to install `requests` library for this
        import requests
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    else:
        return jsonify({'error': 'No image data provided.'}), 400

    # Make the prediction using the model
    predicted_class = predict_class(image, model)

    # Return the predicted class to the client
    return jsonify({'predicted_class': predicted_class}), 200

if __name__ == '__main__':
    app.run(debug=True)
