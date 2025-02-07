from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load your pre-trained model (replace 'model.h5' with your actual model file)
model = tf.keras.models.load_model('model.h5')  # Make sure to replace this path with your model file

# Route for home page (serving the front-end HTML)
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    # Get the file from the request
    file = request.files['file']
    
    # If no file was selected, return an error
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Open the uploaded image
    image = Image.open(file)
    image = image.resize((224, 224))  # Resize to match the model's input size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make the prediction using the model
    prediction = model.predict(image)
    class_idx = np.argmax(prediction, axis=1)  # Get index of class with highest probability
    
    # Mapping class index to class name
    class_names = ['Normal', 'Pneumonia', 'COVID']
    predicted_class = class_names[class_idx[0]]

    # Return prediction as JSON
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
