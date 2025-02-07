import os
import numpy as np
import base64
import requests
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Initialize the Flask application
app = Flask(__name__)

# Secret key for session management (optional)
app.secret_key = 'c9f2d37b7c3e4a7b9f58d8ebf5b1b024'

# GitHub URL where the model is stored
MODEL_URL = "https://raw.githubusercontent.com/Gnanesh-Gone/project-1/main/model/covid_pneumonia_normal_cnn_model.h5"

# File upload configuration
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to download the model file from GitHub
def download_model():
    model_path = os.path.join(os.getcwd(), 'model', 'covid_pneumonia_normal_cnn_model.h5')
    
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Download model file from GitHub
        response = requests.get(MODEL_URL)
        with open(model_path, 'wb') as model_file:
            model_file.write(response.content)
    return model_path

# Load the trained model
model = load_model(download_model())

# Route to serve uploaded files from the 'uploads' folder
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route to check if the user is logged in before accessing index (removed login check)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']

    # Check if file is allowed and process the image
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image
        img = image.load_img(file_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Rescale the image (same as in training)

        # Get the prediction for the X-ray image
        prediction_probabilities = model.predict(img_array)
        class_labels = {0: 'COVID', 1: 'Normal', 2: 'Pneumonia'}
        prediction_class = np.argmax(prediction_probabilities)
        prediction = class_labels[prediction_class]

        # Convert image to Base64 to embed in HTML
        with open(file_path, "rb") as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')

        # Render the result page with prediction and Base64 image
        return render_template('result.html', prediction=prediction, image_data=img_base64)

    return 'Invalid file format'

# Run the app
if __name__ == '__main__':
    # Get the port from environment variable (required by Render)
    port = int(os.environ.get("PORT", 5000))

    # Ensure upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Run the app on all available IPs at the specified port
    app.run(debug=True, host='0.0.0.0', port=port)
