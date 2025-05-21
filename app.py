import os
import requests
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Constants
MODEL_PATH = 'modelf.h5'
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1Xzp0sNNe7BeKAsyshnT10NgCFc82kvo2'

# Function to download model if not found
def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Model downloaded successfully.")

# Download model at startup
download_model_if_needed()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this for production

# Configuration
UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load class mapping
df = pd.read_csv('sports.csv')
class_map = dict(zip(df['class id'], df['labels']))

# Load model
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(filepath):
    try:
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_label = class_map.get(predicted_class_index, "Unknown sport")

        return predicted_label, predictions[0][predicted_class_index]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            label, confidence = predict_image(filepath)
            if label:
                confidence_percent = round(confidence * 100, 2)
                return render_template('index.html',
                                       filename=filename,
                                       prediction=label,
                                       confidence=confidence_percent)
            else:
                flash('Error processing image. Please try another.')
                return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
