# Step 1: Import required libraries
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Step 2: Load the CSV file with class id -> label mapping
csv_path = 'sports.csv'   # Assumes it's in the same folder
df = pd.read_csv(csv_path)
print("CSV Loaded. Sample data:")
print(df.head())

# Create mapping dictionary
class_map = dict(zip(df['class id'], df['labels']))

# Step 3: Load your model
model_path = 'modelf.h5'  # Assumes it's in the same folder
model = load_model(model_path, compile=False)
print("Model loaded.")

# Step 4: Provide path to an image file manually
# Replace this with your actual image file name in the same folder
image_path = 'crick.jpg'  # Example: change to your actual image file name

if not os.path.exists(image_path):
    print("Error: Image file not found!")
    exit()

print("Image selected:", image_path)

# Step 5: Preprocess the image (resize to 224x224)
img = load_img(image_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Step 6: Predict
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions)
print("Predicted class index:", predicted_class_index)

# Step 7: Map index to label
predicted_label = class_map.get(predicted_class_index, "Unknown class")
print("Predicted label:", predicted_label)
