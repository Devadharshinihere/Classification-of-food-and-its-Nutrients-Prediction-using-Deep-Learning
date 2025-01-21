import os
import numpy as np
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Define constants
IMG_SIZE = (224, 224)  # Image size for the model
UPLOAD_FOLDER = 'static/uploads'  # Directory to save uploaded images
MODEL_PATH = r'C:\Users\devad\OneDrive\Desktop\d\d\food1_vgg16_model.h5'

# Configure app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
try:
    model = load_model(MODEL_PATH)
except OSError:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please check the path.")

# Nutrition data for food classes
NUTRITION_DATA = {
    'bread': {'calories': 75, 'protein': 2.6, 'fat': 1.0, 'carbs': 14.0},
    'burger': {'calories': 375, 'protein': 16.0, 'fat': 17.5, 'carbs': 37.5},
    'butter_naan': {'calories': 200, 'protein': 4.0, 'fat': 5.0, 'carbs': 28.0},
    'chai': {'calories': 120, 'protein': 2.0, 'fat': 5.0, 'carbs': 16.0},
    'chapati': {'calories': 80, 'protein': 3.0, 'fat': 1.5, 'carbs': 15.0},
    'chole_bhature': {'calories': 500, 'protein': 12.0, 'fat': 15.0, 'carbs': 60.0},
    'dal_makhani': {'calories': 350, 'protein': 15.0, 'fat': 18.0, 'carbs': 30.0},
    'dhokla': {'calories': 150, 'protein': 6.0, 'fat': 3.0, 'carbs': 25.0},
    'fried_rice': {'calories': 275, 'protein': 6.0, 'fat': 10.0, 'carbs': 40.0},
    'idly': {'calories': 40, 'protein': 2.0, 'fat': 0.4, 'carbs': 8.0},
    'jalebi': {'calories': 250, 'protein': 3.0, 'fat': 12.0, 'carbs': 30.0},
    'kaathi_roll': {'calories': 250, 'protein': 10.0, 'fat': 10.0, 'carbs': 25.0},
    'kadai_paneer': {'calories': 400, 'protein': 18.0, 'fat': 22.0, 'carbs': 25.0},
    'kulfi': {'calories': 200, 'protein': 4.0, 'fat': 12.0, 'carbs': 20.0},
    'masala_dosa': {'calories': 200, 'protein': 4.0, 'fat': 6.0, 'carbs': 30.0},
    'momos': {'calories': 30, 'protein': 2.0, 'fat': 1.0, 'carbs': 5.0},
    'naan': {'calories': 200, 'protein': 4.0, 'fat': 5.0, 'carbs': 28.0},
    'pani_puri': {'calories': 100, 'protein': 2.0, 'fat': 3.0, 'carbs': 15.0},
    'pakode': {'calories': 175, 'protein': 4.0, 'fat': 10.0, 'carbs': 20.0},
    'pav_bhaji': {'calories': 350, 'protein': 8.0, 'fat': 15.0, 'carbs': 45.0},
    'pizza': {'calories': 250, 'protein': 8.0, 'fat': 10.0, 'carbs': 25.0},
    'samosa': {'calories': 175, 'protein': 3.0, 'fat': 8.0, 'carbs': 25.0},
}

# Preprocess a single image
def preprocess_image(img_path):
    try:
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
        return np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

# Predict the food class and nutritional facts
def predict_food_class(img_path):
    img_array = preprocess_image(img_path)
    if img_array is None:
        return "Invalid image.", None

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = list(NUTRITION_DATA.keys())[predicted_index]
    nutrition_facts = NUTRITION_DATA.get(predicted_class, {})
    return predicted_class, nutrition_facts

# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files['image']
        if uploaded_file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)
            
            predicted_class, nutrition_facts = predict_food_class(file_path)
            if nutrition_facts:
                return render_template("result.html", 
                                       predicted_class=predicted_class, 
                                       nutrition_facts=nutrition_facts, 
                                       image_path=url_for('static', filename=f'uploads/{uploaded_file.filename}'))
            else:
                return render_template("error.html", message="Error in processing image.")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
