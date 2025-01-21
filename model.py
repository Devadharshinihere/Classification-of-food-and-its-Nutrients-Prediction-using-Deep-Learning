from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("D:\d\food_classification.h5")

# Preprocess the uploaded image before prediction
def preprocess_img(img_stream):
    # Load the image from the stream and resize it to (224, 224)
    img = image.load_img(img_stream, target_size=(224, 224))  # Resize to (224, 224)
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    return img_array

# Predict the result using the loaded model
def predict_result(img_array):
    # Predict the class of the image
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction)  # Get the index of the highest probability
    class_names = ['adhirasam', 'aloo_gobi', 'biryani', 'butter_chicken', 'chana_masala', 'poha', 'gulab_jamun', 'chicken_tikka']  # Update with all class names
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class
