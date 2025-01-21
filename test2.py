import tensorflow as tf  # Import TensorFlow
from tensorflow.keras.preprocessing import image
import numpy as np

# Path to the image to be predicted
img_path = r"D:\d\018.png"  # Replace with your image path

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))  # Resize to 224x224
img_array = image.img_to_array(img)  # Convert to array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize pixel values to [0, 1]

# Load the model
model = tf.keras.models.load_model(r"D:\d\food1_vgg16_model.h5")

# Predict the class
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions, axis=1)[0]
confidence_score = np.max(predictions) * 100

# Define class labels explicitly (replace these with your actual class names)
class_labels = ['pizza', 'burger', 'naan', 'salad', 'fries', 'sandwich', 
                'noodles', 'soup', 'dessert', 'rice', 'curry', 'biriyani', 
                'dal', 'roti', 'idli', 'dosa', 'samosa', 'pani_puri', 
                'chaat', 'tandoori', 'kebab', 'paratha']  # Replace with your class names

# Map index to class label
predicted_class = class_labels[predicted_class_index]

# Output results
print(f"Predicted Class: {predicted_class}")
print(f"Confidence Score: {confidence_score:.2f}%")
