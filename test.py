import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define constants
img_size = (224, 224)  # Image size expected by the model
model_path = r"D:\d\food1_vgg16_model.h5"  # Path to your saved model
image_path = r"D:\d\001.jpg"  # Path to your single test image (corrected)
train_dir = r"D:\d\Dataset\train"  # Path to your training dataset

# Load the saved model
model = load_model(model_path)
print(f"Model loaded from {model_path}")

# Load the class labels from the training data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Without augmentation for testing
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=32,
    class_mode="categorical"
)

# Get the class labels from the generator
class_labels = train_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}  # Reverse the dictionary (index -> label)

# Load and preprocess the image
def preprocess_image(image_path, img_size):
    img = load_img(image_path, target_size=img_size)  # Load image and resize
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Preprocess the input image
input_image = preprocess_image(image_path, img_size)

# Make a prediction
predictions = model.predict(input_image)
predicted_class = np.argmax(predictions, axis=1)[0]  # Get the class index
confidence = np.max(predictions)  # Get the confidence score

# Map class index to class label
predicted_label = class_labels[predicted_class]

# Print the result
print(f"Predicted Class: {predicted_label}")
print(f"Confidence Score: {confidence * 100:.2f}%")
