import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('age_gender_model.h5')

# Load and preprocess the image
image_path = './test1.jpg'
image = Image.open(image_path).convert('L')  # Open the image in grayscale
image = image.resize((128, 128), Image.LANCZOS)  # Resize the image to match the model input shape
image = np.array(image) / 255.0  # Normalize the image

# Reshape the image for model input
image = np.expand_dims(image, axis=-1)  # Add an extra dimension for the grayscale channel
image = np.expand_dims(image, axis=0)  # Add an extra dimension for the batch size

# Make predictions
gender_dict={0:'Male',1:'Female'}
predictions = model.predict(image)
gender_prediction = gender_dict[round(predictions[0][0][0])]
age_prediction = int(predictions[1][0][0])


# Print the predictions
print("Predicted Gender:", gender_prediction)
print("Predicted Age:", age_prediction)
