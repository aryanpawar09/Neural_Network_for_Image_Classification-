import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained model
MODEL_PATH = "fashion_mnist_model (2).keras"  # Ensure the correct file exists
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

st.title("👕 Fashion MNIST Classifier")
st.write("Upload a 28x28 grayscale image, and the model will classify it.")

# Upload image
uploaded_file = st.file_uploader("Upload a 28x28 grayscale image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        # Open image & convert to grayscale
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale (1 channel)
        image = image.resize((28, 28))  # Resize to match model input

        # Convert to NumPy array
        image_array = np.array(image, dtype=np.float32)

        # Ensure correct preprocessing
        if image_array.max() > 1:  # If values are in [0, 255], normalize to [0,1]
            image_array /= 255.0  

        # Debugging: Check image array properties
        st.write(f"Image Shape Before Reshape: {image_array.shape}")
        st.write(f"Pixel Value Range: Min {image_array.min()}, Max {image_array.max()}")

        # Reshape for model (batch size, height, width, channels)
        image_array = image_array.reshape(1, 28, 28, 1)

        # Predict class
        prediction = model.predict(image_array)

        # Find predicted class
        predicted_class = np.argmax(prediction)

        # Display result
        st.write(f"### 🎯 Prediction: {class_names[predicted_class]}")

        # Debugging: Show raw probabilities
        st.write("🔍 Raw Predictions:", prediction)
    
    except Exception as e:
        st.error(f"Error processing image: {e}")
