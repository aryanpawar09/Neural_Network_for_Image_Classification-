import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("fashion_mnist_model.keras")

# Class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

st.title("👕 Fashion MNIST Classifier")
st.write("Upload an image, and the model will classify it.")

# Upload image
uploaded_file = st.file_uploader("Upload a 28x28 grayscale image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Open image & convert to grayscale
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale (1 channel)
    image = image.resize((28, 28))  # Resize to match model input

    # Convert to NumPy array
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1]

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", width=150, clamp=True, channels="gray")

    # Reshape to match model's expected input shape (1, 28, 28, 1)
    image = image.reshape(1, 28, 28, 1)  # Add batch & channel dimension

    # Predict class
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    # Display result
    st.write(f"### Prediction: {class_names[predicted_class]}")
