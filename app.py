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
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image) / 255.0  # Normalize

    # Display the image
    st.image(image, caption="Uploaded Image", width=150, clamp=True, channels="gray")

    # Predict
    prediction = model.predict(image.reshape(1, 28, 28))
    predicted_class = np.argmax(prediction)

    st.write(f"### Prediction: {class_names[predicted_class]}")
