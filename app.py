import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Load the trained model from the HDF5 format
model = tf.keras.models.load_model('mnist_cnn_improved.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = np.array(image).astype('float32') / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

# Function to get feature maps
def get_feature_maps(model, image):
    layer_outputs = [layer.output for layer in model.layers if isinstance(layer, Conv2D) or isinstance(layer, MaxPooling2D)]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(image)
    return activations

# Streamlit app
st.title('Handwritten Digit Recognition with CNN')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make predictions
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    
    st.write(f"Predicted Digit: {predicted_digit}")

    # Get feature maps
    feature_maps = get_feature_maps(model, processed_image)

    # Plot feature maps
    layer_names = [layer.name for layer in model.layers if isinstance(layer, Conv2D) or isinstance(layer, MaxPooling2D)]
    
    for layer_name, layer_activation in zip(layer_names, feature_maps):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        fig, axes = plt.subplots(1, n_features, figsize=(15, 15))
        for i in range(n_features):
            ax = axes[i]
            ax.imshow(layer_activation[0, :, :, i], cmap='viridis')
            ax.axis('off')
        st.write(f"Feature maps of layer: {layer_name}")
        st.pyplot(fig)

# Run the Streamlit app
if __name__ == '__main__':
    st._is_running_with_streamlit = True
    st.run()
