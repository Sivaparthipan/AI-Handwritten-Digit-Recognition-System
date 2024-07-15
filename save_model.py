import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model from the HDF5 file
model = load_model('mnist_cnn_improved.h5')

# Save the model in the new .keras format
model.save('mnist_cnn_improved.keras')
