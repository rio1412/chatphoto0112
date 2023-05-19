import streamlit as st
import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# Constants
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

# Load the model
def load_model():
    model = hub.load(SAVED_MODEL_PATH)
    return model

model = load_model()

# Preprocess the input image to make it model ready
def preprocess_image(image):
  hr_image = np.array(image)
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

# Save the output image
def save_image(image, filename):
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save("%s.jpg" % filename)

# Plot the output image
def plot_image(image, title=""):
  image = np.asarray(image)
  image = tf.clip_by_value(image, 0, 255)
  image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  st.image(image, caption=title, use_column_width=True)

# Main function
def app():
  # Add title to the app
  st.title("Super Resolution")

  # Add sidebar to the app
  st.sidebar.title("Settings")
  contrast = st.sidebar.slider('Contrast', min_value=0.0, max_value=2.0, value=1.0, step=0.1)
  st.sidebar.write('Contrast:', contrast)
  brightness = st.sidebar.slider('Brightness', min_value=-0.5, max_value=0.5, value=0.0, step=0.05)
  st.sidebar.write('Brightness:', brightness)
  gamma = st.sidebar.slider('Gamma', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
  st.sidebar.write('Gamma:', gamma)
  hue = st.sidebar.slider('Hue', min_value=-0.5, max_value=0.5, value=0.0, step=0.05)
  st.sidebar.write('Hue:', hue)

  # Add image uploader to the app
  image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

  # If an image is uploaded
  if image_file is not None:
      # Display the original image
      input_image = Image.open(image_file)
      st.image(input_image, caption="Original Image", use_column_width=True)
      
      # Preprocess the input image for the model
      hr_image = preprocess_image(input_image)
      
      # If the user clicks the "Super Resolution" button
      if st.button('Super Resolution'):
          if hr_image is not None:
              # Generate the high resolution image using the ESRGAN model
              fake_image = model(hr_image)

              # Display the super resolution image
              st.image(fake_image[0], caption="Super Resolution Image", use_column_width=True)

              # Save the super resolution image
              save_image(fake_image[0], "super_resolution_image")
        
  # If no image is uploaded
  else:
      st.write("Please upload an image to use the app.")


