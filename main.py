# CNN
# Importing the libraries
import streamlit as st
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array, save_img
import sys
import os
import numpy as np
from PIL import Image

# Stating the path to the trained model
def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

cnn = resource_path('model.h5')

# Title and description
st.title('Image Classification Model')
st.write("""
Utilizing a Convolutional Neural Network model, we can detect if it is cat or a dog.
""")
st.write('---')

# Creating the image selectbox
img_chosen = st.sidebar.selectbox(
    'Select an Image',
    ('cat1.jpg', 'cat2.jpg', 'cat3.jpg', 'cat4.jpg', 'cat5.jpg', 'cat6.jpg', 'dog1.jpg', 'dog2.jpg', 'dog3.jpg',
     'dog4.jpg', 'dog5.jpg', 'dog6.jpg')
)
input_image = "images/content-images/" + img_chosen
st.write('### Selected image:')
image = Image.open(input_image)
st.image(image, width=400)  # image: numpy array

# Making a single prediction
# Creating the running button
clicked = st.button('Predict')

# Our working model
def prediction(testing_image_path):
    model = keras.models.load_model(cnn)
    test_image = load_img(testing_image_path, target_size=(64, 64))

    # This lines of code is to convert the image into an array
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    return result

def generate_result(result):
    if result[0][0] == 1:
        st.write("""
        The model predicts it is a dog.
        """)
    else:
        st.write("""
        The model predicts it is a cat.
        """)


# Clicking rule
if clicked:
    final_prediction = prediction(input_image)
    generate_result(final_prediction)

# Now lets give the user a chance to upload their own images
st.title('Use your own cat or dog images!')
st.write("""
The model seems good, but there are cats that looks similar to a dog and vice versa.
""")
st.write("""
So you have the opportunity to upload your own images and try my CNN prediction model.
""")

# Image user browser/uploader
img_file_buffer = st.file_uploader("Upload an image here")
try:
    image2 = Image.open(img_file_buffer)
    test_image2 = np.array(image2)
    st.write('### Selected image:')
    st.image(image2, width=400)  # image: numpy array
except:
    st.write("""
		### Browse for an image inside your PC
		""")

# Making a single prediction
# Creating the running button
clicked2 = st.button('Predict your own image')

# Our working model
def prediction_2(testing_image_path_2):
    model = keras.models.load_model(cnn)
    test_image2 = load_img(testing_image_path_2, target_size=(64, 64))
    # This lines of code is to convert the image into an array
    test_image2 = img_to_array(test_image2)
    test_image2 = np.expand_dims(test_image2, axis=0)
    result2 = model.predict(test_image2)
    return result2

def generate_result(result2):
    if result2[0][0] == 1:
        st.write("""
        The model predicts it is a dog.
        """)
    else:
        st.write("""
        The model predicts it is a cat.
        """)

# Clicking rule
if clicked2:
    try:
        save_img("temp_dir/test_image.jpg", test_image2)
        image_path = "temp_dir/test_image.jpg"
        final_prediction2 = prediction_2(image_path)
        generate_result(final_prediction2)
    except:
        st.write("""
    		### Something is wrong...
    		""")