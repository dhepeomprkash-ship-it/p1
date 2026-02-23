import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# १. तुमचा Google Drive File ID इथे टाका
file_id = '1A2B3C4D5E6F7G8H9I0J'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'sugarcane_model.h5'

# मॉडेल नसल्यास डाउनलोड करा
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)
st.title("उसावरील रोग ओळखणे (Sugarcane Disease AI)")

# मॉडेल लोड करणे
model = tf.keras.models.load_model('sugarcane_model.h5')

# फोटो अपलोड करण्याचे बटण
uploaded_file = st.file_uploader("उसाच्या पानाचा फोटो निवडा...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='अपलोड केलेला फोटो', use_column_width=True)
    st.write("ओळखत आहे...")
    
    # इमेज प्रोसेसिंग
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # निकाल
    prediction = model.predict(img_array)
    if np.argmax(prediction) == 0:
        st.success("निकाल: ऊस निरोगी (Healthy) आहे!")
    else:
        st.error("निकाल: उसावर रोग (Diseased) आढळला आहे!")
