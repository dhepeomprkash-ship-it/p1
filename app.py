import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# १. तुमचा अचूक Google Drive File ID
file_id = '1BN12K8BnyULv5X_nNQ8kQTYSLN_OZ_DI'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'sugarcane_model.h5'

# मॉडेल नसल्यास डाऊनलोड करा
if not os.path.exists(output):
    with st.spinner('AI मॉडेल लोड होत आहे, कृपया थांबा...'):
        gdown.download(url, output, quiet=False)

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
    img_array = np.expand_dims(img_array, axis=0

    # तुमच्या ट्रेनिंगच्या क्रमानुसार ही नावे तपासा (उदा. 0, 1, 2)
classes = ['Healthy (निरोगी)', 'Bacterial Blight (जीवाणूजन्य करपा)', 'Red Rot (लाल कुज)']

prediction = model.predict(img_array)
result_index = np.argmax(prediction)
confidence = np.max(prediction) * 100

st.subheader("तपासणीचा निकाल:")

if result_index == 0:
    st.success(f"तुमचा ऊस निरोगी आहे! (खात्री: {confidence:.2f}%)")
else:
    st.error(f"सावधान! उसावर **{classes[result_index]}** आढळला आहे. (खात्री: {confidence:.2f}%)")
    
    # बी.एस्सी. ॲग्रीकल्चरच्या ज्ञानानुसार छोटा सल्ला
    if result_index == 2: # Red Rot साठी
        st.warning("सल्ला: बाधित झाडे मुळासकट उपटून नष्ट करा आणि बोर्डो मिश्रणाची फवारणी करा.")
    
   
