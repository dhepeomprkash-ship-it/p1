import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

st.set_page_config(page_title="Sugarcane Disease AI", layout="wide")
st.title("🌱 Sugarcane Disease Detection (उसावरील रोग ओळखणे)")

# १. तुमचा अचूक Google Drive ID इथे टाका
file_id = 'तुमचा_Google_Drive_ID_इथे_टाका'
url = f'https://drive.google.com/uc?export=download&id={file_id}'
output = 'sugarcane_model.h5'

# मॉडेल डाऊनलोड आणि लोड करणे
if not os.path.exists(output):
    with st.spinner('AI मॉडेल लोड होत आहे...'):
        gdown.download(url, output, quiet=False)

model = tf.keras.models.load_model(output)

# फोटो अपलोड बटण
uploaded_file = st.file_uploader("उसाच्या पानाचा स्वच्छ फोटो अपलोड करा...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='अपलोड केलेला फोटो', width=400)
    
    # इमेज प्रोसेसिंग
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # प्रेडिक्शन (निकाल)
    prediction = model.predict(img_array)
    result_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    # २. नावे तुमच्या ट्रेनिंग फोल्डर्सच्या क्रमानुसार बदला
    classes = ['Healthy (निरोगी)', 'Bacterial Blight (जीवाणूजन्य करपा)', 'Red Rot (लाल कुज)']
    
    st.markdown("---")
    st.subheader("तपासणीचा निकाल:")
    
    if result_index == 0:
        st.success(f"✅ ऊस निरोगी (Healthy) आहे! (खात्री: {confidence:.2f}%)")
    else:
        st.error(f"⚠️ उसावर **{classes[result_index]}** आढळला आहे! (खात्री: {confidence:.2f}%)")
        
        # बी.एस्सी. ॲग्रीकल्चर सल्ला
        if result_index == 1:
            st.info("💡 सल्ला: कॉपर ऑक्सिक्लोराईडची फवारणी करा आणि शेतात पाण्याचा निचरा व्यवस्थित ठेवा.")
        elif result_index == 2:
            st.info("💡 सल्ला: बाधित झाडे उपटून जाळून टाका. बेणे प्रक्रियेसाठी कार्बेन्डाझिम वापरा.")
