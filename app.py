import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from fpdf import FPDF
import gdown

# १. पेज सेटिंग्ज
st.set_page_config(page_title="Sugarcane Disease AI", layout="wide")

# २. मॉडेल डाउनलोड फंक्शन (Google Drive वरून)
@st.cache_resource
def load_model_from_drive():
    file_id = '1BN12K8BnYULv5X_nNQ8kQTYSLN_OZ_DI'
    output = 'sugarcane_model.h5'
    if not os.path.exists(output):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)
    return tf.keras.models.load_model(output)

# मॉडेल लोड करणे
try:
    model = load_model_from_drive()
except Exception as e:
    st.error("मॉडेल लोड करताना त्रुटी आली. कृपया Drive परमिशन तपासा.")
    model = None

# ३. मुख्य UI
st.title("🌾 Sugarcane Disease Mapping & Advisory")
st.write("M.Sc. Geoinformatics Project: Precision Agriculture Tool")

uploaded_file = st.file_uploader("ड्रोन ऑर्थोमोझॅक किंवा शेताचा फोटो अपलोड करा...", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # विश्लेषण (टायलिंग)
    tile_size = 224
    width, height = image.size
    cols, rows = width // tile_size, height // tile_size
    
    detected_diseases = []
    classes = ['Healthy', 'Bacterial Blight', 'Red Rot']
    
    st.info("विश्लेषण सुरू आहे...")
    progress_bar = st.progress(0)
    
    count = 0
    for r in range(rows):
        for c in range(cols):
            left, top = c * tile_size, r * tile_size
            tile = image.crop((left, top, left + tile_size, top + tile_size))
            
            # Prediction
            img_array = np.array(tile.resize((224, 224))) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array, verbose=0)
            res_idx = np.argmax(prediction)
            
            if res_idx > 0:
                # काल्पनिक GIS को-ऑर्डिनेट्स (प्रोजेक्टसाठी)
                lat, lon = 18.5204 + (r * 0.0003), 73.8567 + (c * 0.0003)
                detected_diseases.append({
                    "तुकडा": f"R{r+1}C{c+1}",
                    "रोग": classes[res_idx],
                    "lat": lat, "lon": lon
                })
            count += 1
            progress_bar.progress(count / (rows * cols))

    # ४. मॅप आणि रिपोर्ट दाखवणे
    st.success("विश्लेषण पूर्ण झाले!")
    m = folium.Map(location=[18.5204, 73.8567], zoom_start=18)
    
    if detected_diseases:
        heat_data = [[d["lat"], d["lon"]] for d in detected_diseases]
        HeatMap(heat_data).add_to(m)
        for d in detected_diseases:
            folium.Marker([d["lat"], d["lon"]], popup=d["रोग"]).add_to(m)
        
        st_folium(m, width=700, height=450)
        st.table(detected_diseases)
    else:
        st.success("शेतात कोणताही रोग आढळला नाही!")
        st_folium(m, width=700, height=450)
