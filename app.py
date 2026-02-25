import streamlit as st
import folium
from streamlit_folium import st_folium
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

st.set_page_config(page_title="Sugarcane Disease AI", layout="wide")
st.title("🌱 Sugarcane Disease Detection (उसावरील रोग ओळखणे)")

# १. तुमचा अचूक Google Drive ID इथे टाका
file_id = '1BN12K8BnyULv5X_nNQ8kQTYSLN_OZ_DI'
url = f'https://drive.google.com/uc?export=download&id={file_id}'
output = 'sugarcane_model.h5'

# मॉडेल डाऊनलोड आणि लोड करणे
if not os.path.exists(output):
    with st.spinner('AI मॉडेल लोड होत आहे...'):
        gdown.download(url, output, quiet=False)

model = tf.keras.models.load_model(output)

# हा डेटा आपण नंतर ड्रोन इमेजमधून ऑटोमॅटिकली काढणार आहोत
disease_locations = [
    {"lat": 18.5204, "lon": 73.8567, "name": "Red Rot - Area 1"},
    {"lat": 18.5250, "lon": 73.8600, "name": "Bacterial Blight - Area 2"}
]

# फोटो अपलोड बटण
uploaded_file = st.file_uploader("उसाच्या पानाचा स्वच्छ फोटो अपलोड करा...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # --- प्रतिमेचे तुकडे (Tiling) सुरू ---
    image = Image.open(uploaded_file)
    st.image(image, caption='मूळ फोटो (Original Image)', width=500)
    
    width, height = image.size
    mid_x, mid_y = width // 2, height // 2
    
    # ४ तुकड्यांचे बॉक्स
    tiles = [
        (0, 0, mid_x, mid_y),       # वरचा डावा
        (mid_x, 0, width, mid_y),    # वरचा उजवा
        (0, mid_y, mid_x, height),   # खालचा डावा
        (mid_x, mid_y, width, height) # खालचा उजवा
    ]
    
    st.markdown("---")
    st.subheader("🔍 तुकड्यांनुसार विश्लेषण (Tile-based Analysis):")
    cols = st.columns(2) 
    
    classes = ['Healthy (निरोगी)', 'Bacterial Blight', 'Red Rot']

    for i, box in enumerate(tiles):
        tile_img = image.crop(box)
        
        # AI मॉडेलसाठी प्रोसेसिंग
        resized_tile = tile_img.resize((224, 224))
        img_array = np.array(resized_tile) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        result_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        with cols[i % 2]:
            st.image(tile_img, caption=f"तुकडा {i+1}", use_container_width=True)
            if result_index == 0:
                st.success(f"तुकडा {i+1}: सुरक्षित ({confidence:.1f}%)")
            else:
                st.error(f"तुकडा {i+1}: {classes[result_index]} आढळला! ({confidence:.1f}%)")


st.markdown("---")
st.header("📍 Disease Hotspots (नकाशावर आधारित विश्लेषण)")

# १. नकाशाचा केंद्रबिंदू ठरवा
m = folium.Map(location=[18.5204, 73.8567], zoom_start=14)

# २. लूप वापरून प्रत्येक रोगाच्या ठिकाणावर मार्कर लावा
for loc in disease_locations:
    folium.Marker(
        [loc["lat"], loc["lon"]], 
        popup=loc["name"],
        icon=folium.Icon(color='red' if "Red Rot" in loc["name"] else 'orange')
    ).add_to(m)

# ३. नकाशा वेबसाईटवर दाखवा
st_folium(m, width=800, height=500)
