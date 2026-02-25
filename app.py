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
    image = Image.open(uploaded_file)
    st.image(image, caption='मूळ फोटो (Original Image)', width=500)
    
    # इमेजची साईज मिळवा
    width, height = image.size
    mid_x, mid_y = width // 2, height // 2
    
    # ४ तुकड्यांचे बॉक्स
    tiles = [(0, 0, mid_x, mid_y), (mid_x, 0, width, mid_y), 
             (0, mid_y, mid_x, height), (mid_x, mid_y, width, height)]
    
    st.markdown("---")
    st.subheader("🔍 तुकड्यांनुसार विश्लेषण:")
    cols = st.columns(2) 
    
    classes = ['Healthy (निरोगी)', 'Bacterial Blight', 'Red Rot']
    
    # --- नवीन भाग: सापडलेले रोग साठवण्यासाठी लिस्ट ---
    detected_diseases = []
    
    # काल्पनिक लोकेशन्स (M.Sc. Project साठी)
    mock_locations = [
        {"lat": 18.5204, "lon": 73.8567},
        {"lat": 18.5250, "lon": 73.8600},
        {"lat": 18.5180, "lon": 73.8520},
        {"lat": 18.5280, "lon": 73.8650}
    ]

    for i, box in enumerate(tiles):
        tile_img = image.crop(box)
        resized_tile = tile_img.resize((224, 224))
        img_array = np.array(resized_tile) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        result_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
       # --- सुधारित निकाल दाखवणे (Logic with elif) ---
        with cols[i % 2]:
            st.image(tile_img, caption=f"तुकडा {i+1}", use_container_width=True)
            
            if result_index == 0:
                st.success(f"तुकडा {i+1}: सुरक्षित (Healthy)")
            
            elif result_index == 2:
                # जर इंडेक्स २ असेल तर तो Red Rot आहे
                st.error(f"तुकडा {i+1}: 🚩 Red Rot आढळला!")
                detected_diseases.append({
                    "name": f"तुकडा {i+1}: Red Rot",
                    "lat": mock_locations[i]["lat"],
                    "lon": mock_locations[i]["lon"]
                })
            
            else:
                # इंडेक्स १ साठी Bacterial Blight
                st.warning(f"तुकडा {i+1}: Bacterial Blight आढळला")
                detected_diseases.append({
                    "name": f"तुकडा {i+1}: Bacterial Blight",
                    "lat": mock_locations[i]["lat"],
                    "lon": mock_locations[i]["lon"]
                })

    # --- नकाशाचा भाग (ओळ ९१ च्या आसपास पेस्ट करा) ---
    st.markdown("---")
    st.header("🗺️ Disease Mapping (Spatial Distribution)")
    
    # १. नकाशाचा बेस तयार करा
    m = folium.Map(location=[18.5204, 73.8567], zoom_start=14)
    
    # २. जर लिस्टमध्ये रोग आढळले असतील, तरच मार्कर लावा
    if detected_diseases:
        for d in detected_diseases:
            folium.Marker(
                [d["lat"], d["lon"]],
                popup=d["name"],
                icon=folium.Icon(color='red')
            ).add_to(m)
        st_folium(m, width=700, height=450)
    else:
        # ३. जर रोग नसेल तर नुसता नकाशा आणि यशाचा मेसेज दाखवा
        st.success("अभिनंदन! शेतात कुठेही रोग आढळला नाही.")
        st_folium(m, width=700, height=450)
