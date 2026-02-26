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

# --- १. पेज कॉन्फिगरेशन ---
st.set_page_config(page_title="Sugarcane Disease AI Mapping", layout="wide")

# --- २. कृषी सल्ला डेटा ---
advisory_map = {
    "Bacterial Blight": {
        "औषध": "Streptocycline (100 ppm) + Copper Oxychloride (0.25%)",
        "सल्ला": "बाधित पाने कापून नष्ट करा. नत्राचा (Nitrogen) वापर टाळा."
    },
    "Red Rot": {
        "औषध": "Carbendazim (0.1%) किंवा Trichoderma viride",
        "सल्ला": "पाण्याचा निचरा सुधारा. बाधित खुंट उपटून टाका. बेणे प्रक्रिया करा."
    }
}

classes = ['Healthy', 'Bacterial Blight', 'Red Rot']

# --- ३. उपयुक्त फंक्शन्स ---

# PDF रिपोर्ट तयार करणे
def create_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Sugarcane Disease Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    for d in data:
        pdf.cell(200, 10, txt=f"- {d['तुकडा']}: {d['रोग']} (Lat: {d['lat']}, Lon: {d['lon']})", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# मॉडेल डाउनलोड आणि लोड करणे
@st.cache_resource
def load_my_model():
    file_id = '1BN12K8BnYULv5X_nNQ8kQTYSLN_OZ_DI' # तुमचा खरा Google Drive ID
    output = 'sugarcane_model.h5'
    if not os.path.exists(output):
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        gdown.download(url, output, quiet=False)
    
    return tf.keras.models.load_model(output)

# --- ४. मुख्य प्रोग्राम ---
st.title("🌾 Sugarcane Disease Mapping & Advisory System")
st.write("M.Sc. Geoinformatics Project: Drone Imagery & Deep Learning")

try:
    model = load_my_model()
except Exception as e:
    st.error(f"मॉडेल लोड करताना त्रुटी आली: {e}")
    model = None

uploaded_file = st.file_uploader("ड्रोन ऑर्थोमोझॅक फोटो अपलोड करा...", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Drone Imagery', use_container_width=True)
    
    # टायलिंगसाठी सेटिंग्ज
    width, height = image.size
    tile_size = 224
    cols = width // tile_size
    rows = height // tile_size
    
    detected_diseases = [] # NameError टाळण्यासाठी इथे लिस्ट तयार केली
    
    st.info(f"विश्लेषण सुरू आहे: एकूण {rows * cols} तुकड्यांची तपासणी होत आहे...")
    progress_bar = st.progress(0)
    current_tile = 0

    # ५. प्रोसेसिंग लूप
    for r in range(rows):
        for c in range(cols):
            left = c * tile_size
            top = r * tile_size
            right = left + tile_size
            bottom = top + tile_size
            
            tile_img = image.crop((left, top, right, bottom))
            
            # मॉडेल प्रेडिक्शन
            img_array = np.array(tile_img.resize((224, 224))) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            if model:
                prediction = model.predict(img_array, verbose=0)
                result_index = np.argmax(prediction)
                
                if result_index > 0: # जर Healthy नसेल तर
                    # काल्पनिक लोकेशन लॉजिक (M.Sc. Project साठी)
                    lat = 18.5204 + (r * 0.0005)
                    lon = 73.8567 + (c * 0.0005)
                    
                    detected_diseases.append({
                        "तुकडा": f"Row {r+1}, Col {c+1}",
                        "रोग": classes[result_index],
                        "lat": lat,
                        "lon": lon
                    })
            
            current_tile += 1
            progress_bar.progress(current_tile / (rows * cols))

    # ६. रिझल्ट्स आणि नकाशा (Mapping)
    st.success("विश्लेषण पूर्ण झाले!")

    # नकाशा बेस
    m = folium.Map(location=[18.5204, 73.8567], zoom_start=17, tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Satellite')

    if detected_diseases:
        # --- हीटमॅप जोडणे ---
        heat_data = [[d["lat"], d["lon"]] for d in detected_diseases]
        HeatMap(heat_data, radius=15, blur=10).add_to(m)
        
        # --- मार्कर्स जोडणे ---
        for d in detected_diseases:
            folium.Marker(
                [d["lat"], d["lon"]],
                popup=f"{d['तुकडा']}: {d['रोग']}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)

        # नकाशा दाखवा
        st_folium(m, width=900, height=500)

        # ७. ॲडव्हायझरी आणि रिपोर्ट
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("🌱 कृषी सल्ला")
            found_diseases = set([d["रोग"] for d in detected_diseases])
            for dis in found_diseases:
                if dis in advisory_map:
                    with st.expander(f"🚩 {dis} साठी उपाय"):
                        st.write(f"💊 **औषध:** {advisory_map[dis]['औषध']}")
                        st.write(f"📢 **सल्ला:** {advisory_map[dis]['सल्ला']}")
        
        with col2:
            st.header("📥 रिपोर्ट")
            pdf_bytes = create_pdf(detected_diseases)
            st.download_button("Download PDF Report", data=pdf_bytes, file_name="Report.pdf")

        st.subheader("📋 तपशीलवार माहिती")
        st.table(detected_diseases)
    else:
        st.balloons()
        st.success("तुमचे शेत निरोगी आहे! नकाशावर कोणताही रोग आढळला नाही.")
        st_folium(m, width=900, height=500)
