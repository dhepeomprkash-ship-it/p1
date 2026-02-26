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
st.set_page_config(page_title="Sugarcane AI Mapper", layout="wide")

# --- २. कृषी सल्ला डेटा (Advisory Data) ---
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

# --- ३. उपयुक्त फंक्शन्स (Functions) ---

# PDF रिपोर्ट तयार करणे
def create_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Sugarcane Disease Surveillance Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for d in data:
        pdf.cell(200, 10, txt=f"- {d['तुकडा']}: {d['रोग']} (Lat: {d['lat']:.4f}, Lon: {d['lon']:.4f})", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# मॉडेल लोड करणे
@st.cache_resource
def load_my_model():
    file_id = 1BN12K8BnyULv5X_nNQ8kQTYSLN_OZ_DI
    # तुमचा Drive ID
    output = 'sugarcane_model.h5'
    if not os.path.exists(output):
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        gdown.download(url, output, quiet=False)
    return tf.keras.models.load_model(output)

# --- ४. मुख्य युजर इंटरफेस (UI) ---
st.title("🌾 Sugarcane Disease Mapping & Advisory System")
st.write("M.Sc. Geoinformatics Project | Precision Agriculture")

try:
    model = load_my_model()
except Exception as e:
    st.error(f"मॉडेल लोड करताना त्रुटी: {e}")
    model = None

uploaded_file = st.file_uploader("ड्रोन ऑर्थोमोझॅक फोटो अपलोड करा...", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Field Image', use_container_width=True)
    
    # टायलिंग लॉजिक
    width, height = image.size
    tile_size = 224
    cols, rows = width // tile_size, height // tile_size
    
    # १. नेम एरर टाळण्यासाठी लिस्ट आधीच घोषित करा
    detected_diseases = []
    
    st.info(f"विश्लेषण सुरू आहे: {rows * cols} तुकड्यांची तपासणी होत आहे...")
    progress_bar = st.progress(0)
    count = 0

    # ५. प्रोसेसिंग लूप (Indentation नीट तपासले आहे)
    for r in range(rows):
        for c in range(cols):
            left, top = c * tile_size, r * tile_size
            tile = image.crop((left, top, left + tile_size, top + tile_size))
            
            # प्रेडिक्शन
            img_array = np.array(tile.resize((224, 224))) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            if model:
                prediction = model.predict(img_array, verbose=0)
                result_index = np.argmax(prediction)
                
                if result_index > 0:
                    # काल्पनिक मॅपिंग लॉजिक
                    lat, lon = 18.5204 + (r * 0.0003), 73.8567 + (c * 0.0003)
                    detected_diseases.append({
                        "तुकडा": f"Tile R{r+1}C{c+1}",
                        "रोग": classes[result_index],
                        "lat": lat, "lon": lon
                    })
            
            count += 1
            progress_bar.progress(count / (rows * cols))

    # ६. नकाशा आणि रिझल्ट्स
    st.success("विश्लेषण पूर्ण झाले!")
    m = folium.Map(location=[18.5204, 73.8567], zoom_start=18, 
                   tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', 
                   attr='Google Satellite')

    if detected_diseases:
        # हीटमॅप आणि मार्कर्स
        heat_data = [[d["lat"], d["lon"]] for d in detected_diseases]
        HeatMap(heat_data, radius=15, blur=10).add_to(m)
        
        for d in detected_diseases:
            folium.Marker([d["lat"], d["lon"]], popup=d["रोग"], 
                          icon=folium.Icon(color='red')).add_to(m)
        
        st_folium(m, width=900, height=500)

        # ७. ॲडव्हायझरी आणि रिपोर्ट (सर्वात शेवटी)
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("🌱 कृषी सल्ला")
            found = set([d["रोग"] for d in detected_diseases])
            for f in found:
                if f in advisory_map:
                    with st.expander(f"🚩 {f} उपाय"):
                        st.write(f"💊 **औषध:** {advisory_map[f]['औषध']}")
                        st.write(f"📢 **सल्ला:** {advisory_map[f]['सल्ला']}")
        
        with col2:
            st.header("📥 अहवाल")
            pdf_bytes = create_pdf(detected_diseases)
            st.download_button("Download PDF Report", data=pdf_bytes, file_name="Health_Report.pdf")

        st.subheader("📋 तपशील")
        st.table(detected_diseases)
    else:
        st.balloons()
        st.success("शेतात कुठेही रोग आढळला नाही!")
        st_folium(m, width=900, height=500)
