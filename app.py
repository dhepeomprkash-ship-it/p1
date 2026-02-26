import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from fpdf import FPDF
import urllib.request

# --- १. पेज कॉन्फिगरेशन ---
st.set_page_config(page_title="Sugarcane AI Mapper", layout="wide")

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
    pdf.cell(200, 10, txt="Sugarcane Disease Surveillance Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for d in data:
        pdf.cell(200, 10, txt=f"- {d['तुकडा']}: {d['रोग']} (Lat: {d['lat']:.4f}, Lon: {d['lon']:.4f})", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# Hugging Face वरून मॉडेल लोड करणे
@st.cache_resource
def load_model_permanent():
    url = "https://huggingface.co/datasets/omdhepe/smodel/resolve/main/sugarcane_model.h5?download=true"
    output = "sugarcane_model.h5"
    if not os.path.exists(output):
        with st.spinner('Hugging Face वरून मॉडेल लोड होत आहे... कृपया थोडा वेळ थांबा.'):
            urllib.request.urlretrieve(url, output)
    return tf.keras.models.load_model(output)

# --- ४. मुख्य प्रोग्राम ---
st.title("🌾 Sugarcane Disease Mapping & Advisory System")
st.write("M.Sc. Geoinformatics Project | Powered by Hugging Face & Streamlit")

try:
    model = load_model_permanent()
except Exception as e:
    st.error(f"मॉडेल लोड करताना चूक झाली: {e}")
    model = None

uploaded_file = st.file_uploader("ड्रोन ऑर्थोमोझॅक किंवा फोटो अपलोड करा...", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Imagery', use_container_width=True)
    
    # टायलिंग लॉजिक
    width, height = image.size
    tile_size = 224
    cols, rows = width // tile_size, height // tile_size
    
    detected_diseases = []
    
    st.info(f"विश्लेषण सुरू आहे: एकूण {rows * cols} तुकड्यांची तपासणी होत आहे...")
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
                lat, lon = 18.5204 + (r * 0.0003), 73.8567 + (c * 0.0003)
                detected_diseases.append({
                    "तुकडा": f"Tile R{r+1}C{c+1}",
                    "रोग": classes[res_idx],
                    "lat": lat, "lon": lon
                })
            count += 1
            progress_bar.progress(count / (rows * cols))

    # ५. रिझल्ट्स आणि मॅपिंग
    st.success("विश्लेषण पूर्ण झाले!")
    m = folium.Map(location=[18.5204, 73.8567], zoom_start=18, 
                   tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', 
                   attr='Google Satellite Hybrid')

    if detected_diseases:
        # Heatmap
        heat_data = [[d["lat"], d["lon"]] for d in detected_diseases]
        HeatMap(heat_data, radius=15, blur=10).add_to(m)
        
        # Marketers
        for d in detected_diseases:
            folium.Marker([d["lat"], d["lon"]], popup=d["रोग"], 
                          icon=folium.Icon(color='red', icon='info-sign')).add_to(m)
        
        st_folium(m, width=900, height=500)

        # ६. ॲडव्हायझरी आणि रिपोर्ट (खालील मांडणी आता अचूक आहे)
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("🌱 कृषी सल्ला")
            found_diseases = set([d["रोग"] for d in detected_diseases])
            for dis in found_diseases:
                if dis in advisory_map:
                    with st.expander(f"🚩 {dis} उपाय"):
                        st.write(f"💊 **औषध:** {advisory_map[dis]['औषध']}")
                        st.write(f"📢 **सल्ला:** {advisory_map[dis]['सल्ला']}")
        
        with col2:
            st.header("📥 अहवाल")
            pdf_bytes = create_pdf(detected_diseases)
            st.download_button("Download PDF Report", data=pdf_bytes, file_name="Health_Report.pdf")

        st.subheader("📋 तपशीलवार माहिती")
        st.table(detected_diseases)
    else:
        st.balloons()
        st.success("शेतात कोणताही रोग आढळला नाही!")
        st_folium(m, width=900, height=500)
