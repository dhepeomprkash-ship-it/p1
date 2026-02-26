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
st.set_page_config(page_title="Sugarcane Disease AI Mapper", layout="wide")

# --- २. कृषी सल्ला आणि डेटा ---
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

# मॉडेल लोड करणे (Download via Google Drive)
@st.cache_resource
def load_my_model():
    # तुमची फाईल पब्लिक असल्याची खात्री करा (Anyone with the link can view)
    file_id = '1BN12K8BnYULv5X_nNQ8kQTYSLN_OZ_DI' 
    output = 'sugarcane_model.h5'
    
    if not os.path.exists(output):
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, output, quiet=False)
        except Exception as e:
            st.error(f"मॉडेल डाउनलोड करता आले नाही. कृपया गुगल ड्राइव्हची फाईल 'Public' करा.")
            return None
            
    return tf.keras.models.load_model(output)

# --- ४. मुख्य UI ---
st.title("🌾 Sugarcane Disease Mapping & Advisory System")
st.write("M.Sc. Geoinformatics Project")

model = load_my_model()

uploaded_file = st.file_uploader("ड्रोन ऑर्थोमोझॅक किंवा फोटो अपलोड करा...", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # टायलिंग लॉजिक
    width, height = image.size
    tile_size = 224
    cols, rows = width // tile_size, height // tile_size
    
    detected_diseases = [] # NameError फिक्स
    
    st.info(f"विश्लेषण सुरू आहे: {rows * cols} तुकड्यांची तपासणी होत आहे...")
    progress_bar = st.progress(0)
    count = 0

    if model:
        for r in range(rows):
            for c in range(cols):
                left, top = c * tile_size, r * tile_size
                tile = image.crop((left, top, left + tile_size, top + tile_size))
                
                # प्रेडिक्शन
                img_array = np.array(tile.resize((224, 224))) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                prediction = model.predict(img_array, verbose=0)
                result_index = np.argmax(prediction)
                
                if result_index > 0: # जर रोग असेल तर
                    lat, lon = 18.5204 + (r * 0.0003), 73.8567 + (c * 0.0003)
                    detected_diseases.append({
                        "तुकडा": f"Tile R{r+1}C{c+1}",
                        "रोग": classes[result_index],
                        "lat": lat, "lon": lon
                    })
                
                count += 1
                progress_bar.progress(count / (rows * cols))

        # ५. निकाल दाखवणे
        st.success("विश्लेषण पूर्ण झाले!")
        m = folium.Map(location=[18.5204, 73.8567], zoom_start=18, 
                       tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', 
                       attr='Google Satellite Hybrid')

        if detected_diseases:
            # Heatmap आणि मार्कर्स
            heat_data = [[d["lat"], d["lon"]] for d in detected_diseases]
            HeatMap(heat_data, radius=15, blur=10).add_to(m)
            
            for d in detected_diseases:
                folium.Marker([d["lat"], d["lon"]], popup=d["रोग"], 
                              icon=folium.Icon(color='red')).add_to(m)
            
            st_folium(m, width=900, height=500)

            # ६. ॲडव्हायझरी आणि रिपोर्ट
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.header("🌱 कृषी सल्ला")
                unique_found = set([d["रोग"] for d in detected_diseases])
                for f in unique_found:
                    if f in advisory_map:
                        with st.expander(f"🚩 {f} उपाय"):
                            st.write(f"💊 **औषध:** {advisory_map[f]['औषध']}")
                            st.write(f"📢 **सल्ला:** {advisory_map[f]['सल्ला']}")
            
            with col2:
                st.header("📥 रिपोर्ट")
                pdf_bytes = create_pdf(detected_diseases)
                st.download_button("Download PDF Report", data=pdf_bytes, file_name="Health_Report.pdf")
            
            st.table(detected_diseases)
        else:
            st.balloons()
            st.success("शेतात कोणताही रोग आढळला नाही!")
            st_folium(m, width=900, height=500)
