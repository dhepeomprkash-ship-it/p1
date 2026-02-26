import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from fpdf import FPDF

# --- १. पेज कॉन्फिगरेशन ---
st.set_page_config(page_title="Sugarcane Disease AI Mapper", layout="wide")

# --- २. कृषी सल्ला आणि डेटा (Advisory Data) ---
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

# --- ३. उपयुक्त फंक्शन्स (Helper Functions) ---

# PDF रिपोर्ट तयार करणे
def create_pdf(data, total_tiles):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Sugarcane Disease Surveillance Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Total Areas Scanned: {total_tiles}", ln=True)
    pdf.cell(200, 10, txt=f"Infected Hotspots Found: {len(data)}", ln=True)
    pdf.ln(5)
    for d in data:
        pdf.cell(200, 10, txt=f"- {d['तुकडा']}: {d['रोग']} (Lat: {d['lat']:.4f}, Lon: {d['lon']:.4f})", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# मॉडेल लोड करणे
@st.cache_resource
def load_my_model():
    # तुमच्या मॉडेलची फाईल इथे असावी (उदा. model.h5)
    # जर मॉडेल नसेल तर हा भाग एरर देऊ शकतो, तिथे तुमच्या मॉडेलचा खरा पाथ द्या
    try:
        model = tf.keras.models.load_model('sugarcane_model.h5')
        return model
    except:
        st.error("मॉडेल फाईल (sugarcane_model.h5) सापडली नाही!")
        return None

model = load_my_model()

# --- ४. मुख्य युजर इंटरफेस (UI) ---
st.title("🌾 Sugarcane Disease Mapping & Advisory System")
st.write("M.Sc. Geoinformatics Project: Precision Agriculture Tool")

uploaded_file = st.file_uploader("ड्रोन ऑर्थोमोझॅक किंवा शेताचा फोटो अपलोड करा...", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Field Image', use_container_width=True)
    
    # इमेज प्रोसेसिंगची तयारी
    img_array_full = np.array(image)
    width, height = image.size
    tile_size = 224  # तुमच्या मॉडेलचा इनपुट साईज
    
    cols = width // tile_size
    rows = height // tile_size
    total_tiles = cols * rows
    
    detected_diseases = []
    
    # ५. स्वयंचलित टायलिंग लूप (Automated Tiling)
    st.info(f"तुमच्या फोटोचे {total_tiles} तुकड्यांत विश्लेषण होत आहे...")
    progress_bar = st.progress(0)
    count = 0

    for r in range(rows):
        for c in range(cols):
            # तुकडा कापा
            left = c * tile_size
            top = r * tile_size
            right = left + tile_size
            bottom = top + tile_size
            tile = image.crop((left, top, right, bottom))
            
            # मॉडेल प्रेडिक्शन
            tile_to_model = tile.resize((224, 224))
            tile_to_model = np.array(tile_to_model) / 255.0
            tile_to_model = np.expand_dims(tile_to_model, axis=0)
            
            if model is not None:
                prediction = model.predict(tile_to_model, verbose=0)
                result_index = np.argmax(prediction)
                
                # जर रोग (Bacterial Blight किंवा Red Rot) सापडला तर
                if result_index > 0:
                    # काल्पनिक को-ऑर्डिनेट्स (M.Sc. प्रोजेक्टसाठी)
                    lat = 18.5204 + (r * 0.0002)
                    lon = 73.8567 + (c * 0.0002)
                    
                    detected_diseases.append({
                        "तुकडा": f"Tile R{r+1}C{c+1}",
                        "रोग": classes[result_index],
                        "lat": lat,
                        "lon": lon
                    })
            
            count += 1
            progress_bar.progress(count / total_tiles)

    # ६. नकाशा आणि रिझल्ट्स (Display Results)
    st.success("विश्लेषण पूर्ण झाले!")
    
    # नकाशा तयार करा
    m = folium.Map(
        location=[18.5204, 73.8567], 
        zoom_start=18, 
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', 
        attr='Google Satellite'
    )

    if detected_diseases:
        # हीटमॅप लेयर
        heat_data = [[d["lat"], d["lon"]] for d in detected_diseases]
        HeatMap(heat_data, radius=15, blur=10).add_to(m)
        
        # मार्कर्स लेयर
        for d in detected_diseases:
            folium.Marker(
                [d["lat"], d["lon"]],
                popup=f"{d['तुकडा']}: {d['रोग']}",
                icon=folium.Icon(color='red', icon='leaf')
            ).add_to(m)
        
        # नकाशा दाखवा
        st_folium(m, width=900, height=500)

        # ७. ॲडव्हायझरी आणि रिपोर्ट (Final Section)
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("🌱 कृषी सल्ला (Advisory)")
            unique_found = set([d["रोग"] for d in detected_diseases])
            for dis in unique_found:
                if dis in advisory_map:
                    with st.expander(f"🚩 {dis} उपाय"):
                        st.write(f"💊 **औषध:** {advisory_map[dis]['औषध']}")
                        st.write(f"📢 **सल्ला:** {advisory_map[dis]['सल्ला']}")
        
        with col2:
            st.header("📥 अहवाल (Report)")
            pdf_data = create_pdf(detected_diseases, total_tiles)
            st.download_button(
                label="Download PDF Report",
                data=pdf_data,
                file_name="Crop_Health_Report.pdf",
                mime="application/pdf"
            )

        st.write("📋 **सविस्तर माहिती:**")
        st.table(detected_diseases)
        
    else:
        st.balloons()
        st.success("तुमचे शेत पूर्णपणे निरोगी आहे!")
        st_folium(m, width=900, height=500)
