import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os
from fpdf import FPDF
import base64

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
# --- कृषी सल्ला (Advisory) ---
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

# --- PDF फंक्शन ---
def create_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Sugarcane Disease Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    for d in data:
        pdf.cell(200, 10, txt=f"- {d['तुकडा']}: {d['रोग']}", ln=True)
    return pdf.output(dest='S').encode('latin-1')
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
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Drone Image', use_container_width=True)
    
#     # १. स्वयंचलित टायलिंग (Automated Tiling)
#     width, height = image.size
#     tile_size = 224 # तुमच्या मॉडेलचा इनपुट आकार
    
#     # किती तुकडे होतील हे मोजा
#     cols = width // tile_size
#     rows = height // tile_size
#     st.info(f"तुमच्या फोटोचे एकूण {cols * rows} तुकड्यांत विश्लेषण होत आहे...")

#     detected_diseases = []
    
#     # प्रगती दाखवण्यासाठी प्रोग्रेस बार
#     progress_bar = st.progress(0)
#     total_tiles = cols * rows
#     current_tile = 0

#     # २. लूप वापरून आपोआप तुकडे करणे
#     for r in range(rows):
#         for c in range(cols):
#             left = c * tile_size
#             top = r * tile_size
#             right = left + tile_size
#             bottom = top + tile_size
            
#             # तुकडा कापा
#             tile_img = image.crop((left, top, right, bottom))
            
#             # मॉडेलसाठी प्रोसेसिंग
#             img_array = np.array(tile_img.resize((224, 224))) / 255.0
#             img_array = np.expand_dims(img_array, axis=0)
            
#             prediction = model.predict(img_array, verbose=0)
#             result_index = np.argmax(prediction)
            
#             # जर रोग आढळला तर लोकेशन साठवा
#             if result_index > 0:
#                 # काल्पनिक को-ऑर्डिनेट्स (M.Sc. साठी पुण्याचे सॅम्पल)
#                 # खऱ्या जीआयएस मध्ये इथे पिक्सेल-टू-कोऑर्डिनेट लॉजिक येईल
#                 lat = 18.5204 + (r * 0.0005) 
#                 lon = 73.8567 + (c * 0.0005)
                
#                 detected_diseases.append({
#                     "तुकडा": f"Row {r+1}, Col {c+1}",
#                     "रोग": classes[result_index],
#                     "lat": lat,
#                     "lon": lon
#                 })
            
#             current_tile += 1
#             progress_bar.progress(current_tile / total_tiles)

#     # ३. नकाशावर निकाल दाखवणे
#     st.success("विश्लेषण पूर्ण झाले!")
    
#     m = folium.Map(
#         location=[18.5204, 73.8567], 
#         zoom_start=17, 
#         tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', 
#         attr='Google Satellite Hybrid'
#     )

#     if detected_diseases:
#         for d in detected_diseases:
#             folium.Marker(
#                 [d["lat"], d["lon"]],
#                 popup=f"{d['तुकडा']}: {d['रोग']}",
#                 icon=folium.Icon(color='red', icon='info-sign')
#             ).add_to(m)
        
#         st_folium(m, width=700, height=450)
#         st.write("📋 **सापडलेल्या रोगांचा तपशील:**")
#         st.table(detected_diseases)
#     else:
#         st.balloons()
#         st.success("तुमचे शेत पूर्णपणे निरोगी आहे! नकाशावर कोणतेही रोग आढळले नाहीत.")
#         st_folium(m, width=700, height=450)


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Drone Image', use_container_width=True)
    
    # १. स्वयंचलित टायलिंग (Automated Tiling)
    width, height = image.size
    tile_size = 224 # तुमच्या मॉडेलचा इनपुट आकार
    
    # किती तुकडे होतील हे मोजा
    cols = width // tile_size
    rows = height // tile_size
    st.info(f"तुमच्या फोटोचे एकूण {cols * rows} तुकड्यांत विश्लेषण होत आहे...")

    detected_diseases = []
    
    # प्रगती दाखवण्यासाठी प्रोग्रेस बार
    progress_bar = st.progress(0)
    total_tiles = cols * rows
    current_tile = 0
    detected_diseases = []

    # २. लूप वापरून आपोआप तुकडे करणे
    for r in range(rows):
        for c in range(cols):
            left = c * tile_size
            top = r * tile_size
            right = left + tile_size
            bottom = top + tile_size
            
            # तुकडा कापा
            tile_img = image.crop((left, top, right, bottom))
            
            # मॉडेलसाठी प्रोसेसिंग
            img_array = np.array(tile_img.resize((224, 224))) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array, verbose=0)
            result_index = np.argmax(prediction)
            
            # जर रोग आढळला तर लोकेशन साठवा
            if result_index > 0:
                # काल्पनिक को-ऑर्डिनेट्स (M.Sc. साठी पुण्याचे सॅम्पल)
                # खऱ्या जीआयएस मध्ये इथे पिक्सेल-टू-कोऑर्डिनेट लॉजिक येईल
                lat = 18.5204 + (r * 0.0005) 
                lon = 73.8567 + (c * 0.0005)
# --- अचूक मांडणी (ओळ २४७ नंतर) ---
    if detected_diseases:
    # १. हीटमॅप (इथे डावीकडून ४ स्पेस सोडा)
        heat_data = [[d["lat"], d["lon"]] for d in detected_diseases]
        HeatMap(heat_data, radius=15, blur=10).add_to(m)
    
    # २. मार्कर्स (हे 'for' सुद्धा ४ स्पेसवर हवे)
    for d in detected_diseases:
        folium.Marker(
            [d["lat"], d["lon"]],
            popup=f"{d['तुकडा']}: {d['रोग']}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)

    # ३. नकाशा दाखवा (हा 'if' च्या आतच हवा)
        st_folium(m, width=700, height=450)
        detected_diseases.append({
        "तुकडा": f"Row {r+1}, Col {c+1}",
        "रोग": classes[result_index],
        "lat": lat,
        "lon": lon
                })
            
    current_tile += 1
    progress_bar.progress(current_tile / total_tiles)

    # ३. नकाशावर निकाल दाखवणे
    if detected_diseases:
        st.success("विश्लेषण पूर्ण झाले!")
    
        m = folium.Map(
        location=[18.5204, 73.8567], 
        zoom_start=17, 
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', 
        attr='Google Satellite Hybrid'
    )

        heat_data = [[d["lat"], d["lon"]] for d in detected_diseases]
        HeatMap(heat_data, radius=15, blur=10).add_to(m) # हीटमॅप जोडलाा
        for d in detected_diseases:
            folium.Marker(
                [d["lat"], d["lon"]],
                popup=f"{d['तुकडा']}: {d['रोग']}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
        
        st_folium(m, width=700, height=450)
        st.write("📋 **सापडलेल्या रोगांचा तपशील:**")
        st.table(detected_diseases)
    else:
        st.balloons()
        st.success("तुमचे शेत पूर्णपणे निरोगी आहे! नकाशावर कोणतेही रोग आढळले नाहीत.")
        st_folium(m, width=700, height=450)
        # ४. कृषी सल्ला आणि रिपोर्ट (हे 'if detected_diseases' च्या आत हवे)
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🌱 कृषी सल्ला")
        unique_diseases = set([d["रोग"] for d in detected_diseases])
        for disease in unique_diseases:
            if disease in advisory_map:
                with st.expander(f"🚩 {disease} उपाय"):
                    st.write(f"💊 **औषध:** {advisory_map[disease]['औषध']}")
                    st.write(f"📢 **सल्ला:** {advisory_map[disease]['सल्ला']}")
    
    with col2:
        st.header("📥 रिपोर्ट")
        pdf_data = create_pdf(detected_diseases)
        st.download_button("Download PDF Report", data=pdf_data, file_name="Report.pdf")

    # सविस्तर टेबल
    st.write("📋 **सापडलेल्या रोगांचा तपशील:**")
    st.table(detected_diseases)

else:
    # जर रोग सापडला नाही तर
    st.balloons()
    st.success("शेतात कुठेही रोग आढळला नाही!")
# # --- कृषी सल्ला (Advisory) ---
# advisory_map = {
#     "Bacterial Blight": {
#         "औषध": "Streptocycline (100 ppm) + Copper Oxychloride (0.25%)",
#         "सल्ला": "बाधित पाने कापून नष्ट करा. नत्राचा (Nitrogen) वापर काही काळ टाळा."
#     },
#     "Red Rot": {
#         "औषध": "Carbendazim (0.1%) किंवा Trichoderma viride",
#         "सल्ला": "पाण्याचा निचरा सुधारा. बाधित खुंट उपटून टाका. बेणे प्रक्रिया करा."
#     }
# }

# # --- PDF फंक्शन ---
# def create_pdf(data):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", 'B', 16)
#     pdf.cell(200, 10, txt="Sugarcane Disease Report", ln=True, align='C')
#     pdf.set_font("Arial", size=12)
#     pdf.ln(10)
#     for d in data:
#         pdf.cell(200, 10, txt=f"- {d['तुकडा']}: {d['रोग']}", ln=True)
#     return pdf.output(dest='S').encode('latin-1')
