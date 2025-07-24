import streamlit as st
import torch
from unsloth import FastLanguageModel
from PIL import Image, ImageDraw, ImageFont
import json
import pandas as pd
from collections import Counter
import piexif
import random
import pydeck as pdk

# --- CONFIGURACIÓN DE LA PÁGINA Y DEL MODELO ---

st.set_page_config(
    page_title="OceanGuard AI - Mapeo Avanzado",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

torch._dynamo.disable()
torch._dynamo.config.cache_size_limit = 99999999999999999999

@st.cache_resource
def setup_model():
    """Carga y configura el modelo Gemma 3n y el tokenizador."""
    print("--- Cargando el modelo Gemma 3n (esto solo ocurrirá una vez) ---")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-3n-e2b-it",
        max_seq_length=4096,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map={'': 0}
    )
    return model, tokenizer

# --- Base de datos de impacto y coordenadas ---
DEBRIS_IMPACT_DATA = {
    "plastic": {"degradation": "450+ años", "risk": "Ingestión, microplásticos.", "icon": "💧", "risk_score": 8},
    "metal": {"degradation": "50-200 años", "risk": "Lixiviación de químicos.", "icon": "🔩", "risk_score": 6},
    "glass": {"degradation": "1,000,000+ años", "risk": "Heridas físicas.", "icon": "🍾", "risk_score": 4},
    "net": {"degradation": "600+ años", "risk": "'Pesca fantasma'.", "icon": "🕸️", "risk_score": 10},
    "fabric": {"degradation": "1-200+ años", "risk": "Liberación de microfibras.", "icon": "👕", "risk_score": 5},
    "rubber": {"degradation": "50-80 años", "risk": "Lixiviación de tóxicos.", "icon": "🛞", "risk_score": 7},
    "default": {"degradation": "Desconocido", "risk": "Desconocido.", "icon": "❓", "risk_score": 3}
}

DIVE_SITES = [
    # Tenerife dive sites
    {"name": "El Puertito de Adeje",                "lat": 28.0863,   "lon": -16.7942},
    {"name": "Pecio de Tabaiba",                    "lat": 28.3883,   "lon": -16.3205},
    {"name": "Las Galletas",                        "lat": 28.0050,   "lon": -16.6600},
    {"name": "Baja de los Guanches (Garachico)",    "lat": 28.3758,   "lon": -16.7664},
    {"name": "Radazul",                             "lat": 28.4069,   "lon": -16.3361},

    # Dive sites worldwide
    {"name": "Isla de Benidorm",                    "lat": 38.5013,     "lon": -0.1248},     # :contentReference[oaicite:0]{index=0}
    {"name": "Great Blue Hole (Belize)",            "lat": 17.316010,   "lon": -87.535103},  # :contentReference[oaicite:1]{index=1}
    {"name": "Shark & Yolanda Reef (Ras Mohammed, Egipto)", "lat": 27.72521, "lon": 34.25889},    # :contentReference[oaicite:2]{index=2}
    {"name": "SS Thistlegorm (Egipto)",             "lat": 27.808497,   "lon": 33.918664},   # :contentReference[oaicite:3]{index=3}
    {"name": "SS Yongala (Australia)",               "lat": -19.3045,    "lon": 147.6218},    # :contentReference[oaicite:4]{index=4}
    {"name": "Sipadan (Malasia)",                   "lat": 4.119467,    "lon": 118.629817},   # :contentReference[oaicite:5]{index=5}
    {"name": "Cocos Island (Costa Rica)",           "lat": 5.5180,      "lon": -87.069667},   # :contentReference[oaicite:6]{index=6}
    {"name": "Blue Corner Wall (Palau)",            "lat": 7.133611,    "lon": 134.220278},   # :contentReference[oaicite:7]{index=7}
    {"name": "Norman Reef (Gran Barrera de Coral, Australia)", "lat": -16.426719, "lon": 145.994347}, # :contentReference[oaicite:8]{index=8}
    {"name": "USAT Liberty (Tulamben, Indonesia)",  "lat": -8.27396,    "lon": 115.59307},    # :contentReference[oaicite:9]{index=9}
    {"name": "Dos Ojos Cenote (México)",            "lat": 20.32483,    "lon": -87.39059}     # :contentReference[oaicite:10]{index=10}
]

# --- FUNCIONES AUXILIARES ---

def get_gps_from_exif(image_bytes):
    """Extrae coordenadas GPS de los metadatos EXIF de una imagen."""
    try:
        exif_dict = piexif.load(image_bytes)
        gps_info = exif_dict.get("GPS")
        if not gps_info: return None
        def convert_to_degrees(value):
            d, m, s = [float(v[0]) / float(v[1]) for v in value]
            return d + (m / 60.0) + (s / 3600.0)
        lat = convert_to_degrees(gps_info[piexif.GPSIFD.GPSLatitude])
        if gps_info[piexif.GPSIFD.GPSLatitudeRef].decode() == 'S': lat = -lat
        lon = convert_to_degrees(gps_info[piexif.GPSIFD.GPSLongitude])
        if gps_info[piexif.GPSIFD.GPSLongitudeRef].decode() == 'W': lon = -lon
        return {"lat": lat, "lon": lon, "source": "EXIF"}
    except Exception: return None

def assign_random_dive_spot():
    """Asigna un punto de inmersión aleatorio de inmersion."""
    spot = random.choice(DIVE_SITES)
    return {"lat": spot["lat"], "lon": spot["lon"], "source": spot["name"]}

def calculate_ecosystem_health(detections):
    """Calcula la puntuación de salud basada en las detecciones."""
    if not detections: return 100
    total_risk_score = sum(DEBRIS_IMPACT_DATA.get(next((k for k in DEBRIS_IMPACT_DATA if k in (d.get('material','').lower() or d.get('debris_type','').lower())), "default"), {}).get('risk_score', 3) for d in detections)
    return max(0, 100 - total_risk_score)

def run_analysis(model, tokenizer, image):
    """Ejecuta el análisis de detección en una imagen, solicitando la confianza."""
    prompt_text = (
        "You are a precise, expert marine debris detection system. "
        "Analyze the image and identify ALL man-made waste items. "
        "Return a JSON list of objects. Each object must have:\n"
        "1. 'debris_type': The specific type of object.\n"
        "2. 'material': The likely material.\n"
        "3. 'confidence_score': A float from 0.0 to 1.0 indicating your certainty.\n"
        "4. 'bounding_box': A list of four normalized coordinates [x_min, y_min, x_max, y_max].\n"
        "Your response MUST be ONLY the JSON list."
    )
    prompt_content = [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]
    messages = [{"role": "user", "content": prompt_content}]
    
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
    output_tokens = model.generate(**inputs, max_new_tokens=2048, temperature=0.1)
    response_text = tokenizer.batch_decode(output_tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    torch.cuda.empty_cache()
    
    try:
        detections = json.loads(response_text.strip().replace("```json", "").replace("```", ""))
        for det in detections:
            if 'debris_type' in det: det['debris_type'] = det['debris_type'].strip().title()
        return detections if isinstance(detections, list) else []
    except (json.JSONDecodeError, TypeError): return []

# --- FUNCIONES DE VISUALIZACIÓN ---

def visualize_bounding_boxes(image, detections):
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    width, height = img_draw.size
    color_map = {"plastic": "#FF1493", "metal": "#1E90FF", "fabric": "#32CD32", "textile": "#32CD32", "rubber": "#FFD700", "glass": "#00CED1", "nylon": "#9400D3", "synthetic": "#9400D3", "fishing": "#FF8C00", "net": "#FF8C00", "default": "#FFFFFF"}
    try: font = ImageFont.truetype("arial.ttf", 20)
    except IOError: font = ImageFont.load_default()

    for det in detections:
        box, label, material = det.get('bounding_box'), det.get('debris_type', 'Unknown'), det.get('material', 'default').lower()
        if not isinstance(box, list) or len(box) != 4: continue
        color_key = next((k for k in color_map if k in material or k in label.lower()), "default")
        color = color_map[color_key]
        xmin, ymin, xmax, ymax = box
        draw.rectangle([(xmin * width, ymin * height), (xmax * width, ymax * height)], outline=color, width=4)
        text_bbox = draw.textbbox((xmin * width, ymin * height - 22), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((xmin * width, ymin * height - 22), label, fill="black", font=font)
    return img_draw

def display_detailed_report(report_data):
    st.header(f"Informe Detallado: {report_data['name']}")
    col1, col2 = st.columns(2)
    with col1: st.image(report_data['image'], caption="Imagen Original", use_column_width=True)
    with col2:
        annotated_image = visualize_bounding_boxes(report_data['image'].copy(), report_data['detections'])
        st.image(annotated_image, caption="Imagen con Detecciones", use_column_width=True)
    st.subheader("Panel de Impacto Ambiental")
    if not report_data['detections']: st.success("✅ No se detectaron residuos con el filtro de confianza actual.")
    else:
        for det in report_data['detections']:
            debris_type, material = det.get('debris_type', 'Desconocido'), det.get('material', 'default').lower()
            confidence = det.get('confidence_score', 0.0)
            impact_key = next((k for k in DEBRIS_IMPACT_DATA if k in material or k in debris_type.lower()), "default")
            impact_info = DEBRIS_IMPACT_DATA[impact_key]
            with st.container(border=True):
                col_icon, col_info = st.columns([1, 5])
                with col_icon: st.markdown(f"<p style='font-size: 48px; text-align: center;'>{impact_info['icon']}</p>", unsafe_allow_html=True)
                with col_info:
                    st.subheader(debris_type)
                    st.markdown(f"**🎯 Confianza de la Detección:** {confidence:.1%}")
                    st.markdown(f"**⏳ Tiempo de degradación:** {impact_info['degradation']} | **🚨 Riesgo principal:** {impact_info['risk']}")

# --- INTERFAZ PRINCIPAL DE STREAMLIT ---

st.title("🛰️ OceanGuard AI: Plataforma de Mapeo Científico")
st.markdown("Crea un proyecto de campo, analiza la fiabilidad de las detecciones y exporta tus resultados.")

model, tokenizer = setup_model()

if 'project_files' not in st.session_state: st.session_state.project_files = []

with st.sidebar:
    st.header("⚙️ Controles del Proyecto")
    uploaded_files = st.file_uploader("Sube las imágenes de tu transecto", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        current_files = {f['name'] for f in st.session_state.project_files}
        for file in uploaded_files:
            if file.name not in current_files:
                gps_coords = get_gps_from_exif(file.getvalue()) or assign_random_dive_spot()
                st.session_state.project_files.append({"name": file.name, "file": file, "coords": gps_coords, "analysis": None})

    if st.session_state.project_files:
        st.markdown("---")
        confidence_threshold = st.slider("Filtrar por Confianza de Detección", 0.0, 1.0, 0.5, 0.05, help="Muestra solo detecciones con una confianza superior a este valor.")
        
        if st.button("Analizar Proyecto Completo", type="primary"):
            progress_bar = st.progress(0, "Iniciando análisis...")
            for i, file_data in enumerate(st.session_state.project_files):
                if file_data['analysis'] is None:
                    progress_bar.progress((i+1)/len(st.session_state.project_files), f"Analizando: {file_data['name']}")
                    detections = run_analysis(model, tokenizer, Image.open(file_data['file']).convert("RGB"))
                    file_data['analysis'] = {"detections": detections}
            progress_bar.progress(1.0, "Análisis completado.")

# Filtrar datos basados en el umbral de confianza
filtered_project_files = []
if st.session_state.project_files:
    for file_data in st.session_state.project_files:
        if file_data['analysis']:
            filtered_detections = [d for d in file_data['analysis']['detections'] if d.get('confidence_score', 0.0) >= confidence_threshold]
            filtered_project_files.append({
                **file_data,
                "filtered_analysis": {
                    "detections": filtered_detections,
                    "health_score": calculate_ecosystem_health(filtered_detections)
                }
            })
        else:
            filtered_project_files.append(file_data)

analyzed_data = [f for f in filtered_project_files if f.get('filtered_analysis') and f.get('coords')]

if not analyzed_data:
    st.info("Sube imágenes y haz clic en 'Analizar Proyecto Completo' para ver el dashboard.")
else:
    st.header("Dashboard Geoespacial del Proyecto")
    
    map_df = pd.DataFrame([{
        "lat": d['coords']['lat'], "lon": d['coords']['lon'],
        "health": d['filtered_analysis']['health_score'],
        "debris_count": len(d['filtered_analysis']['detections']),
        "location_name": d['coords']['source'] if d['coords']['source'] != 'EXIF' else 'Ubicación Personalizada'
    } for d in analyzed_data])
    
    view_state = pdk.ViewState(latitude=map_df['lat'].mean(), longitude=map_df['lon'].mean(), zoom=8, pitch=50)
    layer = pdk.Layer(
        'ScatterplotLayer', data=map_df, get_position='[lon, lat]',
        get_color='[(100 - health) * 2.55, health * 2.55, 0, 160]',
        get_radius='(debris_count * 50) + 100', radius_min_pixels=5,
        pickable=True
    )
    tooltip = {"html": "<b>{location_name}</b><br/>Puntuación de Salud: <b>{health}</b>/100<br/>Residuos Detectados: <b>{debris_count}</b>"}
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

    st.subheader("Estadísticas Globales del Proyecto")
    all_filtered_detections = [det for d in analyzed_data for det in d['filtered_analysis']['detections']]
    if not all_filtered_detections:
        st.success(f"✅ ¡Felicidades! No se encontraron residuos con una confianza superior a {confidence_threshold:.0%}.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Imágenes Analizadas", len(analyzed_data))
            st.metric("Total de Residuos Detectados (filtrados)", len(all_filtered_detections))
        with col2:
            counts = Counter([d.get('debris_type', 'Desconocido') for d in all_filtered_detections])
            df_counts = pd.DataFrame(list(counts.items()), columns=['Tipo de Residuo', 'Cantidad'])
            st.bar_chart(df_counts.set_index('Tipo de Residuo'))

    st.markdown("---")
    st.header("Explorar Informes Individuales")
    selected_image_name = st.selectbox("Selecciona una imagen para ver su informe detallado:", [d['name'] for d in analyzed_data])
    
    selected_data = next((d for d in analyzed_data if d['name'] == selected_image_name), None)
    if selected_data:
        report_payload = {
            "name": selected_data['name'],
            "image": Image.open(selected_data['file']).convert("RGB"),
            "detections": selected_data['filtered_analysis']['detections']
        }
        display_detailed_report(report_payload)
    
    st.sidebar.markdown("---")
    st.sidebar.header("📥 Exportar Resultados")
    report_data_list = []
    for d in analyzed_data:
        for det in d['filtered_analysis']['detections']:
            report_data_list.append({
                'image_name': d['name'],
                'location_name': d['coords']['source'],
                'latitude': d['coords']['lat'],
                'longitude': d['coords']['lon'],
                'debris_type': det.get('debris_type'),
                'material': det.get('material'),
                'confidence_score': det.get('confidence_score')
            })
    
    if report_data_list:
        df_report = pd.DataFrame(report_data_list)
        csv = df_report.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
           label="Descargar Informe (CSV)",
           data=csv,
           file_name=f"ocean_guard_report_conf_{confidence_threshold:.2f}.csv",
           mime="text/csv",
        )
    else:
        st.sidebar.info("No hay datos para exportar con el filtro actual.")
