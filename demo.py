"""
OceanGuard AI - Plataforma Avanzada de Mapeo de Residuos
"""
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
    page_icon="🗺️",
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

TENERIFE_DIVE_SITES = [
    {"name": "El Puertito de Adeje", "lat": 28.0863, "lon": -16.7942},
    {"name": "Pecio de Tabaiba", "lat": 28.3883, "lon": -16.3205},
    {"name": "Las Galletas", "lat": 28.0050, "lon": -16.6600},
    {"name": "Baja de los Guanches (Garachico)", "lat": 28.3758, "lon": -16.7664},
    {"name": "Radazul", "lat": 28.4069, "lon": -16.3361}
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
    except Exception:
        return None

def assign_random_tenerife_spot():
    """Asigna un punto de inmersión aleatorio de Tenerife."""
    spot = random.choice(TENERIFE_DIVE_SITES)
    return {"lat": spot["lat"], "lon": spot["lon"], "source": spot["name"]}

def calculate_ecosystem_health(detections):
    """Calcula la puntuación de salud basada en las detecciones."""
    if not detections: return 100
    total_risk_score = sum(DEBRIS_IMPACT_DATA.get(next((k for k in DEBRIS_IMPACT_DATA if k in (d.get('material','').lower() or d.get('debris_type','').lower())), "default"), {}).get('risk_score', 3) for d in detections)
    return max(0, 100 - total_risk_score)

def run_analysis(model, tokenizer, image):
    """Ejecuta el análisis de detección en una imagen."""
    prompt_text = (
        "You are a precise, expert marine debris detection system. "
        "Analyze the image and identify ALL man-made waste items. "
        "Return a JSON list of objects. Each object must have 'debris_type', 'material', and 'bounding_box' ([x_min, y_min, x_max, y_max] normalized from 0.0 to 1.0). "
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
        # MEJORA: Normalizar nombres de clases aquí
        for det in detections:
            if 'debris_type' in det:
                det['debris_type'] = det['debris_type'].strip().title()
        return detections if isinstance(detections, list) else []
    except (json.JSONDecodeError, TypeError):
        return []

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
    with col1: st.image(report_data['image'], caption="Imagen Original", use_container_width=True)
    with col2:
        annotated_image = visualize_bounding_boxes(report_data['image'].copy(), report_data['detections'])
        st.image(annotated_image, caption="Imagen con Detecciones", use_container_width=True)
    st.subheader("Panel de Impacto Ambiental")
    if not report_data['detections']: st.success("✅ No se detectaron residuos en esta imagen.")
    else:
        for det in report_data['detections']:
            debris_type, material = det.get('debris_type', 'Desconocido'), det.get('material', 'default').lower()
            impact_key = next((k for k in DEBRIS_IMPACT_DATA if k in material or k in debris_type.lower()), "default")
            impact_info = DEBRIS_IMPACT_DATA[impact_key]
            with st.container(border=True):
                col_icon, col_info = st.columns([1, 5])
                with col_icon: st.markdown(f"<p style='font-size: 48px; text-align: center;'>{impact_info['icon']}</p>", unsafe_allow_html=True)
                with col_info:
                    st.subheader(debris_type)
                    st.markdown(f"**⏳ Tiempo de degradación:** {impact_info['degradation']} | **🚨 Riesgo principal:** {impact_info['risk']}")

# --- INTERFAZ PRINCIPAL DE STREAMLIT ---

st.title("🗺️ OceanGuard AI: Plataforma de Mapeo de Residuos")
st.markdown("Crea un proyecto de campo subiendo múltiples imágenes para generar un análisis geoespacial de la contaminación.")

model, tokenizer = setup_model()

if 'project_files' not in st.session_state:
    st.session_state.project_files = []

with st.sidebar:
    st.header("🛰️ Proyecto de Campo")
    uploaded_files = st.file_uploader("Sube las imágenes de tu transecto", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        current_files = {f['name'] for f in st.session_state.project_files}
        for file in uploaded_files:
            if file.name not in current_files:
                gps_coords = get_gps_from_exif(file.getvalue()) or assign_random_tenerife_spot()
                st.session_state.project_files.append({"name": file.name, "file": file, "coords": gps_coords, "analysis": None})

    if st.session_state.project_files:
        st.markdown("---")
        st.subheader("Imágenes del Proyecto")
        for file_data in st.session_state.project_files:
            with st.expander(f"{file_data['name']}"):
                st.image(file_data['file'], width=100)
                if file_data['coords']['source'] == "EXIF": st.success(f"GPS (EXIF): {file_data['coords']['lat']:.4f}, {file_data['coords']['lon']:.4f}")
                else: st.info(f"📍 Coordenadas asignadas: {file_data['coords']['source']}")
        
        if st.button("Analizar Proyecto Completo", type="primary"):
            progress_bar = st.progress(0, "Iniciando análisis...")
            for i, file_data in enumerate(st.session_state.project_files):
                if file_data['analysis'] is None:
                    progress_bar.progress((i+1)/len(st.session_state.project_files), f"Analizando: {file_data['name']}")
                    detections = run_analysis(model, tokenizer, Image.open(file_data['file']).convert("RGB"))
                    file_data['analysis'] = {"detections": detections, "health_score": calculate_ecosystem_health(detections)}
            progress_bar.progress(1.0, "Análisis completado.")

analyzed_data = [f for f in st.session_state.project_files if f.get('analysis') and f.get('coords')]

if not analyzed_data:
    st.info("Sube imágenes y haz clic en 'Analizar Proyecto Completo' para ver el dashboard.")
else:
    st.header("Dashboard Geoespacial del Proyecto")
    
    view_state = pdk.ViewState(latitude=28.2916, longitude=-16.6291, zoom=8, pitch=50)
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=pd.DataFrame([{
            "lat": d['coords']['lat'], "lon": d['coords']['lon'],
            "health": d['analysis']['health_score'],
            "debris_count": len(d['analysis']['detections']),
            "location_name": d['coords']['source'] if d['coords']['source'] != 'EXIF' else 'Ubicación Personalizada'
        } for d in analyzed_data]),
        get_position='[lon, lat]',
        get_color='[255, (100 - health) * 2.55, 0, 160]',
        get_radius='(100 - health) * 10 + 100',
        pickable=True
    )
    tooltip = {
        "html": "<b>{location_name}</b><br/>Puntuación de Salud: <b>{health}</b>/100<br/>Residuos Detectados: <b>{debris_count}</b>",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

    st.subheader("Análisis por Regiones de Inmersión")
    regional_summary = {}
    for d in analyzed_data:
        region = d['coords']['source']
        if region not in regional_summary: regional_summary[region] = {"scores": [], "debris_counts": 0, "image_count": 0}
        regional_summary[region]["scores"].append(d['analysis']['health_score'])
        regional_summary[region]["debris_counts"] += len(d['analysis']['detections'])
        regional_summary[region]["image_count"] += 1
    
    cols = st.columns(len(regional_summary))
    for i, (region, data) in enumerate(regional_summary.items()):
        with cols[i]:
            avg_score = sum(data['scores']) / len(data['scores'])
            st.metric(label=f"Salud Media en {region}", value=f"{avg_score:.1f}/100")
            st.info(f"{data['debris_counts']} residuos en {data['image_count']} imágenes.")

    st.subheader("Estadísticas Globales del Proyecto")
    all_detections = [det for d in analyzed_data for det in d['analysis']['detections']]
    if not all_detections:
        st.success("✅ ¡Felicidades! No se encontraron residuos en ninguna de las imágenes analizadas.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Imágenes Analizadas", len(analyzed_data))
            st.metric("Total de Residuos Detectados", len(all_detections))
        with col2:
            counts = Counter([d.get('debris_type', 'Desconocido') for d in all_detections])
            df_counts = pd.DataFrame(list(counts.items()), columns=['Tipo de Residuo', 'Cantidad'])
            st.bar_chart(df_counts.set_index('Tipo de Residuo'))

    st.markdown("---")
    st.header("Explorar Informes Individuales")
    selected_image_name = st.selectbox("Selecciona una imagen para ver su informe detallado:", [d['name'] for d in analyzed_data])
    
    selected_data = next((d for d in analyzed_data if d['name'] == selected_image_name), None)
    if selected_data:
        report_payload = {
            "name": selected_data['name'],
            "image": Image.open(next(f['file'] for f in st.session_state.project_files if f['name'] == selected_image_name)).convert("RGB"),
            "detections": selected_data['analysis']['detections']
        }
        display_detailed_report(report_payload)
