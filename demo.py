"""
OceanGuard AI - Aplicación Interactiva
"""
import streamlit as st
import torch
from unsloth import FastLanguageModel
from PIL import Image, ImageDraw, ImageFont
import json
import pandas as pd
from collections import Counter

# --- CONFIGURACIÓN DE LA PÁGINA Y DEL MODELO ---

st.set_page_config(
    page_title="OceanGuard AI - Informe de Análisis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Desactivar dynamo para compatibilidad con Unsloth
torch._dynamo.disable()
torch._dynamo.config.cache_size_limit = 99999999999999999999

@st.cache_resource
def setup_model():
    """
    Carga y configura el modelo Gemma 3n y el tokenizador.
    """
    print("--- Cargando el modelo Gemma 3n (esto solo ocurrirá una vez) ---")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-3n-e2b-it",
        max_seq_length=4096,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map={'': 0}
    )
    return model, tokenizer

# --- Base de datos enriquecida con riesgo y acciones ---
DEBRIS_IMPACT_DATA = {
    "plastic": {"degradation": "450+ años", "risk": "Ingestión, microplásticos, asfixia.", "icon": "💧", "risk_score": 8, "risk_type": "Microplásticos", "action": "Retirada prioritaria para evitar fragmentación."},
    "metal": {"degradation": "50-200 años", "risk": "Lixiviación de químicos, heridas.", "icon": "🔩", "risk_score": 6, "risk_type": "Químico", "action": "Manejar con cuidado. Puede tener bordes cortantes."},
    "glass": {"degradation": "1,000,000+ años", "risk": "Heridas físicas, no se degrada.", "icon": "🍾", "risk_score": 4, "risk_type": "Físico", "action": "Retirar con equipo de protección; riesgo de corte."},
    "net": {"degradation": "600+ años", "risk": "Alto riesgo de 'pesca fantasma'.", "icon": "🕸️", "risk_score": 10, "risk_type": "Enredo", "action": "¡ALERTA MÁXIMA! Requiere retirada inmediata por equipos especializados."},
    "fabric": {"degradation": "1-200+ años", "risk": "Liberación de microfibras.", "icon": "👕", "risk_score": 5, "risk_type": "Microplásticos", "action": "Las fibras sintéticas se descomponen en microfibras. Retirar."},
    "rubber": {"degradation": "50-80 años", "risk": "Lixiviación de aditivos tóxicos.", "icon": "🛞", "risk_score": 7, "risk_type": "Químico", "action": "Puede lixiviar tóxicos durante su lenta degradación."},
    "default": {"degradation": "Desconocido", "risk": "Impacto desconocido.", "icon": "❓", "risk_score": 3, "risk_type": "Desconocido", "action": "Se requiere identificación para determinar el plan de acción."}
}

# --- FUNCIONES DE PROCESAMIENTO Y VISUALIZACIÓN ---

def visualize_bounding_boxes(image: Image.Image, detections: list) -> Image.Image:
    """
    Dibuja los bounding boxes y etiquetas sobre una imagen.
    """
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    width, height = img_draw.size
    color_map = {
        "plastic": "#FF1493", "metal": "#1E90FF", "fabric": "#32CD32",
        "textile": "#32CD32", "rubber": "#FFD700", "glass": "#00CED1",
        "nylon": "#9400D3", "synthetic": "#9400D3", "fishing": "#FF8C00", "net": "#FF8C00",
        "default": "#FFFFFF"
    }
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for det in detections:
        box = det.get('bounding_box')
        label = det.get('debris_type', 'Unknown')
        material = det.get('material', 'default').lower()
        if not isinstance(box, list) or len(box) != 4: continue
        
        color_key = next((key for key in color_map if key in material or key in label.lower()), "default")
        color = color_map[color_key]

        xmin, ymin, xmax, ymax = box
        draw.rectangle([(xmin * width, ymin * height), (xmax * width, ymax * height)], outline=color, width=4)
        text_bbox = draw.textbbox((xmin * width, ymin * height - 22), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((xmin * width, ymin * height - 22), label, fill="black", font=font)
        
    return img_draw

def run_analysis(model, tokenizer, image, prompt_text):
    """
    Función genérica para ejecutar el modelo con un prompt.
    """
    prompt_content = [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]
    messages = [{"role": "user", "content": prompt_content}]
    
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to("cuda")

    output_tokens = model.generate(**inputs, max_new_tokens=2048, temperature=0.1)
    response_text = tokenizer.batch_decode(output_tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    torch.cuda.empty_cache()
    return response_text

# --- Función para calcular la salud del ecosistema ---
def calculate_ecosystem_health(detections):
    if not detections:
        return 100, set(), "Saludable"
    
    total_risk_score = 0
    identified_risks = set()
    
    for det in detections:
        debris_type = det.get('debris_type', 'Desconocido').lower()
        material = det.get('material', 'default').lower()
        impact_key = next((key for key in DEBRIS_IMPACT_DATA if key in material or key in debris_type), "default")
        impact_info = DEBRIS_IMPACT_DATA[impact_key]
        total_risk_score += impact_info['risk_score']
        identified_risks.add(impact_info['risk_type'])
        
    health_score = max(0, 100 - total_risk_score)
    
    if health_score > 80: level = "Saludable"
    elif health_score > 60: level = "Moderado"
    elif health_score > 40: level = "En Riesgo"
    else: level = "Crítico"
        
    return health_score, identified_risks, level

# --- INTERFAZ DE STREAMLIT ---

st.title("🌊 OceanGuard AI: Informe de Análisis de Residuos Marinos")
st.markdown("Sube una imagen del fondo marino para generar un informe completo e interactuar con la IA.")

model, tokenizer = setup_model()

# Inicializar el estado de la sesión
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'detections' not in st.session_state: st.session_state.detections = None
if 'image_id' not in st.session_state: st.session_state.image_id = None

with st.sidebar:
    st.header("⚙️ Controles")
    uploaded_file = st.file_uploader("1. Sube tu imagen aquí", type=["jpg", "jpeg", "png"])
    
    # --- CORRECCIÓN DEL ERROR ---
    if uploaded_file and uploaded_file.file_id != st.session_state.image_id:
        st.session_state.image_id = uploaded_file.file_id
        st.session_state.detections = None
        st.session_state.chat_history = []
    
    analyze_button = st.button("Generar Informe de Análisis", type="primary", disabled=(uploaded_file is None))

if uploaded_file:
    original_image_pil = Image.open(uploaded_file).convert("RGB")
    
    if analyze_button and st.session_state.detections is None:
        with st.spinner('Generando informe... El modelo está analizando la imagen 🧠'):
            detection_prompt = (
                "You are a precise, expert marine debris detection system. "
                "Analyze the image and identify ALL man-made waste items. "
                "Return a JSON list of objects. Each object must have 'debris_type', 'material', and 'bounding_box' ([x_min, y_min, x_max, y_max] normalized from 0.0 to 1.0). "
                "Your response MUST be ONLY the JSON list."
            )
            response = run_analysis(model, tokenizer, original_image_pil, detection_prompt)
            try:
                json_str = response.strip().replace("```json", "").replace("```", "")
                detections_list = json.loads(json_str)
                st.session_state.detections = detections_list if isinstance(detections_list, list) else []
            except (json.JSONDecodeError, TypeError):
                st.error("El modelo no devolvió un JSON de detección válido.")
                st.code(response)
                st.session_state.detections = []

    if st.session_state.detections is not None:
        detections = st.session_state.detections
        
        # Cálculo y visualización del riesgo ---
        health_score, identified_risks, health_level = calculate_ecosystem_health(detections)
        st.header(f"Evaluación General del Ecosistema: Nivel {health_level}")
        
        score_color = "green" if health_level == "Saludable" else "orange" if health_level in ["Moderado", "En Riesgo"] else "red"
        st.metric(label="Puntuación de Salud del Ecosistema", value=f"{health_score} / 100", delta=f"-{100-health_score} puntos de riesgo", delta_color="inverse")
        st.progress(health_score / 100)
        
        if identified_risks:
            st.markdown(f"**Principales Tipos de Riesgo Detectados:** `{'`, `'.join(identified_risks)}`")
        
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1: st.image(original_image_pil, caption="Imagen Original", use_column_width=True)
        with col2:
            annotated_image = visualize_bounding_boxes(original_image_pil, detections)
            st.image(annotated_image, caption="Imagen con Detecciones", use_column_width=True)

        tabs = ["📊 Resumen", "🐠 Impacto y Acciones", "📄 Datos (JSON)", "💬 Chat con la IA"]
        tab1, tab2, tab3, tab4 = st.tabs(tabs)

        with tab1:
            st.subheader("Resumen Cuantitativo")
            if not detections: st.success("✅ ¡Buenas noticias! No se ha detectado ningún residuo en esta imagen.")
            else:
                st.metric(label="Número Total de Residuos Detectados", value=len(detections))
                counts = Counter([d.get('debris_type', 'Desconocido') for d in detections])
                df = pd.DataFrame(list(counts.items()), columns=['Tipo de Residuo', 'Cantidad'])
                st.bar_chart(df.set_index('Tipo de Residuo'))

        with tab2:
            st.subheader("Panel de Impacto Ambiental y Sugerencias de Acción")
            if not detections: st.success("✅ No se detectaron residuos, por lo que no hay impacto que detallar.")
            else:
                for det in detections:
                    debris_type = det.get('debris_type', 'Desconocido')
                    material = det.get('material', 'default').lower()
                    impact_key = next((key for key in DEBRIS_IMPACT_DATA if key in material or key in debris_type.lower()), "default")
                    impact_info = DEBRIS_IMPACT_DATA[impact_key]
                    with st.container(border=True):
                        col_icon, col_info = st.columns([1, 5])
                        with col_icon: st.markdown(f"<p style='font-size: 48px; text-align: center;'>{impact_info['icon']}</p>", unsafe_allow_html=True)
                        with col_info:
                            st.subheader(debris_type)
                            st.markdown(f"**⏳ Tiempo de degradación:** {impact_info['degradation']} | **🚨 Riesgo principal:** {impact_info['risk']}")
                            # MEJORA FASE 3: Sugerencia de acción
                            st.markdown(f"**💡 Sugerencia de Acción:** {impact_info['action']}")
        
        with tab3:
            st.subheader("Salida del Modelo en Formato JSON")
            if not detections: st.success("✅ No se generó JSON ya que no se detectaron residuos.")
            else: st.json(detections)

        with tab4:
            st.subheader("Chatea con la IA sobre esta imagen")
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]): st.markdown(message["content"])
            
            if prompt := st.chat_input("¿Qué más te gustaría saber?"):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Pensando..."):
                        chat_prompt = f"Contexto: {st.session_state.chat_history}. Pregunta: {prompt}"
                        response = run_analysis(model, tokenizer, original_image_pil, chat_prompt)
                        st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
else:
    st.info("Por favor, sube una imagen para comenzar el análisis.")