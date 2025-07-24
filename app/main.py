import sys
import os
import streamlit as st
from PIL import Image
import pandas as pd
from collections import Counter
import json

# --- Robust Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from our modules
from app import utils, gemma_handler, ui_components, config

# --- Page Configuration ---
st.set_page_config(
    page_title="OceanGuard AI - Advanced Mapping",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- State Management ---
if 'project_files' not in st.session_state:
    st.session_state.project_files = []
if 'analysis_summary' not in st.session_state:
    st.session_state.analysis_summary = None
if 'location_reports' not in st.session_state:
    st.session_state.location_reports = {}
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = 'en'
    
# --- Main App ---
st.title("🛰️ OceanGuard AI: Scientific Mapping Platform")
st.markdown("Upload underwater images, analyze debris pollution, and generate environmental impact reports with **Google Gemma 3n**.")

# Load Model
model, tokenizer = gemma_handler.setup_model()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Project Controls")
    
    uploaded_files = st.file_uploader(
        "Upload your transect images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        new_files_added = False
        current_file_names = {f['name'] for f in st.session_state.project_files}
        for file in uploaded_files:
            if file.name not in current_file_names:
                gps_coords = utils.get_gps_from_exif(file.getvalue()) or utils.assign_random_dive_spot()
                st.session_state.project_files.append({
                    "name": file.name,
                    "file": file,
                    "coords": gps_coords,
                    "analysis": None
                })
                new_files_added = True
        if new_files_added:
            st.success(f"{len(uploaded_files)} images loaded and ready to analyze!")
            # Clear old reports when new images are added
            st.session_state.location_reports = {}


    if st.session_state.project_files:
        st.markdown("---")
        
        confidence_threshold_value = st.slider(
            "Filter by Detection Confidence", 0.0, 1.0, 0.5, 0.05,
            help="Show only detections with a confidence score above this value.",
            key="confidence_slider"
        )

        if st.button("Analyze Project Images", type="primary", use_container_width=True):
            if model:
                progress_bar = st.progress(0, "Initiating analysis...")
                for i, file_data in enumerate(st.session_state.project_files):
                    if file_data['analysis'] is None:
                        progress_bar.progress((i+1)/len(st.session_state.project_files), f"Analyzing: {file_data['name']}")
                        image = Image.open(file_data['file']).convert("RGB")
                        detections = gemma_handler.run_debris_detection(model, tokenizer, image)
                        file_data['analysis'] = {"detections": detections}
                progress_bar.progress(1.0, "✅ Analysis complete.")
                # Clear old reports after re-analyzing
                st.session_state.location_reports = {}
            else:
                st.error("Model not loaded. Cannot start analysis.")
        
# --- Data Filtering ---
analyzed_data = []
confidence_threshold = st.session_state.get('confidence_slider', 0.5)

if st.session_state.project_files:
    for file_data in st.session_state.project_files:
        if file_data.get('analysis'):
            filtered_detections = [
                d for d in file_data['analysis']['detections'] 
                if d.get('confidence_score', 0.0) >= confidence_threshold
            ]
            analyzed_data.append({
                **file_data,
                "filtered_analysis": {
                    "detections": filtered_detections,
                    "health_score": utils.calculate_ecosystem_health(filtered_detections)
                }
            })

# --- Main Content Area ---
if not analyzed_data:
    st.info("Welcome to OceanGuard AI. Upload images and click 'Analyze Project Images' to begin.")
else:
    tab_list = ["📊 Main Dashboard", "📸 Individual Reports"]
    if st.session_state.location_reports:
        tab_list.append("📄 AI Generated Reports")
    
    tabs = st.tabs(tab_list)
    
    with tabs[0]: # Main Dashboard
        ui_components.display_main_dashboard(analyzed_data, confidence_threshold)
    
    with tabs[1]: # Individual Reports
        ui_components.display_individual_reports(analyzed_data)

    if len(tabs) > 2: # Generated Reports
        with tabs[2]:
            st.header("Generated Environmental Reports")
            for loc_name, report_text in st.session_state.location_reports.items():
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader(f"📍 Location: {loc_name}")
                    with col2:
                        report_filename = f"report_{loc_name.lower().replace(' ', '_').lower()}_{st.session_state.get('selected_language', 'en')}.md"
                        st.download_button(
                            label="📥 Download",
                            data=report_text.encode('utf-8'),
                            file_name=report_filename,
                            mime="text/markdown",
                            key=f"download_{loc_name}",
                            help=f"Download the report for {loc_name} as a Markdown file."
                        )
                    st.markdown(report_text)


    # --- Sidebar - Export & Report Generation ---
    with st.sidebar:
        st.markdown("---")
        st.header("📥 Export & Report")
        
        report_data_list = []
        for d in analyzed_data:
            for det in d['filtered_analysis']['detections']:
                report_data_list.append({
                    'image_name': d['name'], 'location_name': d['coords']['source'],
                    'latitude': d['coords']['lat'], 'longitude': d['coords']['lon'],
                    'debris_type': det.get('debris_type'), 'material': det.get('material'),
                    'confidence_score': det.get('confidence_score')
                })
        
        if report_data_list:
            df_report = pd.DataFrame(report_data_list)
            csv = df_report.to_csv(index=False).encode('utf-8')
            st.download_button(
               "Download Report (CSV)", csv, f"ocean_guard_report_conf_{confidence_threshold:.2f}.csv",
               "text/csv", use_container_width=True
            )
        else:
            st.info("No data to export with the current filter.")

        st.markdown("---")
        st.subheader("Location-Specific Reports")

        report_language = st.selectbox(
            "Select Report Language",
            options=["en", "es", "fr", "de", "it", "pt"],
            format_func=lambda x: {"en": "English", "es": "Español", "fr": "Français", "de": "Deutsch", "it": "Italiano", "pt": "Português"}.get(x, "English"),
            key="lang_selector"
        )
        
        if st.button("Generate Reports by Location ✍️", use_container_width=True):
            if model and analyzed_data:
                # Group data by location
                locations = {}
                for d in analyzed_data:
                    loc_name = d['coords']['source']
                    if loc_name not in locations:
                        locations[loc_name] = []
                    locations[loc_name].append(d)
                
                st.session_state.location_reports = {} # Reset reports
                
                with st.spinner(f"Gemma is writing reports for {len(locations)} locations..."):
                    for loc_name, loc_data in locations.items():
                        all_loc_detections = [det for d in loc_data for det in d['filtered_analysis']['detections']]
                        avg_loc_health = (sum(d['filtered_analysis']['health_score'] for d in loc_data) / len(loc_data)) if loc_data else 0
                        
                        summary_payload = {
                            "location_name": loc_name,
                            "total_images_analyzed": len(loc_data),
                            "total_debris_detected": len(all_loc_detections),
                            "debris_types_summary": dict(Counter(d.get('debris_type', 'Unknown') for d in all_loc_detections)),
                            "average_health_score": round(avg_loc_health, 2)
                        }
                        
                        report = gemma_handler.generate_summary_report(
                            model, tokenizer, summary_payload, language=report_language
                        )
                        st.session_state.location_reports[loc_name] = report
                st.success("Reports generated successfully! You can view and download them in the AI Generated Reports section.", icon="✅")
                st.rerun() # Rerun to show the new tab with reports
            elif not model:
                st.error("Model not loaded.")
            else:
                st.warning("No analyzed data to generate reports from.")
