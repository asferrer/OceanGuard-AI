import streamlit as st
import pandas as pd
import pydeck as pdk
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
from app import config

def visualize_bounding_boxes(image: Image.Image, detections: list):
    """Draws bounding boxes on an image based on detection data."""
    if not isinstance(image, Image.Image):
        st.error("Invalid image provided for bounding box visualization.")
        return Image.new('RGB', (500, 500), color = 'white')

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    width, height = img_draw.size

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for det in detections:
        box = det.get('bounding_box')
        label = det.get('debris_type', 'Unknown')
        material = det.get('material', 'default').lower()

        if not isinstance(box, list) or len(box) != 4:
            continue

        color_key = next((k for k in config.COLOR_MAP if k in material or k in label.lower()), "default")
        color = config.COLOR_MAP[color_key]
        
        xmin, ymin, xmax, ymax = box
        draw.rectangle([(xmin * width, ymin * height), (xmax * width, ymax * height)], outline=color, width=4)
        
        text_bbox = draw.textbbox((xmin * width, ymin * height - 22), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((xmin * width, ymin * height - 22), label, fill="black", font=font)
        
    return img_draw

def display_main_dashboard(analyzed_data: list, confidence_threshold: float):
    """Displays the main dashboard with map and statistics (overall or by location)."""
    st.header("Geospatial Project Dashboard")
    
    # --- 1. Geospatial Map (Always shows all locations) ---
    map_df = pd.DataFrame([{
        "lat": d['coords']['lat'], "lon": d['coords']['lon'],
        "health": d['filtered_analysis']['health_score'],
        "debris_count": len(d['filtered_analysis']['detections']),
        "location_name": d['coords']['source'] if d['coords']['source'] != 'EXIF' else 'Custom Location'
    } for d in analyzed_data])
    
    view_state = pdk.ViewState(latitude=map_df['lat'].mean(), longitude=map_df['lon'].mean(), zoom=8, pitch=50)
    
    layer = pdk.Layer(
        'ScatterplotLayer', data=map_df, get_position='[lon, lat]',
        get_color='[(100 - health) * 2.55, health * 2.55, 0, 160]',
        get_radius='(debris_count * 50) + 100', radius_min_pixels=5,
        pickable=True
    )
    
    tooltip = {"html": "<b>{location_name}</b><br/>Health Score: <b>{health}</b>/100<br/>Debris Items: <b>{debris_count}</b>"}
    
    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
    st.pydeck_chart(deck)

    st.markdown("---")

    # --- 2. Location-Specific Statistics ---
    st.header("Environmental Impact Analysis")
    
    location_list = ["Overall"] + sorted(list(set(d['coords']['source'] for d in analyzed_data)))
    selected_location = st.selectbox(
        "View statistics for a specific location:",
        options=location_list
    )

    # Filter data based on selection
    if selected_location == "Overall":
        data_to_display = analyzed_data
        st.subheader("Aggregate Project Statistics")
    else:
        data_to_display = [d for d in analyzed_data if d['coords']['source'] == selected_location]
        st.subheader(f"Statistics for: {selected_location}")
    
    # Calculate stats for the selected scope
    all_filtered_detections = [det for d in data_to_display for det in d['filtered_analysis']['detections']]
    
    if not all_filtered_detections:
        st.success(f"✅ Great news! No debris found for '{selected_location}' with a confidence above {confidence_threshold:.0%}.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            total_images = len(data_to_display)
            total_debris = len(all_filtered_detections)
            avg_health = sum(d['filtered_analysis']['health_score'] for d in data_to_display) / total_images if total_images > 0 else 0
            
            st.metric("Images Analyzed", total_images)
            st.metric("Total Debris Detected", total_debris)
            st.metric("Average Ecosystem Health", f"{avg_health:.1f} / 100")

        with col2:
            st.markdown("**Debris Types Distribution**")
            counts = Counter([d.get('debris_type', 'Unknown') for d in all_filtered_detections])
            df_counts = pd.DataFrame(list(counts.items()), columns=['Debris Type', 'Count']).sort_values("Count", ascending=False)
            st.bar_chart(df_counts.set_index('Debris Type'))

def display_individual_reports(analyzed_data: list):
    """Displays a detailed report for a single selected image."""
    st.header("Explore Individual Reports")
    
    image_names = [d['name'] for d in analyzed_data]
    selected_image_name = st.selectbox(
        "Select an image to see its detailed report:",
        options=image_names,
        key="individual_report_selector"
    )
    
    selected_data = next((d for d in analyzed_data if d['name'] == selected_image_name), None)
    
    if selected_data:
        st.subheader(f"Detailed Report: {selected_data['name']}")
        
        report_image = Image.open(selected_data['file']).convert("RGB")
        detections = selected_data['filtered_analysis']['detections']
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(report_image, caption="Original Image", use_container_width=True)
        with col2:
            annotated_image = visualize_bounding_boxes(report_image, detections)
            st.image(annotated_image, caption="Image with Detections", use_container_width=True)
            
        st.subheader("Environmental Impact Panel")
        if not detections:
            st.success("✅ No debris detected with the current confidence filter.")
        else:
            for det in detections:
                debris_type = det.get('debris_type', 'Unknown')
                material = det.get('material', 'Default').lower()

                normalized_debris_type = debris_type.lower().replace('_', ' ')

                confidence = det.get('confidence_score', 0.0)
                
                impact_key = next(
                    (k for k in config.DEBRIS_IMPACT_DATA if k.lower().replace('_', ' ') in normalized_debris_type), 
                    "Default"
                )
                impact_info = config.DEBRIS_IMPACT_DATA[impact_key]
                
                with st.container(border=True):
                    col_icon, col_info = st.columns([1, 5])
                    with col_icon:
                        st.markdown(f"<p style='font-size: 48px; text-align: center;'>{impact_info['icon']}</p>", unsafe_allow_html=True)
                    with col_info:
                        st.subheader(debris_type)
                        st.markdown(f"**🎯 Detection Confidence:** {confidence:.1%}")
                        st.markdown(f"**⏳ Degradation Time:** {impact_info['degradation']} | **🚨 Primary Risk:** {impact_info['risk']}")

