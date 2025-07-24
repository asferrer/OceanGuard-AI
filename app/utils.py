import piexif
import random
import json
import re
import logging
from . import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_gps_from_exif(image_bytes):
    """Extracts GPS coordinates from image EXIF data."""
    try:
        exif_dict = piexif.load(image_bytes)
        gps_info = exif_dict.get("GPS")
        if not gps_info:
            return None

        def convert_to_degrees(value):
            d, m, s = [float(v[0]) / float(v[1]) for v in value]
            return d + (m / 60.0) + (s / 3600.0)

        lat = convert_to_degrees(gps_info[piexif.GPSIFD.GPSLatitude])
        if gps_info[piexif.GPSIFD.GPSLatitudeRef] == b'S':
            lat = -lat

        lon = convert_to_degrees(gps_info[piexif.GPSIFD.GPSLongitude])
        if gps_info[piexif.GPSIFD.GPSLongitudeRef] == b'W':
            lon = -lon

        return {"lat": lat, "lon": lon, "source": "EXIF"}
    except Exception as e:
        logging.warning(f"Could not extract EXIF GPS data: {e}")
        return None

def assign_random_dive_spot():
    """Assigns a random dive spot from the predefined list."""
    spot = random.choice(config.DIVE_SITES)
    return {"lat": spot["lat"], "lon": spot["lon"], "source": spot["name"]}

def calculate_ecosystem_health(detections):
    """Calculates an ecosystem health score based on detected debris."""
    if not detections:
        return 100
    
    total_risk_score = 0
    for d in detections:
        # Get the detected type, default to 'default', and make it lowercase for consistent matching
        debris_type_str = d.get('debris_type', 'default').lower()
        
        # Find the best matching key from our config, ignoring case.
        # The `k.lower() in debris_type_str` part allows for partial matches (e.g., 'net' in 'fishing_net').
        impact_key = next(
            (k for k in config.DEBRIS_IMPACT_DATA if k.lower() in debris_type_str), 
            "Default" # Fallback to the correctly capitalized "Default" key if no match is found
        )
        
        # Safely access the dictionary using the found key.
        total_risk_score += config.DEBRIS_IMPACT_DATA[impact_key].get('risk_score', 3)
        
    health_score = max(0, 100 - total_risk_score)
    return health_score

def extract_json_from_response(text):
    """
    Robustly extracts a JSON list from the model's text response,
    handling cases with or without markdown code blocks.
    """
    # Pattern to find JSON within ```json ... ```
    match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if match:
        json_str = match.group(1)
    else:
        # If no markdown block, assume the whole string is the JSON
        # and find the start of the list.
        json_str = text[text.find('['): text.rfind(']') + 1]

    try:
        # Clean up potential artifacts and load
        detections = json.loads(json_str.strip())
        if isinstance(detections, list):
            # Sanitize debris_type field
            for det in detections:
                if 'debris_type' in det and isinstance(det['debris_type'], str):
                    det['debris_type'] = det['debris_type'].strip().title()
            return detections
        return []
    except (json.JSONDecodeError, TypeError) as e:
        logging.error(f"Failed to decode JSON from response: {e}\nResponse text: '{text}'")
        return []
    
def robust_json_parser(text: str):
    """
    Parses a JSON object from a string that might contain extra text.
    It also attempts to fix common syntax errors like missing commas between objects in a list.
    """
    # Find the JSON block (list or object) using a greedy regex
    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    
    if not match:
        logging.warning("No JSON object found in the model's response.")
        return []

    json_str = match.group(0)

    try:
        # First, try to parse the string as is.
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # If parsing fails, it might be due to a common LLM error like a missing comma.
        logging.warning(f"Initial JSON parse failed: {e}. Attempting to fix...")
        
        try:
            # Attempt to fix the most common error: a missing comma between JSON objects in a list.
            # This regex finds a '}' followed by optional whitespace and then a '{',
            # and inserts a comma between them.
            fixed_json_str = re.sub(r'}\s*{', '}, {', json_str)
            
            # Try parsing the fixed string.
            return json.loads(fixed_json_str)
        except json.JSONDecodeError as e_fix:
            # If the fix also fails, log the error with the original string and return an empty list.
            logging.error(f"Failed to parse JSON even after attempting a fix: {e_fix}")
            logging.error(f"Original problematic JSON string: {json_str}")
            return []