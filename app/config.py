import torch

# --- Model Configuration ---
MODEL_NAME = "unsloth/gemma-3n-e4b-it"
MAX_SEQ_LENGTH = 4096
DTYPE = torch.float16
LOAD_IN_4BIT = True

# --- Prompts ---
DETECTION_PROMPT = (
    "You are a precise, expert marine debris detection system. "
    "Analyze the image and identify ALL man-made waste items. "
    "Return a JSON list of objects. Each object must have:\n"
    "1. 'debris_type': The specific type of object (e.g., 'Plastic Bottle', 'Fishing Net').\n"
    "2. 'material': The likely material (e.g., 'plastic', 'metal', 'nylon').\n"
    "3. 'confidence_score': A float from 0.0 to 1.0 indicating your certainty.\n"
    "4. 'bounding_box': A list of four normalized coordinates [x_min, y_min, x_max, y_max].\n"
    "Your response MUST be ONLY the JSON list, without any introductory text or code block markers."
)

SUMMARY_REPORT_PROMPT_TEMPLATE = (
    "You are a senior marine biologist and environmental analyst. Based on the following summary data "
    "from an underwater survey, write a concise executive summary report in Markdown format. "
    "The report should be addressed to a regional environmental agency.\n\n"
    "The report must include:\n"
    "1.  **Overview**: A brief summary of the findings.\n"
    "2.  **Key Findings**: Highlight the most common types of debris and the locations with the highest concentration.\n"
    "3.  **Environmental Impact Assessment**: Briefly explain the potential risks these specific debris types pose to the local marine ecosystem (e.g., ghost fishing from nets, microplastic pollution).\n"
    "4.  **Recommendations**: Suggest 2-3 actionable steps for mitigation and cleanup efforts in the area.\n\n"
    "**Survey Data:**\n"
    "```json\n"
    "{summary_data}\n"
    "```\n\n"
    "Begin the report with '### Executive Summary: OceanGuard AI Survey'."
)


# Data on the environmental impact of different types of debris
DEBRIS_IMPACT_DATA = {
    # Specific Plastic Items
    "Bottle": {"degradation": "450+ years", "risk": "Ingestion by marine life, microplastic pollution.", "icon": "🍾", "risk_score": 8},
    "Plastic_Bag": {"degradation": "20+ years", "risk": "Suffocation hazard for turtles and marine mammals.", "icon": "🛍️", "risk_score": 9},
    "Food_Wrapper": {"degradation": "50-80 years", "risk": "Commonly ingested, leading to internal injuries.", "icon": "🍫", "risk_score": 7},
    "Fishing_Net": {"degradation": "600+ years", "risk": "Ghost fishing, entanglement of large animals.", "icon": "🕸️", "risk_score": 10},
    "Rope": {"degradation": "50-100 years", "risk": "Entanglement of corals and marine animals.", "icon": " रस्सी", "risk_score": 8},
    
    # Specific Metal Items
    "Can": {"degradation": "50-200 years", "risk": "Physical injury from sharp edges, chemical leaching.", "icon": "🥫", "risk_score": 6},
    
    # Specific Rubber Items
    "Tire": {"degradation": "2000+ years", "risk": "Leaches toxic chemicals (zinc, heavy metals) into the ecosystem.", "icon": "�", "risk_score": 9},
    
    # Generic Material Categories (as fallbacks)
    "Plastic": {"degradation": "450+ years", "risk": "Ingestion, microplastics.", "icon": "💧", "risk_score": 8},
    "Metal": {"degradation": "50-200 years", "risk": "Chemical leaching.", "icon": "🔩", "risk_score": 6},
    "Glass": {"degradation": "1,000,000+ years", "risk": "Physical injury hazard.", "icon": "🏺", "risk_score": 4},
    "Fabric": {"degradation": "1-200+ years", "risk": "Releases microfibers, entanglement.", "icon": "👕", "risk_score": 5},
    "Rubber": {"degradation": "50-80 years", "risk": "Leaches toxic chemicals.", "icon": "🧤", "risk_score": 7},
    "Default": {"degradation": "Unknown", "risk": "Unknown environmental risk.", "icon": "❓", "risk_score": 3}
}

COLOR_MAP = {
    "bottle": "#FF1493",
    "plastic_bag": "#FF69B4",
    "food_wrapper": "#FFC0CB",
    "plastic": "#FF1493",
    "can": "#1E90FF",
    "metal": "#1E90FF",
    "fabric": "#32CD32",
    "tire": "#FFD700",
    "rubber": "#FFD700",
    "glass": "#00CED1",
    "fishing_net": "#FF8C00",
    "net": "#FF8C00",
    "rope": "#FFA500",
    "default": "#FFFFFF"
}

# Pre-defined coordinates for dive sites if images lack EXIF data
DIVE_SITES = [
    # Tenerife dive sites
    {"name": "El Puertito de Adeje",                "lat": 28.0863,   "lon": -16.7942},
    {"name": "Pecio de Tabaiba",                    "lat": 28.3883,   "lon": -16.3205},
    {"name": "Las Galletas",                        "lat": 28.0050,   "lon": -16.6600},
    {"name": "Baja de los Guanches (Garachico)",    "lat": 28.3758,   "lon": -16.7664},
    {"name": "Radazul",                             "lat": 28.4069,   "lon": -16.3361},

    # Dive sites worldwide
    {"name": "Isla de Benidorm",                    "lat": 38.5013,     "lon": -0.1248},
    {"name": "Great Blue Hole (Belize)",            "lat": 17.316010,   "lon": -87.535103},
    {"name": "Shark & Yolanda Reef (Ras Mohammed, Egypt)", "lat": 27.72521, "lon": 34.25889},
    {"name": "SS Thistlegorm (Egypt)",             "lat": 27.808497,   "lon": 33.918664},
    {"name": "SS Yongala (Australia)",               "lat": -19.3045,    "lon": 147.6218},
    {"name": "Sipadan (Malaysia)",                   "lat": 4.119467,    "lon": 118.629817},
    {"name": "Cocos Island (Costa Rica)",           "lat": 5.5180,      "lon": -87.069667},
    {"name": "Blue Corner Wall (Palau)",            "lat": 7.133611,    "lon": 134.220278},
    {"name": "Norman Reef (Great Barrier Reef, Australia)", "lat": -16.426719, "lon": 145.994347},
    {"name": "USAT Liberty (Tulamben, Indonesia)",  "lat": -8.27396,    "lon": 115.59307},
    {"name": "Dos Ojos Cenote (Mexico)",            "lat": 20.32483,    "lon": -87.39059}
]