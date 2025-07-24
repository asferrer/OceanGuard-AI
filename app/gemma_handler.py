import torch
from unsloth import FastLanguageModel
from PIL import Image
import streamlit as st
import logging
import json
from app import utils, config

torch._dynamo.disable()
torch._dynamo.config.cache_size_limit = 99999999999999999999

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_resource
def setup_model():
    """Loads and configures the Gemma 3n model and tokenizer."""
    try:
        logging.info("--- Loading Gemma 3n model (this will happen only once) ---")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gemma-3n-e2b-it",
            max_seq_length=4096,
            dtype=torch.float16,
            load_in_4bit=True,
            device_map={'': 0}
        )
        torch.cuda.empty_cache()
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        logging.error(f"Model loading failed: {e}")
        return None, None

def run_debris_detection(model, tokenizer, image: Image.Image):
    """Runs debris detection analysis on a single image."""

    valid_classes = list(config.DEBRIS_IMPACT_DATA.keys())
    class_list_str = ", ".join([f"'{cls}'" for cls in valid_classes])

    prompt_text = (
        "You are a precise, pixel-perfect marine debris detection system. "
        "Analyze the image and identify ALL man-made waste items. "
        "Return a JSON list of objects. Each object must have:\n"
        f"1. 'debris_type': The specific type of object. It MUST be one of the following: [{class_list_str}].\n"
        "2. 'material': The likely material (e.g., 'Plastic', 'Metal').\n"
        "3. 'confidence_score': A float from 0.0 to 1.0 indicating your certainty.\n"
        "4. 'bounding_box': A list of four normalized coordinates [x_min, y_min, x_max, y_max]. "
        "CRITICAL: The bounding box must be extremely precise, tightly enclosing the object with no extra padding. "
        "Do not include background, water, or shadows. Aim for pixel-perfect accuracy.\n"
        "Your response MUST be ONLY the JSON list. Do not add any extra text or markdown."
    )
    prompt_content = [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]
    messages = [{"role": "user", "content": prompt_content}]
    
    try:
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        output_tokens = model.generate(**inputs, max_new_tokens=2048, temperature=0.1)
        response_text = tokenizer.batch_decode(output_tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        torch.cuda.empty_cache()
        
        detections = utils.robust_json_parser(response_text)
        # Normalize labels
        if isinstance(detections, list):
            for det in detections:
                if 'debris_type' in det and isinstance(det['debris_type'], str):
                    det['debris_type'] = det['debris_type'].strip().title()
        return detections if isinstance(detections, list) else []
    except Exception as e:
        logging.error(f"Error during debris detection inference: {e}")
        return []

def generate_summary_report(model, tokenizer, analysis_summary: dict, language: str = "en"):
    """Generates a multilingual summary report."""
    summary_json = json.dumps(analysis_summary, indent=2)

    language_map = {
        "en": "English", "es": "Spanish", "fr": "French",
        "de": "German", "it": "Italian", "pt": "Portuguese"
    }
    target_language = language_map.get(language, "English")

    prompt_text = (
        f"You are a senior marine biologist and data analyst. Based on the following JSON summary of an underwater "
        f"debris survey, write a concise and impactful summary in **{target_language}**. The tone should be formal but accessible. "
        f"Start with a clear overview, then highlight the key findings (e.g., most common debris type, areas of concern), "
        f"and conclude with 2-3 actionable recommendations for conservation efforts.\n\n"
        f"**Analysis Data:**\n{summary_json}\n\n"
        f"**Your Report (in {target_language}):**"
    )
    
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]

    try:
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        output_tokens = model.generate(**inputs, max_new_tokens=1024, temperature=0.5, do_sample=True)
        report_text = tokenizer.batch_decode(output_tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return report_text
    except Exception as e:
        logging.error(f"Error during summary report generation: {e}")
        return "Failed to generate the report due to an internal error."
    finally:
        # Clean up GPU memory
        torch.cuda.empty_cache()