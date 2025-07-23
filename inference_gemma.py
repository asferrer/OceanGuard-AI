"""
Script para la Detección y Conteo de Residuos Marinos usando Gemma 3n.

Este script ofrece dos modos de operación:
1. 'detect': Identifica residuos, su material y dibuja un bounding box preciso.
2. 'count': Realiza un conteo rápido de los tipos de residuos en la imagen.
"""
import torch
from unsloth import FastLanguageModel
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import json
import os
from collections import Counter

# Configuraciones de Torch
torch._dynamo.disable()
torch._dynamo.config.cache_size_limit = 99999999999999999999

def setup_model():
    """
    Carga y configura el modelo Gemma 3n y el tokenizador desde Hugging Face.
    """
    print("🚀 Cargando el modelo Gemma 3n... Esto puede tardar unos minutos.")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-3n-e2b-it",
        max_seq_length=4096,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map={'': 0}
    )
    return model, tokenizer

def load_image_from_path(path: str) -> Image.Image:
    """
    Carga una imagen desde una ruta de archivo local.
    """
    try:
        image = Image.open(path).convert("RGB")
        print(f"✅ Imagen cargada correctamente desde la ruta local: {path}")
        return image
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo en la ruta: {path}")
        return None
    except Exception as e:
        print(f"❌ Error al abrir la imagen local: {e}")
        return None

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
        "nylon": "#9400D3", "synthetic": "#9400D3", "fishing": "#FF8C00",
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
        if not box or len(box) != 4: continue
        
        color_key = next((key for key in color_map if key in material), "default")
        color = color_map[color_key]

        xmin, ymin, xmax, ymax = box
        draw.rectangle([(xmin * width, ymin * height), (xmax * width, ymax * height)], outline=color, width=4)
        text_bbox = draw.textbbox((xmin * width, ymin * height - 22), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((xmin * width, ymin * height - 22), label, fill="black", font=font)
        
    return img_draw

def analyze_seabed_image(model, tokenizer, image_path: str, mode: str = 'detect'):
    """
    Analiza una imagen del fondo marino en el modo especificado ('detect' o 'count').
    """
    print("-" * 50)
    print(f"🖼️  Analizando la imagen: {image_path} (Modo: {mode})")

    image = load_image_from_path(image_path)
    if image is None:
        return None

    if mode == 'detect':
        prompt_text = (
            "You are a precise, expert marine debris detection system. "
            "Analyze the image and identify ALL man-made waste items. "
            "For each item, provide a TIGHT bounding box. "
            "Return a JSON list of objects. Each object must have 'debris_type', 'material', and 'bounding_box' ([x_min, y_min, x_max, y_max]). "
            "Your response MUST be ONLY the JSON list, without any other text or markdown."
        )
    elif mode == 'count':
        prompt_text = (
            "You are an efficient marine debris counter. Your task is to analyze the image and count all man-made waste items. "
            "Group the items by their type and return a single JSON object where keys are the 'debris_type' and values are the integer count. "
            "Example: {\"Plastic Bottle\": 2, \"Metal Can\": 1, \"Fishing Net\": 1}. "
            "Your response MUST be ONLY the JSON object, without any other text or markdown."
        )
    else:
        print(f"❌ Error: Modo '{mode}' no reconocido. Use 'detect' o 'count'.")
        return None

    prompt_content = [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]
    messages = [{"role": "user", "content": prompt_content}]

    try:
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to("cuda")

        output_tokens = model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
        response_text = tokenizer.batch_decode(output_tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        print("\n🤖 Respuesta del Modelo (JSON crudo):")
        print(response_text)
        
        json_str = response_text.strip().replace("```json", "").replace("```", "")
        parsed_json = json.loads(json_str)

        # ### NUEVO: Procesamiento según el modo ###
        if mode == 'detect':
            print(f"\n✅ Se encontraron {len(parsed_json)} objetos para detectar.")
            annotated_image = visualize_bounding_boxes(image, parsed_json)
            base, _ = os.path.splitext(image_path)
            output_path = f"{base}_analyzed.jpg"
            annotated_image.save(output_path)
            print(f"💾 Imagen con anotaciones guardada en: {output_path}")
            return parsed_json
        
        elif mode == 'count':
            print("\n📊 Resumen de Conteo para esta imagen:")
            if not parsed_json:
                print("  No se encontraron residuos.")
            else:
                for debris_type, count in parsed_json.items():
                    print(f"  - {debris_type}: {count}")
            return parsed_json

    except json.JSONDecodeError:
        print("❌ Error: La respuesta del modelo no es un JSON válido.")
    except Exception as e:
        print(f"❌ Ha ocurrido un error durante la inferencia: {e}")
    finally:
        torch.cuda.empty_cache()
    
    return None


if __name__ == "__main__":
    gemma_model, gemma_tokenizer = setup_model()

    example_images = [
        "../mis_imagenes_submarinas/HPD2032OUT0050.jpg",
        "../mis_imagenes_submarinas/HPD1938HDTV20368.jpg",
        "../mis_imagenes_submarinas/2K0126IN0025Hp03-05.jpg",
        "../mis_imagenes_submarinas/HPD1814HDTV0820.jpg",
    ]
    
    # --- DEMO MODO CONTEO Y REPORTE AGREGADO ---
    print("\n" + "="*20 + " INICIANDO MODO CONTEO " + "="*20)
    total_counts = Counter()
    for path in example_images:
        if os.path.exists(path):
            counts = analyze_seabed_image(gemma_model, gemma_tokenizer, path, mode='count')
            if counts:
                total_counts.update(counts)
        else:
            print(f"⚠️  Advertencia: El archivo no existe, saltando: {path}")

    print("\n" + "="*20 + " REPORTE FINAL DE CONTEO " + "="*20)
    print("Se ha generado el siguiente resumen agregado de todas las imágenes:")
    if not total_counts:
        print("No se encontraron residuos en ninguna imagen.")
    else:
        for debris_type, count in total_counts.items():
            print(f"  - Total de '{debris_type}': {count}")
    print("="*62)

    # --- DEMO MODO DETECCIÓN---
    print("\n" + "="*20 + " INICIANDO MODO DETECCIÓN " + "="*20)
    if os.path.exists(example_images[0]):
        analyze_seabed_image(gemma_model, gemma_tokenizer, example_images[0], mode='detect')
    print("="*64)