# -*- coding: utf-8 -*-
"""
Script para la Detección de Residuos Marinos usando Gemma 3n y Unsloth.

Este script utiliza el modelo multimodal Gemma 3n de Google, optimizado con Unsloth,
para analizar imágenes del fondo marino e identificar diferentes tipos de residuos
de origen humano.

Dependencias:
- torch
- unsloth
- requests
- Pillow

Instalación (asegúrate de tener un entorno con CUDA si usas GPU):
pip install "unsloth[gemma3n-patch] @ git+https://github.com/unslothai/unsloth.git"
pip install "gemma3n-build @ https://storage.googleapis.com/gemma-3n/gemma3n-0.0.1-py3-none-any.whl"
pip install requests Pillow

Ejecución:
python tu_script.py
"""
import torch
from unsloth import FastLanguageModel, FastVisionModel
from PIL import Image, ImageDraw, ImageFont
from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
import json
import os

torch._dynamo.disable()
torch._dynamo.config.cache_size_limit = 99999999999999999999

def setup_model():
    """
    Carga y configura el modelo Gemma 3n y el tokenizador desde Hugging Face.
    Utiliza cuantización de 4 bits para optimizar el uso de memoria.
    """
    print("🚀 Cargando el modelo Gemma 3n... Esto puede tardar unos minutos.")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-3n-e2b-it",  # e2b = 2B params, it = Instruction Tuned
        max_seq_length=4096,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map={'': 0}
    )
    return model, tokenizer

def load_image_from_url(url: str) -> Image.Image:
    """
    Descarga una imagen desde una URL y la convierte a un objeto PIL.Image.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lanza un error si la petición falla
        image = Image.open(BytesIO(response.content))
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar la imagen: {e}")
        return None

def load_image_from_path(path: str) -> Image.Image:
    """
    Carga una imagen desde una ruta de archivo local.
    """
    try:
        image = Image.open(path)
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
    Dibuja los bounding boxes y etiquetas sobre una imagen usando colores
    diferentes para cada material.

    Args:
        image (Image.Image): La imagen original.
        detections (list): Una lista de diccionarios, donde cada uno representa
                           una detección con 'bounding_box' y 'debris_type'.

    Returns:
        Image.Image: La imagen con las visualizaciones dibujadas.
    """
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    width, height = img_draw.size

    # Mapa de colores para visualizar diferentes tipos de materiales.
    color_map = {
        "plastic": "#FF1493",  # DeepPink
        "metal": "#1E90FF",   # DodgerBlue
        "fabric": "#32CD32",  # LimeGreen
        "textile": "#32CD32", # LimeGreen
        "rubber": "#FFD700",  # Gold
        "glass": "#00CED1",   # DarkTurquoise
        "nylon": "#9400D3",   # DarkViolet
        "synthetic": "#9400D3",# DarkViolet
        "fishing": "#FF8C00", # DarkOrange
        "default": "#FFFFFF"  # White for unknown
    }

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for det in detections:
        box = det.get('bounding_box')
        label = det.get('debris_type', 'Unknown')
        material = det.get('material', 'default').lower()

        if not box or len(box) != 4:
            continue
            
        # Encontrar el color correspondiente al material.
        # Si no se encuentra una coincidencia directa, usa el color por defecto.
        color_key = "default"
        for key in color_map:
            if key in material:
                color_key = key
                break
        color = color_map.get(color_key, color_map["default"])

        # Desnormalizar las coordenadas del bounding box
        xmin, ymin, xmax, ymax = box
        abs_xmin = xmin * width
        abs_ymin = ymin * height
        abs_xmax = xmax * width
        abs_ymax = ymax * height

        # Dibujar el rectángulo
        draw.rectangle([(abs_xmin, abs_ymin), (abs_xmax, abs_ymax)], outline=color, width=4)

        # Dibujar la etiqueta de texto con un fondo
        text_bbox = draw.textbbox((abs_xmin, abs_ymin - 22), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((abs_xmin, abs_ymin - 22), label, fill="black", font=font)
        
    return img_draw

def detect_marine_waste(model, tokenizer, image_url: str):
    """
    Realiza la inferencia sobre una imagen para detectar residuos marinos.

    Args:
        model: El modelo Gemma 3n cargado.
        tokenizer: El tokenizador asociado al modelo.
        image_url (str): La URL de la imagen a analizar.
    """
    print("-" * 50)
    print(f"🖼️  Analizando la imagen desde: {image_url}")

    # Determinar si la fuente es una URL o una ruta local
    if image_url.startswith('http://') or image_url.startswith('https://'):
        print(f"ℹ️  Detectada fuente de imagen como URL.")
        image = load_image_from_url(image_url)
    else:
        print(f"ℹ️  Detectada fuente de imagen como ruta local.")
        image = load_image_from_path(image_url)

    if image is None:
        return

    # El prompt está diseñado para guiar al modelo a enfocarse en los objetos de desecho.
    # Es más efectivo que preguntar genéricamente "¿Qué ves?".
    prompt_text = (
        "You are a precise, expert marine debris detection system. "
        "Your task is to analyze the image and identify ALL man-made waste items. "
        "For each item, provide a TIGHT bounding box that fits snugly around the object, excluding as much background (water, sand, rocks) as possible. "
        "Return a JSON list of objects. Each object must have:\n"
        "1. 'debris_type': The specific type of object (e.g., 'Plastic Bottle', 'Fishing Net').\n"
        "2. 'material': The likely material (e.g., 'Plastic', 'Metal', 'Nylon').\n"
        "3. 'bounding_box': A list of four normalized coordinates [x_min, y_min, x_max, y_max] representing the top-left and bottom-right corners of the tight bounding box.\n"
        "Your response MUST be ONLY the JSON list, without any other text, explanations, or markdown formatting."
    )
    prompt = [
        { "type": "image", "image" : image},
        { "type": "text", "text": prompt_text}
    ]

    messages = [{"role": "user", "content": prompt}]

    # Generar la respuesta usando la función de chat del modelo
    try:
        inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True, # Must add for generation
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
        ).to("cuda")
        output_tokens = model.generate(
            **inputs,
            temperature = 0.1, top_p = 0.95, top_k = 64,
            max_new_tokens = 1024
        )

        response_text = tokenizer.batch_decode(output_tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

        print("\n❓ Pregunta:")
        print(prompt_text)
        
        print("\n🤖 Respuesta del Modelo:")
        print(response_text)

        # Procesar y visualizar la respuesta
        try:
            # Limpiar la respuesta para que sea un JSON válido
            json_str = response_text.strip().replace("```json", "").replace("```", "")
            detections = json.loads(json_str)
            
            print(f"\n✅ Se encontraron {len(detections)} objetos.")

            # Dibujar los bounding boxes
            annotated_image = visualize_bounding_boxes(image, detections)
            
            # Guardar la imagen anotada
            base, ext = os.path.splitext(image_url)
            output_path = f"{base}_analyzed.jpg"
            annotated_image.save(output_path)
            print(f"💾 Imagen con anotaciones guardada en: {output_path}")

        except json.JSONDecodeError:
            print("❌ Error: La respuesta del modelo no es un JSON válido.")
        except Exception as e:
            print(f"❌ Error al procesar las detecciones: {e}")

    except Exception as e:
        print(f"Ha ocurrido un error durante la inferencia: {e}")

    finally:
        # Liberar memoria de la GPU es una buena práctica
        if 'inputs' in locals():
            del inputs
        if 'output_tokens' in locals():
            del output_tokens
        torch.cuda.empty_cache()
        
    print("-" * 50)


if __name__ == "__main__":
    # Cargar el modelo una sola vez
    gemma_model, gemma_tokenizer = setup_model()

    # Lista de imágenes de ejemplo para analizar
    example_images = [
        "../mis_imagenes_submarinas/HPD2032OUT0050.jpg",
        "../mis_imagenes_submarinas/HPD1938HDTV20368.jpg",
        #"../mis_imagenes_submarinas/2K0126IN0025Hp03-05.jpg",
        #"../mis_imagenes_submarinas/HPD1814HDTV0820.jpg",
        #"../mis_imagenes_submarinas/HPD2027HDTV12026.jpg",
        #"../mis_imagenes_submarinas/HPD2027HDTV11708.jpg"
    ]

    # Iterar sobre las imágenes y realizar la detección
    for path in example_images:
        if os.path.exists(path):
            detect_marine_waste(gemma_model, gemma_tokenizer, path)
        else:
            print(f"⚠️  Advertencia: El archivo no existe, saltando: {path}")

