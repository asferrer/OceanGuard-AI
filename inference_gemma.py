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
from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer

torch._dynamo.reset()

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
        "You are an expert in marine biology and environmental science. "
        "Analyze this image of the seabed and identify any man-made waste or debris. "
        "Describe the types of materials you see (e.g., plastic, metal, rubber) and the specific objects if possible."
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
            temperature = 1.0, top_p = 0.95, top_k = 64,
        )

        response_text = tokenizer.batch_decode(output_tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

        print("\n❓ Pregunta:")
        print(prompt_text)
        
        print("\n🤖 Respuesta del Modelo:")
        print(response_text.strip())

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
        "mis_imagenes_submarinas/2K0126IN0025Hp03-05.jpg",
        "mis_imagenes_submarinas/HPD1814HDTV0820.jpg",
        "mis_imagenes_submarinas/HPD2027HDTV12026.jpg",
        "mis_imagenes_submarinas/HPD2027HDTV11708.jpg"
    ]

    # Iterar sobre las imágenes y realizar la detección
    for url in example_images:
        detect_marine_waste(gemma_model, gemma_tokenizer, url)

    # Ejemplo con una imagen local (descomentar para usar)
    # print("\n--- Analizando imagen local ---")
    # try:
    #     local_image_path = "ruta/a/tu/imagen.jpg"
    #     local_image = Image.open(local_image_path)
    #
    #     # Reutilizamos la lógica de la función `detect_marine_waste`
    #     prompt = (
    #         "You are an expert in marine biology. Describe any man-made waste in this image of the seabed."
    #     )
    #     messages = [{"role": "user", "content": prompt}]
    #     images = [local_image]
    #
    #     response = gemma_model.chat(gemma_tokenizer, messages=messages, images=images)
    #     print("\n❓ Pregunta:")
    #     print(prompt)
    #     print("\n🤖 Respuesta del Modelo:")
    #     print(response)
    #
    # except FileNotFoundError:
    #     print(f"Asegúrate de que la imagen exista en la ruta: {local_image_path}")
    # except Exception as e:
    #     print(f"Error al procesar la imagen local: {e}")
