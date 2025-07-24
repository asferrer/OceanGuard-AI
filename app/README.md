# 🛰️ OceanGuard AI: Plataforma de Mapeo Científico

**OceanGuard AI** es una aplicación web desarrollada con Streamlit que utiliza el poder del modelo de lenguaje multimodal **Google Gemma 3n** para detectar, clasificar y mapear residuos en el fondo marino a partir de imágenes submarinas.

Esta herramienta está diseñada para biólogos marinos, ONGs ambientales y buceadores científicos, permitiéndoles transformar sus fotografías submarinas en datos accionables para la conservación de los océanos.

![Imagen de la app OceanGuard AI en Español]

## ✨ Características Principales

-   **Detección de Residuos por IA**: Identifica objetos como plásticos, metales, redes de pesca y otros residuos en las imágenes.
-   **Clasificación y Bounding Box**: No solo detecta, sino que clasifica el tipo de residuo y su material, y dibuja un cuadro delimitador (bounding box) para una localización precisa.
-   **Análisis Geoespacial**: Extrae automáticamente las coordenadas GPS de los metadatos EXIF de las imágenes y las muestra en un mapa interactivo, coloreando las zonas según su nivel de contaminación.
-   **Dashboard Interactivo**: Visualiza estadísticas agregadas, como los tipos de residuos más comunes y la puntuación de salud del ecosistema.
-   **Generación de Informes por IA**: Utiliza Gemma para redactar resúmenes ejecutivos del impacto ambiental y proponer medidas de mitigación.
-   **Exportación de Datos**: Descarga todos los hallazgos en un archivo CSV para análisis posteriores.

## 🛠️ Stack Tecnológico

-   **Framework de la App**: [Streamlit](https://streamlit.io/)
-   **Modelo Multimodal**: [Google Gemma 3n](https://ai.google.dev/gemma) (a través de la librería Unsloth para optimización)
-   **Procesamiento de Datos**: Pandas
-   **Procesamiento de Imágenes**: Pillow, Piexif
-   **Visualización Geoespacial**: Pydeck

## 🚀 Instalación y Uso

### Prerrequisitos

-   Python 3.9+
-   Una GPU NVIDIA con soporte para CUDA (recomendado para un rendimiento óptimo).
-   Credenciales de Hugging Face para descargar el modelo.

### Pasos de Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/OceanGuard-AI.git](https://github.com/tu-usuario/OceanGuard-AI.git)
    cd OceanGuard-AI
    ```

2.  **Crear un entorno virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instalar las dependencias:**
    El archivo `requirements.txt` contiene todas las librerías necesarias. Para instalar `unsloth`, que requiere compilaciones específicas de PyTorch y xformers, se recomienda seguir sus instrucciones oficiales o usar el siguiente comando que suele funcionar en la mayoría de los casos:

    ```bash
    pip install "unsloth[cu121-py310] @ git+[https://github.com/unsloth/unsloth.git](https://github.com/unsloth/unsloth.git)"
    pip install -r requirements.txt
    ```

4.  **Ejecutar la aplicación:**
    Navega a la carpeta raíz del proyecto y ejecuta:
    ```bash
    streamlit run app/app.py
    ```

La aplicación se abrirá en tu navegador web.

## 💡 Cómo Usar la App

1.  **Sube tus imágenes**: Usa el panel lateral para cargar una o más imágenes (`.jpg`, `.png`) de tu exploración submarina.
2.  **Analiza las Imágenes**: Haz clic en el botón "Analizar Imágenes del Proyecto". La aplicación procesará cada imagen con Gemma. Esto puede tardar un poco, especialmente la primera vez.
3.  **Explora el Dashboard**:
    -   Usa el **mapa** para ver la distribución geográfica de los residuos.
    -   Consulta las **estadísticas** para entender los tipos y cantidades de residuos.
    -   Selecciona imágenes individuales para ver el **informe detallado** con las detecciones.
4.  **Genera un Reporte**: En el panel lateral, haz clic en "Generar Resumen Ejecutivo" para que Gemma cree un informe de impacto.
5.  **Descarga tus Datos**: Usa el botón "Descargar Informe (CSV)" para guardar tus resultados.
