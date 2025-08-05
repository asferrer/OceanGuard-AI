# 🛰️ OceanGuard AI: Pollution Mapping Platform

<p align="center">
<img src="https://img.shields.io/badge/Framework-Streamlit-red?style=for-the-badge&logo=streamlit" alt="Framework">
<img src="https://img.shields.io/badge/Model-Gemma%203n-blue?style=for-the-badge&logo=google-gemini" alt="Model">
<img src="https://img.shields.io/badge/Optimized%20by-Unsloth-green?style=for-the-badge" alt="Unsloth">
</p>

**OceanGuard AI** is an interactive web application built with Streamlit that harnesses the power of the Google Gemma 3n multimodal model to detect, classify, and map marine debris from underwater imagery.

This tool is designed for marine biologists, environmental NGOs, and scientific divers, enabling them to transform their underwater photographs into actionable data for ocean conservation.

## ✨ Key Features

- **AI-Powered Debris Detection**: Identifies objects such as plastics, metals, fishing nets, and other waste items in images.
- **Classification & Bounding Box**: Not only detects but also classifies the debris type and its material, drawing a precise bounding box for accurate localization.
- **Geospatial Analysis**: Automatically extracts GPS coordinates from image EXIF metadata and displays them on an interactive 3D map, color-coding locations based on their calculated ecosystem health score.
- **Interactive Dashboard**: Visualizes aggregated statistics, including the most common debris types and overall ecosystem health metrics.
- **AI-Generated Reports**: Leverages Gemma to generate executive summaries on the environmental impact, complete with actionable mitigation recommendations.
- **Data Export**: Allows users to download all findings into a CSV file for further analysis.

## 🛠️ Technology Stack

-   **Demo App Framework:**: [Streamlit](https://streamlit.io/)
-   **Multimodal Model:**: [Google Gemma 3n](https://ai.google.dev/gemma) (optimized via the Unsloth library)
-   **Data Processing:**: Pandas
-   **Image Processing:**: Pillow, Piexif
-   **Geospatial Visualization**: Pydeck

## 🚀 Getting Started

### Prerequisites
- Python 3.9+ (Python 3.10 is recommended)
- Anaconda or Miniconda installed.
- An NVIDIA GPU with CUDA support (recommended for optimal performance).
- Hugging Face credentials configured for model downloading.

### Pasos de Instalación

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/asferrer/OceanGuard-AI.git](https://github.com/asferrer/OceanGuard-AI.git)
    cd OceanGuard-AI
    ```

2.  **Create and activate a Conda environment**
    It's recommended to specify a Python version compatible with the dependencies, like 3.10.
    ```bash
    conda create -n oceanguard python=3.10
    conda activate oceanguard
    ```

3.  **Instalar las dependencias:**
    The  `requirements.txt` ile lists all necessary libraries. To instal `unsloth`, which requires specific PyTorch and xformers builds, it's best to follow their official instructions or use the command below, which works for most setups:

    ```bash
    pip install "unsloth[cu121-py310] @ git+[https://github.com/unsloth/unsloth.git](https://github.com/unsloth/unsloth.git)"
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    From the root directory of the project, execute the following command:
    ```bash
    streamlit run app/app.py
    ```

The application will launch in your default web browser.

## 💡 How to Use the App

1. **Upload Your Images**: Use the sidebar to upload one or more underwater images `(.jpg, .png, .jpeg)`.

2. **Analyze Project Image**s: Click the "Analyze Project Images" button in the sidebar. The app will process each image with Gemma. This may take some time, especially on the first run as the model loads.

3. **Explore the Dashboard**:

- Use the **Geospatial Map** to view the geographical distribution of debris.

- Check the **statistics** to understand the types and quantities of detected debris.

- Navigate to the **Individual Reports** tab to see detailed analyses with bounding boxes for each image.

4. **Generate AI Reports:** In the sidebar, select a language and click "Generate Reports by Location" to have Gemma create environmental impact summaries.

5. **Download Your Data**: Use the "Download Report (CSV)" button to save your findings.
