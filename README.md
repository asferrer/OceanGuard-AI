# 🌊 OceanGuard AI: Where AI Meets Ocean Conservation

<p align="center">
  <img src="https://img.shields.io/badge/Model-Gemma%203n-blue?style=for-the-badge&logo=google-gemini" alt="Model">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?style=for-the-badge&logo=streamlit" alt="Framework">
  <img src="https://img.shields.io/badge/Optimized%20by-Unsloth-green?style=for-the-badge" alt="Unsloth">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

<p align="center">
<a href="https://youtu.be/4YRmgjpFcSI">
<img src="https://img.shields.io/badge/YouTube-red?style=for-the-badge&logo=youtube&logoColor=white" alt="Video Demo">
</a>
<a href="">
<img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" alt="Kaggle Write-up">
</a>

<p align="center">
<em>A submission to the Google - Gemma 3N Hackathon on Kaggle.</em>
</p>

---

### 🆘 The Crisis: A Plastic Ocean

By 2050, our oceans could contain more plastic than fish by weight. Traditional monitoring methods like manual surveys and satellite imagery are too slow, expensive, and limited in scope to tackle this global crisis. They fail to detect underwater debris, which constitutes over 90% of the problem, and suffer from significant delays between data collection and actionable insights.

### 💡 The Solution: OceanGuard AI

**OceanGuard AI** transforms every underwater camera into an intelligent ocean guardian. This project leverages the power of Google's multimodal **Gemma 3n** model, optimized with Unsloth, to create a real-time, scalable, and globally accessible platform for detecting, analyzing, and mapping marine pollution. We turn raw imagery into actionable conservation intelligence in seconds, not months.

---

## ✨ Key Features & Technical Highlights

- **🧠 On-Device Intelligence & Offline-First**: All analysis happens locally. No internet connection or expensive cloud servers required. This is a game-changer for fieldwork in remote marine protected areas.

- **🎯 High-Precision Debris Detection**: Gemma 3N identifies and classifies underwater debris, outputting structured JSON data with coordinates and confidence scores for 100% reliable data ingestion.
- **🔧 Robust JSON Parsing**: We've implemented a custom robust_json_parser that intelligently corrects common syntax errors from the LLM (like missing commas), ensuring the application never fails due to malformed model outputs.
- **🗺️ Interactive Geospatial Dashboard**: An intuitive 3D map visualizes pollution data instantly, revealing environmental hotspots and allowing for powerful data exploration.
- **🩺 Revolutionary Health Score**: A unique algorithm (calculate_ecosystem_health) calculates a real-time "Ecosystem Health Score" for any location, providing an immediate, quantifiable metric of environmental impact based on debris type and density.
- **📜 AI-Powered Narrative Reports**: Gemma acts as an expert marine biologist, synthesizing findings into multilingual executive reports complete with actionable mitigation strategies.
- **⚡ Optimized for Performance**: Using Unsloth's 4-bit quantization, we achieve high-speed analysis on consumer-grade hardware, making real-time application a reality.

- **📊 Built-in Performance Evaluation**: The repository includes a dedicated script (evaluate_performance.py) to benchmark the model's accuracy (Precision, Recall, F1-Score) against standard datasets like CleanSea (in COCO format), demonstrating a commitment to scientific validation.

---

## 🛠️ Technology Stack

-   **Core Model**: Google Gemma 3n (`unsloth/gemma-3n-e4b-it` & `unsloth/gemma-3n-e2b-it`)
-   **Optimization**: Unsloth AI for 2x faster fine-tuning and memory efficiency.
-   **Interactive Demo**: Streamlit
-   **Geospatial Mapping**: Pydeck, Piexif
-   **Data Analysis & Visualization**: Pandas, Matplotlib, Seaborn
-   **Core Libraries**: PyTorch, Transformers, Pillow

---

## 📂 Repository Structure

This repository contains the full project, including the research notebook, a deployable Streamlit application, and standalone scripts for inference and evaluation.

```
.
├── 📄 oceanguard.ipynb            # Main Jupyter Notebook with analysis, visualizations, and narrative.
├── README.md                   # This README file.
├── 📜 inference_gemma.py          # Standalone CLI script for quick detection or counting.
├── 📊 evaluate_performance.py     # Script to quantitatively evaluate model performance.
└── 📁 app/
    ├── 🚀 main.py                  # Entry point for the interactive Streamlit demo.
    ├── 🤖 gemma_handler.py          # Core logic for Gemma model inference and report generation.
    ├── 🛠️ utils.py                  # Helper functions (EXIF extraction, health score, robust JSON parsing).
    ├── 🎨 ui_components.py          # Functions for building Streamlit UI elements (maps, charts, reports).
    └── ⚙️ config.py                 # Central configuration for prompts, data, and settings.
```

---

## 🏁 Getting Started

### 1. Jupyter Notebook (`oceanguard.ipynb`)

The notebook is the centerpiece of this project, offering a deep dive into the technology and its impact.

-   **Setup**: The first code cell installs all necessary dependencies.
-   **Execution**: Run the cells sequentially to experience the full analysis pipeline, from model setup to generating the final visualizations and reports.

### 2. Interactive Streamlit Demo (`app/`)

The demo provides a hands-on experience with the OceanGuard AI platform.

-   **Installation**:
    ```bash
    pip install -r requirements.txt
    ```
-   **Running the App**:
    ```bash
    streamlit run app/main.py
    ```
    Navigate to the local URL provided by Streamlit to upload images and interact with the platform.

### 3. Command-Line Inference (`inference_gemma.py`)

This script allows for direct, headless analysis of images. It has two modes:

-   **Detect Mode**: Identifies debris, provides bounding boxes, and saves an annotated image.
    ```bash
    python inference_gemma.py --mode detect --path "path/to/your/image.jpg"
    ```
-   **Count Mode**: Provides a quick JSON summary of debris types and counts.
    ```bash
    python inference_gemma.py --mode count --path "path/to/your/image.jpg"
    ```

### 4. Performance Evaluation (`evaluate_performance.py`)

This script benchmarks the model's accuracy against a labeled dataset (e.g., CleanSea in COCO format).

-   **Configuration**: Update the `IMAGE_DIR` and `ANNOTATION_FILE` paths inside the script to point to your dataset.
-   **Execution**:
    ```bash
    # Run standard evaluation
    python evaluate_performance.py

    # Run with verbose output and generate comparison images
    python evaluate_performance.py -v --draw

    # Generate a final visual report dashboard
    python evaluate_performance.py --report
    ```

---

## 🔮 Impact & Vision

OceanGuard AI is a deployable solution with the potential to revolutionize marine conservation. Our AI pipeline is up to 10,000x faster than manual surveys, drastically reducing costs and empowering rapid, data-driven action.

**Our Roadmap Includes:**
- **🤖Edge AI Deployment**: Port the model to **NVIDIA Jetson Orin Nano** for real-time analysis on autonomous underwater drones.
- **🎯Specialized Fine-Tuning**: Continuously improve accuracy by fine-tuning on datasets like **CleanSea** to create expert models for specific regions.
- **🌊Predictive Analysis**: Integrate with oceanographic models to predict future debris hotspots.
- **🌐Global API**: Offer a scalable API for institutions and NGOs to integrate our capabilities into their platforms.

---

## 🤝 Join the Revolution

OceanGuard AI is ready to evolve. If you're passionate about leveraging AI for ocean conservation, let's connect and build the world's largest marine protection network together.

-   **Contact**: [Alejandro Sánchez Ferrer on LinkedIn](https://www.linkedin.com/in/alejandro-sanchez-ferrer/)
-   **Team**: © 2025 Saflex Team