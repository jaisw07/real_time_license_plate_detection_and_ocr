# Real-Time License Plate Detection & OCR

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136.3-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B.svg)](https://streamlit.io/)
[![YOLOv26](https://img.shields.io/badge/YOLO-v26n-green.svg)](https://github.com/ultralytics/ultralytics)

This is a high-performance, real-time license plate detection and OCR system. It utilizes a **Hybrid Architecture** that leverages a local GPU (RTX 4050) for heavy inference while providing a seamless, high-speed cloud frontend via Streamlit Community Cloud and Cloudflare Tunnels.

---

## 🌟 Key Features

*   **⚡ Real-Time Tracking:** Integration of **YOLOv26n** and **ByteTrack** for stable, multi-object tracking.
*   **🎨 Plate Colour Extraction:** Extracts the dominant plate background color using a complete HSV space partition on a center-inset crop (middle 70% height, 80% width) and maps it to **Red, White, Blue, Green, Yellow, or Black**.
*   **🔍 High-Accuracy OCR:** Powered by **RapidOCR (ONNX)** with custom preprocessing (CLAHE, Grayscale) and padding optimization.
*   **🎥 Live AR Dashboard:** A custom HTML5/Canvas component for Streamlit that enables direct browser camera access with client-side EMA smoothing.
*   **🛡️ Secure Hybrid Flow:** Local FastAPI backend exposed via **Cloudflare Tunnels** with `X-API-Key` authentication.
*   **📦 Base64 JSON API Flow:** Optimized HTTP endpoints accepting base64-encoded JSON requests. Includes a dedicated Dashboard tab for direct Base64 testing. Supports single images, batch lists, URLs, and ZIP archives.

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Object Detection** | YOLOv26n (NMS-Free Architecture) |
| **Object Tracking** | ByteTrack |
| **Colour Extraction**| HSV Partition Classifier (Red, White, Blue, Green, Yellow, Black) |
| **OCR Engine** | RapidOCR (ONNX Runtime) |
| **Backend API** | FastAPI (Async/Lifespan Management) |
| **Frontend** | Streamlit + Custom HTML/JS Canvas AR |
| **Tunneling** | Cloudflare Tunnels (Argo) |
| **Environment** | Conda + Pip |

---

## 📁 Project Structure

> **Note:** Only the `src` directory and essential configuration files are committed to this repository. Weights, datasets, and logs are ignored.

```text
license-plate-ocr/
├── src/                    # Main Source Code
│   ├── api/                # FastAPI Application & Routers
│   ├── dashboard/          # Streamlit Frontend Dashboard
│   ├── models/             # Pydantic Schemas & Models
│   ├── services/           # Pipeline, OCR, and Preprocessing Logic
│   └── utils/              # Post-processing, Color Extraction, and Auth Utilities
├── context/                # Documentation & Context Files
├── .env.example            # Template for environment variables
├── environment.backend.yml # Conda environment definition
├── requirements.txt        # Pip requirements (frozen list)
└── config.yml              # Cloudflare Tunnel configuration
```

---

## 🚀 Getting Started

### 1. Prerequisites
*   Python 3.10+
*   Conda (Recommended)
*   NVIDIA GPU (RTX 4050 or similar for optimal performance)

### 2. Installation

Clone the repository and set up the environment:

```bash
# Create and activate conda environment
conda env create -f environment.backend.yml
conda activate ttl

# Alternatively, using pip
pip install -r requirements.txt
```

### 3. Configuration
1.  Copy `.env.example` to `.env`.
2.  Set your `API_KEY` in the `.env` file.
3.  Ensure your `config.yml` is correctly configured with your Cloudflare Tunnel credentials.

### 4. Running the System

To run the system in a hybrid mode (Local Backend + Cloud/Tunnel Access):

**Step A: Start the FastAPI Backend**
```bash
python -m src.api.main
```

**Step B: Start the Cloudflare Tunnel**
```bash
# Using the pre-configured config.yml
.\cloudflared.exe tunnel run ttl-backend
```

**Step C: Access the Dashboard**
Access your dashboard via your Streamlit Community Cloud URL or run it locally:
```bash
streamlit run src/dashboard/app.py
```

---

## 📊 Performance & Metrics

*   **Detector (YOLOv11n Baseline):** mAP@50 = 0.9890, mAP@50-95 = 0.7158.
*   **Detector (YOLOv26n):** Equivalent accuracy with 17% reduction in GFLOPs and NMS-Free architecture for stable latency.
*   **OCR:** RapidOCR optimized with 15% padding, contextual correction, regex validation (Indian formats), and IoU-based fragment deduplication/merging.
*   **Plate Colour:** Extracted using HSV space pixel voting on a center inset crop (to filter out vehicle paint/borders).
*   **Latency:** Frames resized to **640px** browser-side to ensure stable ~15+ FPS over tunnels.

---

## 📝 Usage Guidelines

*   **API Docs:** Once the backend is running, visit `http://localhost:8000/docs` for interactive API documentation.
*   **Base64 HTTP APIs:** 
    *   `POST /detect/` accepts `{"image": "<base64_string>"}`
    *   `POST /detect/batch/files` accepts `{"images": ["<base64_string_1>", ...]}`
    *   `POST /detect/batch/zip` accepts `{"zip": "<base64_zip_string>"}`
*   **Live AR:** Prioritizes the environment (back) camera on mobile devices.
*   **Batch ZIP:** Returns a streaming NDJSON response for real-time progress updates on large datasets.

---

## ⚖️ License

[MIT License](LICENSE) - See the LICENSE file for details.
