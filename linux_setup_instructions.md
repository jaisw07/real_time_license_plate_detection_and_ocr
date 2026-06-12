# Linux Server Deployment Guide (Backend + Frontend)

This guide provides step-by-by instructions for deploying both the FastAPI backend and Streamlit frontend on a single Linux machine (e.g., an Ubuntu EC2 instance, DigitalOcean Droplet, etc.).

## 1. Prerequisites

- A Linux server (Ubuntu 22.04+ recommended)
- Python 3.10+ installed
- `git` installed
- (Optional but recommended) NVIDIA GPU with CUDA drivers installed

## 2. System Dependencies

Since we are running OpenCV on a server without a GUI, we use `opencv-python-headless`. Ensure your system has the required base packages:

```bash
sudo apt update
sudo apt install -y build-essential libgl1-mesa-glx libglib2.0-0 tmux
```

## 3. Clone and Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd ttl

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*(Note: The `requirements.txt` file is already optimized for Linux and Streamlit Cloud, utilizing headless OpenCV.)*

## 4. Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` to configure your `API_KEY` and any other variables:
   ```bash
   nano .env
   ```

## 5. Running the Application

For a robust deployment, you can use `systemd` or run them in detached `tmux` sessions. Here is the `tmux` approach:

### Start the FastAPI Backend

```bash
tmux new -s backend
source venv/bin/activate
python -m src.api.main
# Detach with: Ctrl+B, then D
```

### Start the Streamlit Frontend

```bash
tmux new -s frontend
source venv/bin/activate
streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
# Detach with: Ctrl+B, then D
```

## 6. Accessing the Application

- **Frontend Dashboard:** `http://<YOUR_SERVER_IP>:8501`
- **Backend API Docs:** `http://<YOUR_SERVER_IP>:8000/docs`

Ensure your cloud provider's firewall (e.g., AWS Security Groups, ufw) allows inbound traffic on ports `8000` and `8501`.

## 7. Troubleshooting

- **YOLO Model Not Found:** Ensure `weights/yolov26n_plate.pt` exists in the repository root.
- **Out of Memory (OOM):** The OCR and tracking pipeline can consume around 3GB of RAM/VRAM. Ensure your server has at least 4-8GB of memory.
- **Missing Dependencies:** If RapidOCR or OpenCV throw errors, verify `libgl1-mesa-glx` is installed.
