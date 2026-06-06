import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io
import os
import time
import streamlit_shadcn_ui as ui

st.set_page_config(page_title="License Plate Detection & OCR", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .plate-card {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 10px;
        margin-bottom: 20px;
        text-align: center;
        background-color: #f9f9f9;
    }
    .plate-number {
        font-size: 24px;
        font-weight: bold;
        color: #ff4b4b;
        margin-top: 10px;
    }
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">🚗 License Plate Intelligence</div>', unsafe_allow_html=True)

# Configuration (No Sidebar)
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_KEY = "your_placeholder_secret_key_here"
api_base_url = os.getenv("BACKEND_URL", DEFAULT_API_URL)
api_key = os.getenv("API_KEY", DEFAULT_KEY)
headers = {"X-API-Key": api_key}
CONF_THRESHOLD = 0.5

tabs = st.tabs(["📷 Processing", "🎥 Live AR Feed", "📦 Batch ZIP"])

def process_single_image(image_bytes, image_name="image.jpg"):
    try:
        files = {"file": (image_name, image_bytes, "image/jpeg")}
        response = requests.post(f"{api_base_url}/detect/", files=files, headers=headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            st.error("Authentication Failed: Check API Key.")
            return None
        else:
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None

def fetch_history(limit=10):
    try:
        response = requests.get(f"{api_base_url}/history/?limit={limit}", headers=headers)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

# --- MAIN UI ---

with tabs[0]:
    input_type = st.radio("Select Input Source", ["Upload Images", "Image URL"], horizontal=True)
    
    results_to_show = []

    if input_type == "Upload Images":
        uploaded_files = st.file_uploader("Drop images here", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            with st.spinner("Analyzing..."):
                for uploaded_file in uploaded_files:
                    img_bytes = uploaded_file.read()
                    data = process_single_image(img_bytes, uploaded_file.name)
                    if data:
                        results_to_show.append({"name": uploaded_file.name, "image": Image.open(io.BytesIO(img_bytes)), "data": data})
    
    else:
        url = st.text_input("Paste Image URL")
        if url:
            with st.spinner("Fetching..."):
                try:
                    resp = requests.get(url)
                    img_bytes = resp.content
                    data = process_single_image(img_bytes, "url_image.jpg")
                    if data:
                        results_to_show.append({"name": "URL Image", "image": Image.open(io.BytesIO(img_bytes)), "data": data})
                except:
                    st.error("Could not load image from URL.")

    # Display Current Results
    for res in results_to_show:
        detections = res['data'].get("detections", [])
        if res['image']:
            img_np = np.array(res['image'])
            for det in detections:
                if det["confidence"]["detection"] >= CONF_THRESHOLD:
                    box = det["bounding_box"]
                    plate = det["plate_number"]
                    cv2.rectangle(img_np, (int(box["x1"]), int(box["y1"])), (int(box["x2"]), int(box["y2"])), (0, 255, 0), 4)
                    cv2.putText(img_np, plate, (int(box["x1"]), int(box["y1"]) - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            st.image(img_np, use_container_width=True)
            
            # Simple list of plates
            if detections:
                plates = [d["plate_number"] for d in detections if d["confidence"]["detection"] >= CONF_THRESHOLD]
                if plates:
                    st.write("### Detected Plates")
                    cols = st.columns(len(plates))
                    for i, plate in enumerate(plates):
                        with cols[i]:
                            ui.badge(text=plate, variant="outline", key=f"plate_{i}_{time.time()}")

with tabs[1]:
    st.info("Ensure you are using HTTPS for camera access. Mobile browsers recommended.")
    
    ws_url = api_base_url.replace("http://", "ws://").replace("https://", "wss://") + "/stream/"
    ws_url_with_auth = f"{ws_url}?token={api_key}"
    
    # Use the existing AR HTML logic
    ar_component_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <style>
            body {{ margin: 0; font-family: sans-serif; background: #0e1117; color: white; overflow: hidden; }}
            #container {{ position: relative; width: 100%; max-width: 800px; margin: auto; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 30px rgba(0,0,0,0.7); }}
            #video {{ width: 100%; height: auto; display: block; }}
            #overlay {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
            .controls {{ padding: 15px; text-align: center; }}
            button {{ 
                background: linear-gradient(135deg, #ff4b4b, #ff7e7e); color: white; border: none; padding: 14px 30px; 
                border-radius: 10px; cursor: pointer; font-weight: bold; font-size: 18px;
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            button:active {{ transform: scale(0.95); }}
            button:disabled {{ background: #444; cursor: not-allowed; }}
            #status {{ font-size: 14px; color: #888; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <div id="container">
            <video id="video" autoplay playsinline muted></video>
            <canvas id="overlay"></canvas>
        </div>
        <div class="controls">
            <button id="startBtn">Launch AR Mode</button>
            <div id="status">Ready</div>
        </div>

        <script>
            const video = document.getElementById('video');
            const overlay = document.getElementById('overlay');
            const ctx = overlay.getContext('2d');
            const startBtn = document.getElementById('startBtn');
            const status = document.getElementById('status');

            let ws;
            let isProcessing = false;
            let smoothedDetections = {{}}; 

            async function startCamera() {{
                try {{
                    const stream = await navigator.mediaDevices.getUserMedia({{
                        video: {{ facingMode: 'environment', width: {{ ideal: 1280 }}, height: {{ ideal: 720 }} }}
                    }});
                    video.srcObject = stream;
                    video.onloadedmetadata = () => {{
                        overlay.width = video.videoWidth;
                        overlay.height = video.videoHeight;
                        startBtn.disabled = true;
                        connectWebSocket();
                    }};
                }} catch (err) {{
                    status.innerText = "Error: " + err.message;
                }}
            }}

            function connectWebSocket() {{
                ws = new WebSocket("{ws_url_with_auth}");
                ws.binaryType = "blob";
                ws.onopen = () => {{ status.innerText = "Connected - Analyzing Stream"; processLoop(); renderLoop(); }};
                ws.onmessage = (event) => {{
                    const data = JSON.parse(event.data);
                    const now = Date.now();
                    if (data.detections) {{
                        data.detections.forEach(d => {{
                            if (!smoothedDetections[d.track_id]) {{
                                smoothedDetections[d.track_id] = {{ ...d.bounding_box, plate: d.plate_number, pending: d.is_ocr_pending, lastUpdate: now, target: d.bounding_box }};
                            }} else {{
                                smoothedDetections[d.track_id].target = d.bounding_box;
                                smoothedDetections[d.track_id].plate = d.plate_number;
                                smoothedDetections[d.track_id].pending = d.is_ocr_pending;
                                smoothedDetections[d.track_id].lastUpdate = now;
                            }}
                        }});
                    }}
                    isProcessing = false;
                }};
            }}

            async function processLoop() {{
                if (ws && ws.readyState === WebSocket.OPEN && !isProcessing) {{
                    isProcessing = true;
                    const offscreen = document.createElement('canvas');
                    offscreen.width = 640; offscreen.height = 360;
                    const oCtx = offscreen.getContext('2d');
                    oCtx.drawImage(video, 0, 0, 640, 360);
                    offscreen.toBlob((blob) => {{ ws.send(blob); }}, 'image/jpeg', 0.5);
                }}
                setTimeout(processLoop, 50);
            }}

            function renderLoop() {{
                const now = Date.now();
                ctx.clearRect(0, 0, overlay.width, overlay.height);
                const scaleX = overlay.width / 640;
                const scaleY = overlay.height / 360;

                for (const tid in smoothedDetections) {{
                    const d = smoothedDetections[tid];
                    if (now - d.lastUpdate > 1000) {{ delete smoothedDetections[tid]; continue; }}
                    
                    const alpha = 0.3;
                    d.x1 = d.x1 * (1-alpha) + d.target.x1 * alpha;
                    d.y1 = d.y1 * (1-alpha) + d.target.y1 * alpha;
                    d.x2 = d.x2 * (1-alpha) + d.target.x2 * alpha;
                    d.y2 = d.y2 * (1-alpha) + d.target.y2 * alpha;

                    ctx.strokeStyle = '#00FF00'; ctx.lineWidth = 6;
                    ctx.strokeRect(d.x1 * scaleX, d.y1 * scaleY, (d.x2 - d.x1) * scaleX, (d.y2 - d.y1) * scaleY);
                    
                    ctx.fillStyle = '#00FF00'; ctx.font = 'bold 32px sans-serif';
                    ctx.fillText(d.pending ? "..." : d.plate, d.x1 * scaleX, d.y1 * scaleY - 10);
                }
                requestAnimationFrame(renderLoop);
            }}

            startBtn.onclick = startCamera;
        </script>
    </body>
    </html>
    """
    import streamlit.components.v1 as components
    components.html(ar_component_html, height=600)

with tabs[2]:
    st.markdown("### High-Speed Batch Upload")
    zip_file = st.file_uploader("Upload ZIP archive", type=["zip"])
    if zip_file and st.button("Start Batch Processing"):
        # Existing logic for ZIP...
        pass

# --- GLOBAL HISTORY SECTION ---
st.markdown("---")
st.markdown("## 🌐 Global Detection History")

history_data = fetch_history(limit=20)

if history_data:
    # Display in a grid
    cols = st.columns(4)
    for i, entry in enumerate(history_data):
        with cols[i % 4]:
            img_url = f"{api_base_url}{entry['image_path']}"
            plate = entry['plate_number']
            
            # Custom HTML card for cleaner display
            st.markdown(f"""
                <div class="plate-card">
                    <img src="{img_url}" style="width:100%; border-radius:5px;">
                    <div class="plate-number">{plate}</div>
                    <div style="font-size: 10px; color: gray;">{time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))}</div>
                </div>
                """, unsafe_allow_html=True)
else:
    st.info("No detections recorded yet.")

# Auto-refresh button (Streamlit way)
if st.button("🔄 Refresh History"):
    st.rerun()
