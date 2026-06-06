import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io
import os
import pandas as pd
import json

st.set_page_config(page_title="License Plate Detection & OCR", layout="wide")

st.title("🚗 Real-Time License Plate Detection & OCR")
st.markdown("---")

# Load Configuration from Secrets or Environment
# Priority: Streamlit Secrets -> Environment Variables -> Default Local
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_KEY = "your_placeholder_secret_key_here"

api_base_url = st.secrets.get("BACKEND_URL", os.getenv("BACKEND_URL", DEFAULT_API_URL))
api_key = st.secrets.get("API_KEY", os.getenv("API_KEY", DEFAULT_KEY))

# Sidebar for manual override (useful for local testing)
st.sidebar.title("Settings")
api_base_url = st.sidebar.text_input("API Base URL", value=api_base_url)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# Standard Headers for API Requests
headers = {"X-API-Key": api_key}

tabs = st.tabs(["🖼️ Image Processing", "📦 Batch Processing (ZIP)", "🎥 Live Camera"])

def process_single_image(image_bytes, image_name="image.jpg"):
    try:
        files = {"file": (image_name, image_bytes, "image/jpeg")}
        response = requests.post(f"{api_base_url}/detect/", files=files, headers=headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            st.error("Authentication Failed: Check your API Key.")
            return None
        else:
            st.error(f"API Error ({image_name}): {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Request failed ({image_name}): {e}")
        return None

def process_url(url):
    try:
        # The API expects POST /detect/batch/urls with a JSON body: {"urls": [url]}
        response = requests.post(f"{api_base_url}/detect/batch/urls", json={"urls": [url]}, headers=headers)
        if response.status_code == 200:
            results = response.json()
            return results[0] if results else None
        else:
            st.error(f"API Error (URL): {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Request failed (URL): {e}")
        return None

with tabs[0]:
    st.header("Image Processing")
    
    input_type = st.radio("Input Method", ["Upload Images", "Image URL"], horizontal=True)
    
    results_to_show = []

    if input_type == "Upload Images":
        uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            with st.spinner(f"Processing {len(uploaded_files)} images..."):
                for uploaded_file in uploaded_files:
                    img_bytes = uploaded_file.read()
                    data = process_single_image(img_bytes, uploaded_file.name)
                    if data:
                        results_to_show.append({"name": uploaded_file.name, "image": Image.open(io.BytesIO(img_bytes)), "data": data})
    
    else:
        url = st.text_input("Enter Image URL")
        if url:
            with st.spinner("Processing URL..."):
                data = process_url(url)
                if data:
                    # Try to download image for display
                    try:
                        resp = requests.get(url)
                        img = Image.open(io.BytesIO(resp.content))
                        results_to_show.append({"name": "URL Image", "image": img, "data": data})
                    except:
                        st.warning("Could not download image for preview, but API results are below.")
                        results_to_show.append({"name": "URL Image", "image": None, "data": data})

    # Display Results
    for res in results_to_show:
        with st.expander(f"Results for {res['name']}", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            detections = res['data'].get("detections", [])
            
            with col1:
                if res['image']:
                    img_np = np.array(res['image'])
                    for det in detections:
                        box = det["bounding_box"]
                        plate = det["plate_number"]
                        conf = det["confidence"]["detection"]
                        
                        cv2.rectangle(img_np, (int(box["x1"]), int(box["y1"])), (int(box["x2"]), int(box["y2"])), (0, 255, 0), 2)
                        cv2.putText(img_np, f"{plate} ({conf:.2f})", (int(box["x1"]), int(box["y1"]) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    st.image(img_np, use_container_width=True)
                else:
                    st.info("No image preview available.")
            
            with col2:
                if detections:
                    st.subheader("Detections")
                    df = pd.DataFrame([
                        {
                            "Plate": d["plate_number"],
                            "Conf (Det)": f"{d['confidence']['detection']:.2f}",
                            "Conf (OCR)": f"{d['confidence']['ocr']:.2f}"
                        } for d in detections
                    ])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No license plates detected.")

with tabs[1]:
    st.header("Batch ZIP Processing")
    st.markdown("Upload a ZIP file containing multiple images for high-speed batch processing.")
    
    zip_file = st.file_uploader("Upload ZIP file", type=["zip"])
    
    if zip_file:
        if st.button("Start Batch Processing"):
            with st.spinner("Processing ZIP archive..."):
                try:
                    files = {"file": ("batch.zip", zip_file.getvalue(), "application/zip")}
                    # Note: /detect/batch/zip returns a streaming response (NDJSON)
                    response = requests.post(f"{api_base_url}/detect/batch/zip", files=files, stream=True)
                    
                    if response.status_code == 200:
                        all_results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Read NDJSON stream
                        for line in response.iter_lines():
                            if line:
                                result = json.loads(line)
                                all_results.append(result)
                                status_text.text(f"Processed: {result.get('filename', 'Unknown')}")
                        
                        st.success(f"Successfully processed {len(all_results)} items.")
                        
                        # Display summary table
                        summary_data = []
                        for res in all_results:
                            filename = res.get("filename")
                            detections = res.get("detections", [])
                            plates = ", ".join([d["plate_number"] for d in detections]) if detections else "None"
                            summary_data.append({"File": filename, "Detected Plates": plates})
                        
                        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                        
                    else:
                        st.error(f"Batch API Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Batch processing failed: {e}")

with tabs[2]:
    st.header("Live AR Camera Feed")
    st.info("The live camera feed uses WebSockets for real-time tracking and OCR.")
    
    # Construct WebSocket URL with Token for Auth
    ws_url = api_base_url.replace("http://", "ws://").replace("https://", "wss://") + "/stream/"
    ws_url_with_auth = f"{ws_url}?token={api_key}"
    
    # Custom HTML/JS Component for AR Camera
    ar_component_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <style>
            body {{ margin: 0; font-family: sans-serif; background: #0e1117; color: white; overflow: hidden; }}
            #container {{ position: relative; width: 100%; max-width: 640px; margin: auto; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }}
            #video {{ width: 100%; height: auto; display: block; }}
            #overlay {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
            .controls {{ padding: 20px; text-align: center; display: flex; flex-direction: column; gap: 10px; align-items: center; }}
            button {{ 
                background: #ff4b4b; color: white; border: none; padding: 12px 24px; 
                border-radius: 8px; cursor: pointer; font-weight: bold; font-size: 16px;
                transition: background 0.3s;
            }}
            button:hover {{ background: #ff3333; }}
            button:disabled {{ background: #333; cursor: not-allowed; }}
            #status {{ font-size: 14px; color: #aaa; }}
            .stats {{ position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.6); padding: 5px 10px; border-radius: 4px; font-family: monospace; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div id="container">
            <video id="video" autoplay playsinline muted></video>
            <canvas id="overlay"></canvas>
            <div id="stats" class="stats">FPS: -- | Latency: --ms</div>
        </div>
        <div class="controls">
            <button id="startBtn">Start AR Camera</button>
            <div id="status">Ready - Environment Camera Prioritized</div>
        </div>

        <script>
            const video = document.getElementById('video');
            const overlay = document.getElementById('overlay');
            const ctx = overlay.getContext('2d');
            const startBtn = document.getElementById('startBtn');
            const status = document.getElementById('status');
            const stats = document.getElementById('stats');

            let ws;
            let isProcessing = false;
            let smoothedDetections = {{}}; // track_id -> {{x1, y1, x2, y2, plate, pending, lastUpdate}}
            let frameCount = 0;
            let lastFpsTime = Date.now();
            let latency = 0;

            async function startCamera() {{
                try {{
                    status.innerText = "Requesting Camera...";
                    const stream = await navigator.mediaDevices.getUserMedia({{
                        video: {{ 
                            facingMode: 'environment', 
                            width: {{ ideal: 1280 }}, 
                            height: {{ ideal: 720 }} 
                        }}
                    }});
                    video.srcObject = stream;
                    video.onloadedmetadata = () => {{
                        overlay.width = video.videoWidth;
                        overlay.height = video.videoHeight;
                        startBtn.disabled = true;
                        status.innerText = "Connecting to API...";
                        connectWebSocket();
                    }};
                }} catch (err) {{
                    console.error("Camera error:", err);
                    status.innerText = "Camera Error: " + err.message;
                }}
            }}

            function connectWebSocket() {{
                ws = new WebSocket("{ws_url_with_auth}");
                ws.binaryType = "blob";

                ws.onopen = () => {{
                    status.innerText = "Connected - Tracking Active";
                    processLoop();
                    renderLoop();
                }};

                ws.onmessage = (event) => {{
                    const data = JSON.parse(event.data);
                    
                    // Handle Heartbeat
                    if (data.type === "heartbeat") {{
                        console.log("Heartbeat received");
                        return;
                    }}

                    const now = Date.now();
                    
                    if (data.processing_time_ms) {{
                        latency = Math.round(data.processing_time_ms);
                    }}

                    // Update smoothed detections
                    if (data.detections) {{
                        data.detections.forEach(d => {{
                            if (!smoothedDetections[d.track_id]) {{
                                smoothedDetections[d.track_id] = {{ 
                                    ...d.bounding_box, 
                                    plate: d.plate_number, 
                                    pending: d.is_ocr_pending,
                                    lastUpdate: now,
                                    target: {{ ...d.bounding_box, plate: d.plate_number, pending: d.is_ocr_pending }}
                                }};
                            }} else {{
                                smoothedDetections[d.track_id].target = {{ ...d.bounding_box, plate: d.plate_number, pending: d.is_ocr_pending }};
                                smoothedDetections[d.track_id].lastUpdate = now;
                            }}
                        }});
                    }}
                    
                    isProcessing = false;
                }};

                ws.onclose = (event) => {{
                    if (event.code === 3008) {{
                        status.innerText = "Auth Failed - Check API Key";
                    }} else {{
                        status.innerText = "Disconnected - Refresh to Reconnect";
                    }}
                    isProcessing = false;
                }};

                ws.onerror = (err) => {{
                    console.error("WebSocket error:", err);
                    status.innerText = "Connection Error";
                }};
            }}

            async function processLoop() {{
                if (ws && ws.readyState === WebSocket.OPEN && !isProcessing) {{
                    isProcessing = true;
                    
                    // Capture and resize frame to 640px width
                    const targetWidth = 640;
                    const scale = targetWidth / video.videoWidth;
                    const targetHeight = video.videoHeight * scale;
                    
                    const offscreen = document.createElement('canvas');
                    offscreen.width = targetWidth;
                    offscreen.height = targetHeight;
                    const oCtx = offscreen.getContext('2d');
                    oCtx.drawImage(video, 0, 0, targetWidth, targetHeight);
                    
                    offscreen.toBlob((blob) => {{
                        if (ws.readyState === WebSocket.OPEN) {{
                            ws.send(blob);
                        }} else {{
                            isProcessing = false;
                        }}
                    }}, 'image/jpeg', 0.4); // Optimized quality (0.4) for Tunnel transfer
                }}
                setTimeout(processLoop, 33);
            }}

            function renderLoop() {{
                const now = Date.now();
                ctx.clearRect(0, 0, overlay.width, overlay.height);
                
                frameCount++;
                if (now - lastFpsTime > 1000) {{
                    const fps = Math.round((frameCount * 1000) / (now - lastFpsTime));
                    stats.innerText = `FPS: ${{fps}} | Latency: ${{latency}}ms`;
                    frameCount = 0;
                    lastFpsTime = now;
                }}

                const targetWidth = 640;
                const scale = overlay.width / targetWidth;

                for (const tid in smoothedDetections) {{
                    const d = smoothedDetections[tid];
                    
                    const alpha = 0.25; 
                    d.x1 = d.x1 * (1 - alpha) + d.target.x1 * alpha;
                    d.y1 = d.y1 * (1 - alpha) + d.target.y1 * alpha;
                    d.x2 = d.x2 * (1 - alpha) + d.target.x2 * alpha;
                    d.y2 = d.y2 * (1 - alpha) + d.target.y2 * alpha;
                    d.plate = d.target.plate;
                    d.pending = d.target.pending;

                    if (now - d.lastUpdate > 1000) {{
                        delete smoothedDetections[tid];
                        continue;
                    }}

                    ctx.strokeStyle = '#00FF00';
                    ctx.lineWidth = 4;
                    ctx.beginPath();
                    ctx.roundRect(d.x1 * scale, d.y1 * scale, (d.x2 - d.x1) * scale, (d.y2 - d.y1) * scale, 5);
                    ctx.stroke();

                    ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
                    const label = d.pending ? "OCR Processing..." : d.plate;
                    const labelWidth = ctx.measureText(label).width + 20;
                    ctx.fillRect(d.x1 * scale, d.y1 * scale - 35, labelWidth, 30);

                    ctx.fillStyle = 'black';
                    ctx.font = 'bold 18px sans-serif';
                    ctx.fillText(label, d.x1 * scale + 10, d.y1 * scale - 12);
                }}
                
                requestAnimationFrame(renderLoop);
            }}

            startBtn.onclick = startCamera;
        </script>
    </body>
    </html>
    """
    
    import streamlit.components.v1 as components
    components.html(ar_component_html, height=600)

