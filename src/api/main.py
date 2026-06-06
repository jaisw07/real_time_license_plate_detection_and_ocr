import os
import sys

# Add project root to sys.path so 'from src' works correctly
# __file__ is src/api/main.py, so we need to go up three levels to get to project root
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import time
import logging
from src.services.pipeline import LicensePlatePipeline
from src.services.history import HistoryManager
from src.api.routers import detect, stream, history
from src.api.utils.auth import get_api_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize pipeline
    logger.info("Initializing LicensePlatePipeline...")
    # Using the verified YOLOv11 model as YOLOv26 is still training
    model_path = r"runs\detect\runs\train\yolov11n_plate\weights\best.pt"
    app.state.pipeline = LicensePlatePipeline(model_path=model_path)
    
    # Initialize History Manager
    app.state.history_manager = HistoryManager()
    
    yield
    # Cleanup
    logger.info("Shutting down LicensePlatePipeline...")
    del app.state.pipeline

app = FastAPI(
    title="Real-Time License Plate Detection & OCR API",
    description="API for detecting license plates and recognizing text using YOLOv26 and RapidOCR.",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use "*" for testing, restrict to ["*.streamlit.app"] later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(detect.router, prefix="/detect", tags=["Detection"], dependencies=[Depends(get_api_key)])
app.include_router(stream.router, prefix="/stream", tags=["Streaming"]) # Stream handles auth internally due to WebSockets
app.include_router(history.router, prefix="/history", tags=["History"])

# Mount static files for history images
app.mount("/history/images", StaticFiles(directory="data/history/images"), name="history_images")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
