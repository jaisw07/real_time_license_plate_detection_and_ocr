import os
import sys

# Add project root to sys.path so 'from src' works correctly
# __file__ is src/api/main.py, so we need to go up three levels to get to project root
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import logging
from src.services.pipeline import LicensePlatePipeline
from src.api.routers import detect, stream
from src.api.utils.auth import get_api_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize pipeline
    logger.info("Initializing LicensePlatePipeline...")
    # Swapped to YOLOv26 for NMS-Free inference and better efficiency
    model_path = r"weights\yolov26n_plate.pt"
    app.state.pipeline = LicensePlatePipeline(model_path=model_path)
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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
