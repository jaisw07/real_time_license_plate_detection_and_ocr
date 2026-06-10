from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import time
import json
import zipfile
import io
import httpx
import base64
from typing import List
from src.models.schemas import (
    PipelineResponse, 
    BatchURLRequest, 
    DetectionResult, 
    BoundingBox, 
    ConfidenceScores,
    SingleBase64Request,
    BatchBase64Request,
    ZipBase64Request
)

router = APIRouter()

def decode_b64_image(b64_str: str) -> bytes:
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        return base64.b64decode(b64_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {e}")

async def process_image_bytes(pipeline, image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return None, None
    
    start_time = time.time()
    detections = pipeline.process_image(image)
    end_time = time.time()
    
    h, w = image.shape[:2]
    
    return detections, (end_time - start_time) * 1000, {"width": w, "height": h}

@router.post("/", response_model=PipelineResponse)
async def detect_single(request: Request, payload: SingleBase64Request):
    pipeline = request.app.state.pipeline
    image_bytes = decode_b64_image(payload.image)
    
    detections, proc_time, frame_info = await process_image_bytes(pipeline, image_bytes)
    
    if detections is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
        
    return PipelineResponse(
        success=True,
        processing_time_ms=proc_time,
        detections=detections,
        frame_info=frame_info
    )

@router.post("/batch/files", response_model=List[PipelineResponse])
async def detect_batch_files(request: Request, payload: BatchBase64Request):
    pipeline = request.app.state.pipeline
    results = []
    
    for b64_image in payload.images:
        try:
            image_bytes = decode_b64_image(b64_image)
            detections, proc_time, frame_info = await process_image_bytes(pipeline, image_bytes)
            
            if detections is not None:
                results.append(PipelineResponse(
                    success=True,
                    processing_time_ms=proc_time,
                    detections=detections,
                    frame_info=frame_info
                ))
            else:
                results.append(PipelineResponse(
                    success=False,
                    processing_time_ms=0,
                    detections=[]
                ))
        except Exception:
            results.append(PipelineResponse(
                success=False,
                processing_time_ms=0,
                detections=[]
            ))
            
    return results

@router.post("/batch/zip")
async def detect_batch_zip(request: Request, payload: ZipBase64Request):
    pipeline = request.app.state.pipeline
    zip_bytes = decode_b64_image(payload.zip)
    
    async def generate_results():
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                for filename in z.namelist():
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        with z.open(filename) as f:
                            image_bytes = f.read()
                            detections, proc_time, frame_info = await process_image_bytes(pipeline, image_bytes)
                            
                            result = {
                                "filename": filename,
                                "success": detections is not None,
                                "processing_time_ms": proc_time,
                                "detections": detections,
                                "frame_info": frame_info
                            }
                            yield json.dumps(result) + "\n"
        except Exception as e:
            yield json.dumps({"success": False, "error": str(e)}) + "\n"

    return StreamingResponse(generate_results(), media_type="application/x-ndjson")

@router.post("/batch/urls", response_model=List[PipelineResponse])
async def detect_batch_urls(request: Request, payload: BatchURLRequest):
    pipeline = request.app.state.pipeline
    results = []
    
    async with httpx.AsyncClient() as client:
        for url in payload.urls:
            try:
                response = await client.get(url, timeout=10.0)
                if response.status_code == 200:
                    image_bytes = response.content
                    detections, proc_time, frame_info = await process_image_bytes(pipeline, image_bytes)
                    
                    if detections is not None:
                        results.append(PipelineResponse(
                            success=True,
                            processing_time_ms=proc_time,
                            detections=detections,
                            frame_info=frame_info
                        ))
                        continue
            except Exception as e:
                pass
            
            results.append(PipelineResponse(
                success=False,
                processing_time_ms=0,
                detections=[]
            ))
            
    return results
