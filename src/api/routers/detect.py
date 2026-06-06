from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import time
import json
import zipfile
import io
import httpx
import asyncio
from typing import List
from src.models.schemas import PipelineResponse, BatchURLRequest, DetectionResult, BoundingBox, ConfidenceScores

router = APIRouter()

async def process_image_bytes(pipeline, image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return None, None, None, None
    
    start_time = time.time()
    detections = pipeline.process_image(image)
    end_time = time.time()
    
    h, w = image.shape[:2]
    
    return image, detections, (end_time - start_time) * 1000, {"width": w, "height": h}

@router.post("/", response_model=PipelineResponse)
async def detect_single(request: Request, file: UploadFile = File(...)):
    pipeline = request.app.state.pipeline
    history_manager = request.app.state.history_manager
    image_bytes = await file.read()
    
    frame, detections, proc_time, frame_info = await process_image_bytes(pipeline, image_bytes)
    
    if detections is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Save detections to history in background
    for det in detections:
        asyncio.create_task(history_manager.add_entry(
            frame=frame,
            plate_number=det["plate_number"],
            confidence=det["confidence"],
            bounding_box=det["bounding_box"]
        ))
        
    return PipelineResponse(
        success=True,
        processing_time_ms=proc_time,
        detections=detections,
        frame_info=frame_info
    )

@router.post("/batch/files", response_model=List[PipelineResponse])
async def detect_batch_files(request: Request, files: List[UploadFile] = File(...)):
    pipeline = request.app.state.pipeline
    history_manager = request.app.state.history_manager
    results = []
    
    for file in files:
        image_bytes = await file.read()
        frame, detections, proc_time, frame_info = await process_image_bytes(pipeline, image_bytes)
        
        if detections is not None:
            results.append(PipelineResponse(
                success=True,
                processing_time_ms=proc_time,
                detections=detections,
                frame_info=frame_info
            ))
            # Save to history
            for det in detections:
                asyncio.create_task(history_manager.add_entry(
                    frame=frame,
                    plate_number=det["plate_number"],
                    confidence=det["confidence"],
                    bounding_box=det["bounding_box"]
                ))
        else:
            results.append(PipelineResponse(
                success=False,
                processing_time_ms=0,
                detections=[]
            ))
            
    return results

@router.post("/batch/zip")
async def detect_batch_zip(request: Request, file: UploadFile = File(...)):
    pipeline = request.app.state.pipeline
    history_manager = request.app.state.history_manager
    
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are supported")
        
    zip_bytes = await file.read()
    
    async def generate_results():
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for filename in z.namelist():
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    with z.open(filename) as f:
                        image_bytes = f.read()
                        frame, detections, proc_time, frame_info = await process_image_bytes(pipeline, image_bytes)
                        
                        if detections is not None:
                            for det in detections:
                                asyncio.create_task(history_manager.add_entry(
                                    frame=frame,
                                    plate_number=det["plate_number"],
                                    confidence=det["confidence"],
                                    bounding_box=det["bounding_box"]
                                ))

                        result = {
                            "filename": filename,
                            "success": detections is not None,
                            "processing_time_ms": proc_time,
                            "detections": detections,
                            "frame_info": frame_info
                        }
                        yield json.dumps(result) + "\n"

    return StreamingResponse(generate_results(), media_type="application/x-ndjson")

@router.post("/batch/urls", response_model=List[PipelineResponse])
async def detect_batch_urls(request: Request, payload: BatchURLRequest):
    pipeline = request.app.state.pipeline
    history_manager = request.app.state.history_manager
    results = []
    
    async with httpx.AsyncClient() as client:
        for url in payload.urls:
            try:
                response = await client.get(url, timeout=10.0)
                if response.status_code == 200:
                    image_bytes = response.content
                    frame, detections, proc_time, frame_info = await process_image_bytes(pipeline, image_bytes)
                    
                    if detections is not None:
                        results.append(PipelineResponse(
                            success=True,
                            processing_time_ms=proc_time,
                            detections=detections,
                            frame_info=frame_info
                        ))
                        for det in detections:
                            asyncio.create_task(history_manager.add_entry(
                                frame=frame,
                                plate_number=det["plate_number"],
                                confidence=det["confidence"],
                                bounding_box=det["bounding_box"]
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
