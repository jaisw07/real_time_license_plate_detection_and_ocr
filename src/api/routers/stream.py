from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Query, status
import cv2
import numpy as np
import time
import json
import logging
from ultralytics import YOLO
import asyncio
from src.api.utils.auth import get_api_key

logger = logging.getLogger(__name__)

router = APIRouter()

@router.websocket("/")
async def websocket_stream(
    websocket: WebSocket,
    token: str = Query(None)
):
    # 1. Manual Auth Check (WebSockets require query param auth)
    try:
        await get_api_key(query_key=token)
    except Exception:
        await websocket.accept()
        await websocket.send_json({"success": False, "error": "Unauthorized"})
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    pipeline = websocket.app.state.pipeline
    
    # Per-connection YOLO instance for isolated tracking
    model_path = pipeline.model_path
    session_model = YOLO(model_path)
    
    # Per-connection cache for OCR results
    ocr_cache = {} # track_id -> {"plate_number": str, "raw_ocr": str, "ocr_conf": float, "is_processing": bool}
    ocr_semaphore = asyncio.Semaphore(2)

    async def run_async_ocr(track_id, frame, box):
        async with ocr_semaphore:
            try:
                loop = asyncio.get_event_loop()
                cleaned, raw, conf, color = await loop.run_in_executor(
                    None, pipeline.run_ocr_on_crop, frame, box
                )
                ocr_cache[track_id] = {
                    "plate_number": cleaned,
                    "raw_ocr": raw,
                    "ocr_conf": conf,
                    "plate_colour": color,
                    "is_processing": False
                }
            except Exception as e:
                logger.error(f"Async OCR error for track {track_id}: {e}")
                if track_id in ocr_cache:
                    del ocr_cache[track_id]

    try:
        while True:
            # 2. Heartbeat/Timeout Logic (Wait 30s for a frame)
            try:
                # Receive frame as bytes with timeout
                frame_bytes = await asyncio.wait_for(websocket.receive_bytes(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send a ping/heartbeat message to keep tunnel alive
                await websocket.send_json({"type": "heartbeat", "timestamp": time.time()})
                continue

            start_time = time.time()
            
            # Decode image
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            h, w = frame.shape[:2]
            
            # Track vehicles
            tracked_detections = pipeline.track_and_process(frame, model=session_model)
            
            final_detections = []
            for det in tracked_detections:
                track_id = det["track_id"]
                
                if track_id in ocr_cache:
                    ocr_result = ocr_cache[track_id]
                else:
                    ocr_cache[track_id] = {
                        "plate_number": "Processing...", 
                        "raw_ocr": "", 
                        "ocr_conf": 0.0, 
                        "plate_colour": "Unknown",
                        "is_processing": True
                    }
                    asyncio.create_task(run_async_ocr(track_id, frame.copy(), det["box"]))
                    ocr_result = ocr_cache[track_id]
                
                final_detections.append({
                    "plate_number": ocr_result["plate_number"],
                    "raw_ocr": ocr_result["raw_ocr"],
                    "confidence": {
                        "detection": det["confidence"],
                        "ocr": ocr_result["ocr_conf"]
                    },
                    "bounding_box": det["bounding_box"],
                    "track_id": track_id,
                    "plate_colour": ocr_result.get("plate_colour", "Unknown"),
                    "is_ocr_pending": ocr_result.get("is_processing", False)
                })
            
            end_time = time.time()
            proc_time = (end_time - start_time) * 1000
            
            await websocket.send_json({
                "success": True,
                "processing_time_ms": proc_time,
                "detections": final_detections,
                "frame_info": {"width": w, "height": h}
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        del session_model
        logger.info("Cleaned up session model")
