from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class ConfidenceScores(BaseModel):
    detection: float
    ocr: float

class DetectionResult(BaseModel):
    plate_number: str
    raw_ocr: str
    confidence: ConfidenceScores
    bounding_box: BoundingBox
    track_id: Optional[int] = None

class FrameInfo(BaseModel):
    width: int
    height: int

class PipelineResponse(BaseModel):
    success: bool
    processing_time_ms: float
    detections: List[DetectionResult]
    frame_info: Optional[FrameInfo] = None

class BatchURLRequest(BaseModel):
    urls: List[str]
