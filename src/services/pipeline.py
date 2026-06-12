import cv2
import numpy as np
import os
from ultralytics import YOLO
from src.services.ocr import PlateOCR
from src.services.preprocessor import crop_with_padding, preprocess_for_ocr
from src.utils.postprocess import clean_ocr_text, validate_plate, deduplicate_plates
from src.utils.color_extractor import extract_plate_color
import logging

logger = logging.getLogger(__name__)

class LicensePlatePipeline:
    def __init__(self, model_path: str = "weights/yolov26n_plate.pt", device: str = "cuda", debug: bool = True):
        """
        Initializes the detection and OCR models.
        """
        logger.info(f"Initializing LicensePlatePipeline with model: {model_path}")
        self.model_path = model_path
        self.detector = YOLO(model_path)
        self.ocr_service = PlateOCR()
        self.device = device
        self.debug = debug
        if self.debug:
            os.makedirs("debug_crops", exist_ok=True)
        
    def process_image(self, image: np.ndarray):
        """
        Processes a single image and returns detections.
        """
        if image is None:
            return []
            
        # 1. Detection
        results = self.detector(image, device=self.device)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        
        detections = []
        
        for i, (box, det_conf) in enumerate(zip(boxes, confs)):
            # 2. Crop & Preprocess
            crop = crop_with_padding(image, box, padding_percent=0.15)
            preprocessed_crop = preprocess_for_ocr(crop)
            
            if self.debug:
                cv2.imwrite(f"debug_crops/crop_{i}.jpg", preprocessed_crop)
            
            # 3. Recognize
            raw_text, ocr_conf = self.ocr_service.recognize(preprocessed_crop)
            
            # 4. Color Extraction
            x1, y1, x2, y2 = map(int, box)
            h_img, w_img = image.shape[:2]
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(w_img, x2), min(h_img, y2)
            color_crop = image[y1_c:y2_c, x1_c:x2_c]
            plate_colour = extract_plate_color(color_crop)
            
            # 5. Post-process
            cleaned_text = clean_ocr_text(raw_text)
            
            x1_f, y1_f, x2_f, y2_f = map(float, box)
            
            detections.append({
                "plate_number": cleaned_text,
                "raw_ocr": raw_text,
                "confidence": {
                    "detection": float(det_conf),
                    "ocr": float(ocr_conf)
                },
                "bounding_box": {
                    "x1": x1_f, "y1": y1_f, "x2": x2_f, "y2": y2_f
                },
                "plate_colour": plate_colour
            })
            
        return deduplicate_plates(detections)

    def track_and_process(self, image: np.ndarray, model=None):
        """
        Used for video streams. Integrates tracking to avoid redundant OCR.
        If model is provided, it uses that specific instance (for session isolation).
        """
        # 1. Detection with tracking
        detector = model if model is not None else self.detector
        results = detector.track(image, persist=True, device=self.device, verbose=False)[0]
        
        if results.boxes.id is None:
            return []
            
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)
        
        detections = []
        for box, det_conf, track_id in zip(boxes, confs, ids):
            # We return detection info; the caller (WebSocket handler) 
            # will decide whether to run OCR based on track_id cache.
            x1, y1, x2, y2 = map(float, box)
            detections.append({
                "track_id": int(track_id),
                "confidence": float(det_conf),
                "bounding_box": {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                },
                "box": box # raw box for cropping if needed
            })
            
        return detections

    def run_ocr_on_crop(self, image: np.ndarray, box: np.ndarray):
        """
        Helper for partial pipeline runs (e.g. OCR-only after tracking cache miss).
        """
        x1, y1, x2, y2 = map(int, box)
        h_img, w_img = image.shape[:2]
        x1_c, y1_c = max(0, x1), max(0, y1)
        x2_c, y2_c = min(w_img, x2), min(h_img, y2)
        color_crop = image[y1_c:y2_c, x1_c:x2_c]
        plate_colour = extract_plate_color(color_crop)

        crop = crop_with_padding(image, box, padding_percent=0.15)
        preprocessed_crop = preprocess_for_ocr(crop)
        raw_text, ocr_conf = self.ocr_service.recognize(preprocessed_crop)
        cleaned_text = clean_ocr_text(raw_text)
        if not validate_plate(cleaned_text):
            cleaned_text = ""
        return cleaned_text, raw_text, float(ocr_conf), plate_colour
