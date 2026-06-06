import cv2
import numpy as np

def crop_with_padding(image: np.ndarray, box: list[int], padding_percent: float = 0.15) -> np.ndarray:
    """
    Crops the image using the bounding box with added percentage padding.
    box: [x1, y1, x2, y2]
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    
    box_w = x2 - x1
    box_h = y2 - y1
    
    pad_x = int(box_w * padding_percent)
    pad_y = int(box_h * padding_percent)
    
    new_x1 = max(0, x1 - pad_x)
    new_y1 = max(0, y1 - pad_y)
    new_x2 = min(w, x2 + pad_x)
    new_y2 = min(h, y2 + pad_y)
    
    return image[new_y1:new_y2, new_x1:new_x2]

def preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    """
    Applies Grayscale and CLAHE for better OCR readability.
    """
    if crop.size == 0:
        return crop
        
    # Resize to standard height (PaddleOCR prefers ~48px height)
    crop_h, crop_w = crop.shape[:2]
    target_h = 48
    if crop_h > 0:
        scale = target_h / crop_h
        target_w = int(crop_w * scale)
        if target_w > 0:
            crop = cv2.resize(crop, (target_w, target_h))
            
    # Convert to Grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
        
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Convert back to 3-channel for PaddleOCR (it expects 3 channels even if gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
