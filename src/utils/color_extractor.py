import cv2
import numpy as np

def extract_plate_color(crop: np.ndarray) -> str:
    """
    Extracts the dominant color of a license plate from its cropped image.
    Classifies the color into Red, White, Blue, Green, Yellow, Black.
    Uses an HSV space partition on an inset crop to exclude vehicle paint and margins.
    """
    if crop is None or crop.size == 0:
        return "Unknown"
        
    # Crop the center region (middle 70% height and 80% width) to avoid margins/surroundings
    h_c, w_c = crop.shape[:2]
    y_start = int(h_c * 0.15)
    y_end = int(h_c * 0.85)
    x_start = int(w_c * 0.10)
    x_end = int(w_c * 0.90)
    
    center_crop = crop[y_start:y_end, x_start:x_end]
    if center_crop.size == 0:
        center_crop = crop
        
    hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Complete partition HSV mapping
    black_mask = (v < 55)
    not_black = ~black_mask
    white_mask = not_black & (s < 45)
    colored = not_black & (s >= 45)
    
    red_mask = colored & ((h < 15) | (h >= 140))
    yellow_mask = colored & (h >= 15) & (h < 38)
    green_mask = colored & (h >= 38) & (h < 90)
    blue_mask = colored & (h >= 90) & (h < 140)
    
    counts = {
        "Black": np.sum(black_mask),
        "White": np.sum(white_mask),
        "Red": np.sum(red_mask),
        "Yellow": np.sum(yellow_mask),
        "Green": np.sum(green_mask),
        "Blue": np.sum(blue_mask)
    }
    
    return max(counts, key=counts.get)
