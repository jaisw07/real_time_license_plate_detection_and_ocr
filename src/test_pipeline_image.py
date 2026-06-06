import cv2
import os
import sys
import argparse

# Add project root to sys.path so 'from src.services' works
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.services.ocr import PlateOCR
from ultralytics import YOLO
from src.services.preprocessor import crop_with_padding, preprocess_for_ocr
from src.utils.postprocess import clean_ocr_text

def test_pipeline(image_path: str):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root_dir, r"C:\Users\SHREY\Desktop\ttl\runs\detect\runs\train\yolov11n_plate\weights\best.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: YOLO model not found at {model_path}")
        return
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
        
    # 1. Detection
    print("--- Stage 1: Detection ---")
    detector = YOLO(model_path)
    results = detector(image)[0]
    
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    
    if len(boxes) == 0:
        print("No plates detected.")
        return
        
    print(f"Detected {len(boxes)} plates.")
    
    # 2. OCR Service Initialization
    ocr_service = PlateOCR()
    
    for i, (box, conf) in enumerate(zip(boxes, confs)):
        print(f"\nPlate {i+1} (Det Conf: {conf:.2f}):")
        
        # 3. Crop & Preprocess
        crop = crop_with_padding(image, box, padding_percent=0.15)
        preprocessed_crop = preprocess_for_ocr(crop)
        
        # 4. Recognize
        raw_text, ocr_conf = ocr_service.recognize(preprocessed_crop)
        
        # 5. Post-process
        cleaned_text = clean_ocr_text(raw_text)
        
        print(f"  Raw OCR:     '{raw_text}' (Conf: {ocr_conf:.2f})")
        print(f"  Cleaned OCR: '{cleaned_text}'")
        
        # Draw on image for visualization
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, cleaned_text, (x1, max(0, y1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
    # Save output
    output_path = os.path.join(root_dir, "pipeline_test_output.jpg")
    cv2.imwrite(output_path, image)
    print(f"\nSaved visualization to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test License Plate Pipeline")
    parser.add_argument("image_path", help="Path to input image")
    args = parser.parse_args()
    
    # Ensure working directory is project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_pipeline(args.image_path)
