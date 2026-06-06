from rapidocr import RapidOCR
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PlateOCR:
    def __init__(self):
        # Initialize RapidOCR
        # We'll use the default init but override parameters during the call
        self.ocr = RapidOCR()
        
    def recognize(self, preprocessed_crop: np.ndarray) -> tuple[str, float]:
        """
        Takes a preprocessed image crop.
        Returns (recognized_text, confidence).
        """
        if preprocessed_crop is None or preprocessed_crop.size == 0:
            return "", 0.0
            
        try:
            # We'll go back to the default call as it works well
            # result = self.ocr(img)
            output = self.ocr(preprocessed_crop)
            
            if not output:
                return "", 0.0
                
            # RapidOCR (v1.3.0+) returns RapidOCROutput or TextRecOutput objects
            # These have .txts and .scores attributes
            texts = []
            scores = []
            
            if hasattr(output, 'txts') and output.txts:
                texts = [str(t) for t in output.txts]
            if hasattr(output, 'scores') and output.scores:
                scores = [float(s) for s in output.scores]
                
            # Fallback for older versions or if it returns [ [box, text, score], ... ]
            if not texts and isinstance(output, list):
                for res in output:
                    if isinstance(res, list) and len(res) >= 3:
                        texts.append(str(res[1]))
                        scores.append(float(res[2]))
            
            if not texts:
                return "", 0.0
                
            final_text = " ".join(texts).strip()
            avg_conf = sum(scores) / len(scores) if scores else 0.0
            
            return final_text, avg_conf
            
        except Exception as e:
            logger.error(f"OCR recognition error: {e}")
            return "", 0.0
