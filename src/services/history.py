import os
import json
import time
import cv2
import numpy as np
import aiofiles
import logging
import uuid
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class HistoryManager:
    def __init__(self, base_dir: str = "data/history"):
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, "images")
        self.metadata_file = os.path.join(base_dir, "history.json")
        
        os.makedirs(self.images_dir, exist_ok=True)
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, "w") as f:
                json.dump([], f)

    async def add_entry(self, frame: np.ndarray, plate_number: str, confidence: Dict[str, float], bounding_box: Dict[str, float], track_id: Optional[int] = None):
        """
        Adds a new detection entry to history.
        """
        entry_id = str(uuid.uuid4())
        filename = f"{entry_id}.jpg"
        filepath = os.path.join(self.images_dir, filename)

        # Draw bounding box on the frame before saving
        frame_with_box = frame.copy()
        cv2.rectangle(
            frame_with_box, 
            (int(bounding_box["x1"]), int(bounding_box["y1"])), 
            (int(bounding_box["x2"]), int(bounding_box["y2"])), 
            (0, 255, 0), 2
        )

        # Save image
        cv2.imwrite(filepath, frame_with_box)

        new_entry = {
            "id": entry_id,
            "timestamp": time.time(),
            "plate_number": plate_number,
            "confidence": confidence,
            "bounding_box": bounding_box,
            "image_path": f"/history/images/{filename}",
            "track_id": track_id
        }

        # Update metadata file (async)
        async with aiofiles.open(self.metadata_file, mode="r+") as f:
            content = await f.read()
            history = json.loads(content)
            history.insert(0, new_entry)
            # Keep only last 100 entries for performance
            history = history[:100]
            await f.seek(0)
            await f.write(json.dumps(history, indent=2))
            await f.truncate()

        return new_entry

    def get_history(self, limit: int = 50) -> List[Dict]:
        """
        Returns the last N history entries.
        """
        try:
            with open(self.metadata_file, "r") as f:
                history = json.load(f)
                return history[:limit]
        except Exception as e:
            logger.error(f"Failed to read history: {e}")
            return []
