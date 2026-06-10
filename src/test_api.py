import httpx
import os
import json
import base64
from dotenv import load_dotenv

def test_single_image(image_path: str, url: str = "http://localhost:8000/detect/"):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    with open(image_path, "rb") as f:
        img_bytes = f.read()
        b64_str = base64.b64encode(img_bytes).decode("utf-8")
        
    payload = {"image": b64_str}
    
    # Load API Key for auth
    load_dotenv()
    api_key = os.getenv("API_KEY", "your_placeholder_secret_key_here")
    headers = {"X-API-Key": api_key}
    
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        if response.status_code == 200:
            print("Success!")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    # Use a sample image where the plate is larger and more likely to be recognized
    sample_img = r"dataset/test/images/CarLongPlate112_jpg.rf.a516ef3b84ef951501d0a501d4d69dcf.jpg"
    test_single_image(sample_img)
