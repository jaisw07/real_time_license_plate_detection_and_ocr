import httpx
import os
import json

def test_single_image(image_path: str, url: str = "http://localhost:8000/detect/"):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        try:
            response = httpx.post(url, files=files, timeout=30.0)
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
