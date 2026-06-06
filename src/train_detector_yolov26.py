from ultralytics import YOLO
import os

def main():
    # Ensure working directory is project root so relative paths work
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)

    # Load a model
    model = YOLO('yolo26n.pt')

    # Train the model
    results = model.train(
        data='dataset_yolo26/data.yaml',
        epochs=100,
        imgsz=416,  # Based on dataset specs
        device=0,   # Use GPU
        project='runs/train',
        name='yolov26n_plate',
        save=True
    )

if __name__ == '__main__':
    main()
