from ultralytics import YOLO
import os
import pandas as pd

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)

    train_dir = 'runs/detect/runs/train/yolov26n_plate'
    model_path = os.path.join(train_dir, 'weights', 'best.pt')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Did training finish?")
        return

    # 1. Evaluate on test split
    print("--- Evaluating on Test Set ---")
    model = YOLO(model_path)
    metrics = model.val(data='dataset_yolo26/data.yaml', split='test', plots=True, save=True)
    
    print("\n--- Test Metrics ---")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@75:    {metrics.box.map75:.4f}")
    
    # 2. Analyze Overfitting (Train vs Val Loss)
    results_csv = os.path.join(train_dir, 'results.csv')
    if os.path.exists(results_csv):
        print("\n--- Overfitting Analysis ---")
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip() # Clean column names
        
        last_epoch = df.iloc[-1]
        best_epoch_idx = df['metrics/mAP50-95(B)'].idxmax()
        best_epoch = df.iloc[best_epoch_idx]
        
        print(f"Best Epoch: {best_epoch['epoch']}")
        print(f"Train Box Loss (Best): {best_epoch['train/box_loss']:.4f} -> (Last): {last_epoch['train/box_loss']:.4f}")
        print(f"Val Box Loss   (Best): {best_epoch['val/box_loss']:.4f} -> (Last): {last_epoch['val/box_loss']:.4f}")
        
        if last_epoch['val/box_loss'] > best_epoch['val/box_loss'] * 1.1:
            print("WARNING: Val loss increased by >10% from best epoch. Likely overfitting.")
        else:
            print("OK: Val loss stable. No severe overfitting.")

    print("\n--- Visual Curves ---")
    print(f"Validation curves and confusion matrix saved at: {metrics.save_dir}")

if __name__ == '__main__':
    main()
