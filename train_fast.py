from ultralytics import YOLO
import torch
import os
import shutil

def train_aggressive():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("CPU detected - using optimized CPU settings")
        batch_size = 32
        workers = 8
        epochs = 500
        cache = 'ram'
        model_size = 'yolov8s.pt'
    else:
        print("GPU detected - using aggressive GPU settings")
        batch_size = 128
        workers = 12
        epochs = 500
        cache = True
        model_size = 'yolov8m.pt'
    
    print(f"Model: {model_size}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {workers}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    
    model = YOLO(model_size)
    
    results = model.train(
        data='dataset/data.yaml',
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        name='recaptcha_aggressive',
        project='runs/train',
        exist_ok=True,
        patience=100,
        save=True,
        save_period=25,
        device=device,
        workers=workers,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.003,
        lrf=0.0001,
        momentum=0.95,
        weight_decay=0.001,
        warmup_epochs=10.0,
        warmup_momentum=0.9,
        warmup_bias_lr=0.15,
        box=8.0,
        cls=0.6,
        dfl=1.8,
        label_smoothing=0.05,
        nbs=64,
        hsv_h=0.03,
        hsv_s=0.9,
        hsv_v=0.6,
        degrees=10.0,
        translate=0.25,
        scale=0.95,
        shear=3.0,
        perspective=0.002,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.5,
        auto_augment='randaugment',
        erasing=0.6,
        crop_fraction=1.0,
        cache=cache,
        amp=True,
        fraction=1.0,
        close_mosaic=25,
        overlap_mask=True,
        plots=True,
        val=True
    )
    
    print("\nExporting models...")
    best_model_path = 'runs/train/recaptcha_aggressive/weights/best.pt'
    
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        
        best_model.export(format='onnx', dynamic=True, simplify=True)
        best_model.export(format='torchscript')
        
        shutil.copy(best_model_path, 'recaptcha_model_best.pt')
        print(f"Best model saved as recaptcha_model_best.pt")
        
        try:
            metrics = results.results_dict
            print(f"\nFinal Training Metrics:")
            print(f"mAP50: {metrics.get('metrics/mAP50(B)', 0):.4f}")
            print(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
            print(f"Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
            print(f"Recall: {metrics.get('metrics/recall(B)', 0):.4f}")
        except:
            pass
    
    last_model_path = 'runs/train/recaptcha_aggressive/weights/last.pt'
    if os.path.exists(last_model_path):
        shutil.copy(last_model_path, 'recaptcha_model_last.pt')
        print(f"Last model saved as recaptcha_model_last.pt")

if __name__ == '__main__':
    print("="*60)
    print("AGGRESSIVE RECAPTCHA TRAINING - OPTIMIZED FOR SPEED & ACCURACY")
    print("="*60)
    train_aggressive()
