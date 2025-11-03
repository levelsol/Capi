from ultralytics import YOLO
import torch
import os

def train_recaptcha_model():
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data='dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='recaptcha_model',
        project='runs/train',
        exist_ok=True,
        patience=20,
        save=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        workers=0,
        pretrained=True,
        optimizer='Adam',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment='randaugment',
        erasing=0.4,
        crop_fraction=1.0
    )
    
    model.export(format='onnx')
    print("Model training completed!")
    
    best_model_path = 'runs/train/recaptcha_model/weights/best.pt'
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, 'recaptcha_model.pt')
        print(f"Best model saved as recaptcha_model.pt")

if __name__ == '__main__':
    import shutil
    train_recaptcha_model()
