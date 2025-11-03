#!/usr/bin/env python3
"""
ULTRA-FAST RECAPTCHA TRAINING SCRIPT
Optimized for maximum speed and accuracy
"""

from ultralytics import YOLO
import torch
import os
import shutil
import yaml
from pathlib import Path
import multiprocessing

def get_optimal_settings():
    """
    Determine optimal training settings based on hardware
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = multiprocessing.cpu_count()
    
    if device == 'cuda':
        # Get GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory >= 12:
            # High-end GPU (RTX 3090, 4090, A100, etc.)
            return {
                'model': 'yolov8m.pt',
                'batch': 64,
                'workers': min(16, num_workers),
                'imgsz': 640,
                'cache': True,
                'epochs': 300
            }
        elif gpu_memory >= 8:
            # Mid-range GPU (RTX 3070, 4070, etc.)
            return {
                'model': 'yolov8s.pt',
                'batch': 32,
                'workers': min(12, num_workers),
                'imgsz': 640,
                'cache': True,
                'epochs': 300
            }
        elif gpu_memory >= 6:
            # Entry-level GPU (RTX 3060, etc.)
            return {
                'model': 'yolov8n.pt',
                'batch': 16,
                'workers': min(8, num_workers),
                'imgsz': 640,
                'cache': True,
                'epochs': 300
            }
        else:
            # Low VRAM GPU
            return {
                'model': 'yolov8n.pt',
                'batch': 8,
                'workers': min(4, num_workers),
                'imgsz': 416,
                'cache': False,
                'epochs': 200
            }
    else:
        # CPU training
        print("CPU Detected - Using CPU-optimized settings")
        return {
            'model': 'yolov8n.pt',
            'batch': 8,
            'workers': min(8, num_workers),
            'imgsz': 416,
            'cache': 'ram',
            'epochs': 200
        }

def verify_dataset(data_yaml='dataset/data.yaml'):
    """
    Verify dataset exists and has proper structure
    """
    if not os.path.exists(data_yaml):
        print(f"ERROR: Dataset config not found at {data_yaml}")
        return False
    
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    train_path = data.get('train', '')
    val_path = data.get('val', '')
    
    # Check if paths exist
    if not os.path.exists(os.path.join('dataset', train_path)):
        print(f"ERROR: Training images not found")
        return False
    
    if not os.path.exists(os.path.join('dataset', val_path)):
        print(f"ERROR: Validation images not found")
        return False
    
    print("✓ Dataset verified")
    return True

def train_ultra_fast():
    """
    Train with maximum speed optimizations
    """
    print("="*70)
    print("ULTRA-FAST RECAPTCHA TRAINING")
    print("="*70)
    
    # Verify dataset
    if not verify_dataset():
        return
    
    # Get optimal settings
    settings = get_optimal_settings()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nTraining Configuration:")
    print(f"  Device: {device}")
    print(f"  Model: {settings['model']}")
    print(f"  Batch Size: {settings['batch']}")
    print(f"  Workers: {settings['workers']}")
    print(f"  Image Size: {settings['imgsz']}")
    print(f"  Cache: {settings['cache']}")
    print(f"  Epochs: {settings['epochs']}")
    print()
    
    # Load model
    model = YOLO(settings['model'])
    
    # Training arguments optimized for speed
    train_args = {
        'data': 'dataset/data.yaml',
        'epochs': settings['epochs'],
        'batch': settings['batch'],
        'imgsz': settings['imgsz'],
        'device': device,
        'workers': settings['workers'],
        'cache': settings['cache'],
        
        # Project settings
        'project': 'runs/train',
        'name': 'recaptcha_ultrafast',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True,
        
        # Optimization settings
        'optimizer': 'AdamW',
        'lr0': 0.01,  # Higher learning rate for faster convergence
        'lrf': 0.001,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # Augmentation (balanced for speed and accuracy)
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.3,
        'auto_augment': 'randaugment',
        'erasing': 0.4,
        
        # Training settings
        'patience': 50,  # Early stopping
        'save': True,
        'save_period': 10,  # Save every 10 epochs
        'val': True,
        'plots': True,
        
        # Performance
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,
        'close_mosaic': 10,
    }
    
    # Start training
    print("Starting training...")
    print("="*70)
    
    results = model.train(**train_args)
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)
    
    # Export model
    best_model_path = 'runs/train/recaptcha_ultrafast/weights/best.pt'
    
    if os.path.exists(best_model_path):
        print("\nExporting models...")
        
        best_model = YOLO(best_model_path)
        
        # Export to different formats
        try:
            # ONNX for deployment
            best_model.export(format='onnx', dynamic=True, simplify=True)
            print("✓ ONNX export complete")
        except Exception as e:
            print(f"✗ ONNX export failed: {e}")
        
        try:
            # TorchScript for PyTorch deployment
            best_model.export(format='torchscript')
            print("✓ TorchScript export complete")
        except Exception as e:
            print(f"✗ TorchScript export failed: {e}")
        
        # Copy best model to root
        shutil.copy(best_model_path, 'recaptcha_best.pt')
        print(f"✓ Best model saved as recaptcha_best.pt")
        
        # Print metrics
        try:
            metrics = results.results_dict
            print(f"\nFinal Metrics:")
            print(f"  mAP50: {metrics.get('metrics/mAP50(B)', 0):.4f}")
            print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
            print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
            print(f"  Recall: {metrics.get('metrics/recall(B)', 0):.4f}")
        except:
            print("\nMetrics not available")
    
    print("\n" + "="*70)
    print("All done! Use recaptcha_best.pt for inference")
    print("="*70)

if __name__ == '__main__':
    train_ultra_fast()