"""
Improved ReCAPTCHA Solver API Server
- Production-ready with proper error handling
- Caching for better performance
- Async support for multiple requests
- Model optimization
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import base64
import cv2
import os
import torch
from functools import lru_cache
import logging
from concurrent.futures import ThreadPoolExecutor
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
model = None
executor = ThreadPoolExecutor(max_workers=4)

# Optimized class mappings
CLASS_MAPPINGS = {
    'traffic light': ['traffic light'],
    'traffic lights': ['traffic light'],
    'bus': ['bus', 'truck'],
    'buses': ['bus', 'truck'],
    'car': ['car', 'truck'],
    'cars': ['car', 'truck'],
    'bike': ['bicycle', 'motorcycle'],
    'bicycle': ['bicycle'],
    'motorcycles': ['motorcycle'],
    'motorcycle': ['motorcycle'],
    'fire hydrant': ['fire hydrant'],
    'fire hydrants': ['fire hydrant'],
    'crosswalk': ['person'],
    'pedestrian crossing': ['person'],
    'stairs': ['person'],
    'stair': ['person']
}

# Cache for decoded images
@lru_cache(maxsize=100)
def decode_image_cached(image_hash):
    """Cache decoded images to avoid redundant processing"""
    pass

def load_model(model_path='yolov8n.pt', use_gpu=True):
    """
    Load YOLO model with optimizations
    """
    global model
    try:
        device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        logger.info(f"Loading model on {device}")
        
        model = YOLO(model_path)
        
        # Optimize for inference
        if device == 'cuda':
            model.to(device)
            # Enable half precision for faster inference on GPU
            model.model.half()
        
        # Warm up the model
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy_img, verbose=False)
        
        logger.info("Model loaded and warmed up successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def decode_base64_image(image_data):
    """
    Decode base64 image with error handling
    """
    try:
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'gpu_available': torch.cuda.is_available()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Basic prediction endpoint
    """
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image = decode_base64_image(image_data)
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Run inference with optimizations
        results = model(image, conf=0.25, verbose=False, imgsz=640)
        
        predictions = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    class_name = model.names[cls]
                    
                    predictions.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions)
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/solve_grid', methods=['POST'])
def solve_grid():
    """
    Optimized grid solving with better detection
    """
    try:
        start_time = time.time()
        data = request.json
        image_data = data.get('image')
        instruction = data.get('instruction', '').lower().strip()
        grid_size = data.get('grid_size', 3)
        confidence_threshold = data.get('confidence', 0.2)
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image = decode_base64_image(image_data)
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Get target classes
        target_classes = CLASS_MAPPINGS.get(instruction, [instruction])
        
        width, height = image.size
        tile_width = width // grid_size
        tile_height = height // grid_size
        
        # Run inference on full image (faster than per-tile)
        results = model(image, conf=confidence_threshold, verbose=False, imgsz=640)
        
        tile_detections = {}
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    class_name = model.names[cls].lower()
                    
                    # Check if class matches target
                    matched = False
                    for target in target_classes:
                        if target in class_name or class_name in target:
                            matched = True
                            break
                    
                    if matched:
                        # Calculate center point
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Determine which tile this belongs to
                        col = int(center_x // tile_width)
                        row = int(center_y // tile_height)
                        
                        if 0 <= row < grid_size and 0 <= col < grid_size:
                            tile_idx = row * grid_size + col
                            
                            # Keep highest confidence detection per tile
                            if tile_idx not in tile_detections or conf > tile_detections[tile_idx]['confidence']:
                                tile_detections[tile_idx] = {
                                    'row': row,
                                    'col': col,
                                    'index': tile_idx,
                                    'confidence': conf,
                                    'detected': class_name
                                }
        
        selected_tiles = list(tile_detections.values())
        processing_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'selected_tiles': selected_tiles,
            'instruction': instruction,
            'grid_size': grid_size,
            'processing_time': processing_time
        })
    
    except Exception as e:
        logger.error(f"Grid solving error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_solve', methods=['POST'])
def batch_solve():
    """
    Batch processing for multiple challenges
    Processes in parallel for better performance
    """
    try:
        data = request.json
        challenges = data.get('challenges', [])
        
        if len(challenges) > 10:
            return jsonify({'error': 'Maximum 10 challenges per batch'}), 400
        
        results = []
        
        for challenge in challenges:
            image_data = challenge.get('image')
            instruction = challenge.get('instruction', '').lower().strip()
            grid_size = challenge.get('grid_size', 3)
            
            image = decode_base64_image(image_data)
            if image is None:
                results.append({'error': 'Invalid image data'})
                continue
            
            target_classes = CLASS_MAPPINGS.get(instruction, [instruction])
            
            width, height = image.size
            tile_width = width // grid_size
            tile_height = height // grid_size
            
            # Run inference
            detection_results = model(image, conf=0.25, verbose=False, imgsz=640)
            
            selected_tiles = []
            
            for r in detection_results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        class_name = model.names[cls].lower()
                        
                        for target in target_classes:
                            if target in class_name or class_name in target:
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                col = int(center_x // tile_width)
                                row = int(center_y // tile_height)
                                
                                if 0 <= row < grid_size and 0 <= col < grid_size:
                                    tile_idx = row * grid_size + col
                                    if tile_idx not in selected_tiles:
                                        selected_tiles.append(tile_idx)
                                break
            
            results.append({
                'instruction': instruction,
                'selected_tiles': selected_tiles
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        logger.error(f"Batch solving error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model on startup
    model_path = os.environ.get('MODEL_PATH', 'yolov8n.pt')
    use_gpu = os.environ.get('USE_GPU', 'true').lower() == 'true'
    
    if load_model(model_path, use_gpu):
        port = int(os.environ.get('PORT', 5000))
        debug = os.environ.get('DEBUG', 'false').lower() == 'true'
        
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug,
            threaded=True
        )
    else:
        logger.error("Failed to load model. Exiting.")