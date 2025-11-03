from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import base64
import cv2
import os

app = Flask(__name__)
CORS(app)

model = None

CLASS_MAPPINGS = {
    'traffic light': ['traffic light'],
    'traffic lights': ['traffic light'],
    'bus': ['bus', 'truck'],
    'buses': ['bus', 'truck'],
    'car': ['car', 'truck'],
    'cars': ['car', 'truck'],
    'bike': ['bicycle', 'motorcycle'],
    'bicycle': ['bicycle'],
    'motorcycle': ['motorcycle'],
    'fire hydrant': ['fire hydrant'],
    'crosswalk': ['person'],
    'pedestrian crossing': ['person'],
    'stairs': ['person']
}

def load_model():
    global model
    model = YOLO('yolov8n.pt')
    print("Model loaded successfully")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        results = model(image)
        
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
        return jsonify({'error': str(e)}), 500

@app.route('/solve_grid', methods=['POST'])
def solve_grid():
    try:
        data = request.json
        image_data = data.get('image')
        instruction = data.get('instruction', '').lower().strip()
        grid_size = data.get('grid_size', 3)
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        target_classes = CLASS_MAPPINGS.get(instruction, [instruction])
        
        width, height = image.size
        tile_width = width // grid_size
        tile_height = height // grid_size
        
        results = model(image, conf=0.2, verbose=False, imgsz=640)
        
        tile_detections = {}
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    class_name = model.names[cls].lower()
                    
                    matched = False
                    for target in target_classes:
                        if target in class_name or class_name in target:
                            matched = True
                            break
                    
                    if matched:
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        col = int(center_x // tile_width)
                        row = int(center_y // tile_height)
                        
                        if 0 <= row < grid_size and 0 <= col < grid_size:
                            tile_idx = row * grid_size + col
                            
                            if tile_idx not in tile_detections or conf > tile_detections[tile_idx]['confidence']:
                                tile_detections[tile_idx] = {
                                    'row': row,
                                    'col': col,
                                    'index': tile_idx,
                                    'confidence': conf,
                                    'detected': class_name
                                }
        
        selected_tiles = list(tile_detections.values())
        
        return jsonify({
            'success': True,
            'selected_tiles': selected_tiles,
            'instruction': instruction,
            'grid_size': grid_size
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_solve', methods=['POST'])
def batch_solve():
    try:
        data = request.json
        challenges = data.get('challenges', [])
        
        results = []
        for challenge in challenges:
            image_data = challenge.get('image')
            instruction = challenge.get('instruction', '').lower().strip()
            grid_size = challenge.get('grid_size', 3)
            
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            target_classes = CLASS_MAPPINGS.get(instruction, [instruction])
            
            width, height = image.size
            tile_width = width // grid_size
            tile_height = height // grid_size
            
            selected_tiles = []
            
            for row in range(grid_size):
                for col in range(grid_size):
                    left = col * tile_width
                    top = row * tile_height
                    right = left + tile_width
                    bottom = top + tile_height
                    
                    tile = image.crop((left, top, right, bottom))
                    
                    tile_results = model(tile, conf=0.25, verbose=False)
                    
                    for r in tile_results:
                        boxes = r.boxes
                        if boxes is not None and len(boxes) > 0:
                            for box in boxes:
                                conf = box.conf[0].item()
                                cls = int(box.cls[0].item())
                                class_name = model.names[cls].lower()
                                
                                for target in target_classes:
                                    if target in class_name or class_name in target:
                                        selected_tiles.append(row * grid_size + col)
                                        break
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
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
