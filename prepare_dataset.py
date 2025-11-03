import os
import shutil
import json
import random
from PIL import Image
import re

CLASSES = {
    'bus': 0,
    'car': 1,
    'bike': 2,
    'pedestrian crossing': 3,
    'fire hydrant': 4,
    'traffic light': 5,
    'motorcycle': 6,
    'crosswalk': 7
}

def parse_challenge_log():
    challenges = {}
    log_paths = ['challenge_log.txt', 'recap harvester/challenge_log.txt']
    log_file = None
    
    for path in log_paths:
        if os.path.exists(path):
            log_file = path
            break
    
    if not log_file:
        print("challenge_log.txt not found!")
        return challenges
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        match = re.match(r'Round (\d+): Select all squares with (.+)', line.strip())
        if match:
            round_num = int(match.group(1)) - 1
            challenge_type = match.group(2).lower()
            if challenge_type != 'unknown':
                challenges[round_num] = challenge_type
    
    return challenges

def prepare_yolo_dataset():
    challenges = parse_challenge_log()
    
    os.makedirs('dataset/images/train', exist_ok=True)
    os.makedirs('dataset/images/val', exist_ok=True)
    os.makedirs('dataset/labels/train', exist_ok=True)
    os.makedirs('dataset/labels/val', exist_ok=True)
    
    all_rounds = list(challenges.keys())
    random.shuffle(all_rounds)
    
    split_idx = int(len(all_rounds) * 0.8)
    train_rounds = all_rounds[:split_idx]
    val_rounds = all_rounds[split_idx:]
    
    for round_num in train_rounds:
        if round_num in challenges:
            process_round(round_num, challenges[round_num], 'train')
    
    for round_num in val_rounds:
        if round_num in challenges:
            process_round(round_num, challenges[round_num], 'val')
    
    with open('dataset/data.yaml', 'w') as f:
        f.write(f'train: {os.path.abspath("dataset/images/train")}\n')
        f.write(f'val: {os.path.abspath("dataset/images/val")}\n\n')
        f.write('nc: 8\n')
        f.write('names: ["bus", "car", "bike", "pedestrian_crossing", "fire_hydrant", "traffic_light", "motorcycle", "crosswalk"]\n')

def process_round(round_num, challenge_type, split):
    full_grid_path = None
    image_dirs = ['recaptcha_images', 'recap harvester/recaptcha_images']
    
    for img_dir in image_dirs:
        if os.path.exists(img_dir):
            for file in os.listdir(img_dir):
                if file.startswith(f'full_grid_round{round_num}_'):
                    full_grid_path = os.path.join(img_dir, file)
                    break
            if full_grid_path:
                break
    
    if full_grid_path and os.path.exists(full_grid_path):
        dest_image = os.path.join('dataset/images', split, f'round_{round_num}.jpg')
        shutil.copy(full_grid_path, dest_image)
        
        img = Image.open(full_grid_path)
        width, height = img.size
        
        label_path = os.path.join('dataset/labels', split, f'round_{round_num}.txt')
        with open(label_path, 'w') as f:
            if challenge_type in CLASSES:
                class_id = CLASSES[challenge_type]
                f.write(f'{class_id} 0.5 0.5 1.0 1.0\n')

if __name__ == '__main__':
    print("Preparing YOLO dataset...")
    prepare_yolo_dataset()
    print("Dataset prepared!")
