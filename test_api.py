import requests
import base64
import os
import json

API_URL = "http://localhost:5000"

def test_health():
    print("Testing API health...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API Status: {data['status']}")
            print(f"✓ Model Loaded: {data['model_loaded']}")
            return True
        else:
            print("✗ API health check failed")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to API: {e}")
        return False

def test_solve_grid():
    print("\nTesting grid solver...")
    
    test_image = None
    for file in os.listdir('recaptcha_images'):
        if file.startswith('full_grid_') and file.endswith('.jpg'):
            test_image = os.path.join('recaptcha_images', file)
            break
    
    if not test_image:
        print("✗ No test images found")
        return False
    
    print(f"Using test image: {test_image}")
    
    with open(test_image, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    test_cases = [
        {"instruction": "bus", "grid_size": 3},
        {"instruction": "traffic light", "grid_size": 3},
        {"instruction": "car", "grid_size": 3},
        {"instruction": "bike", "grid_size": 3}
    ]
    
    for test in test_cases:
        print(f"\nTesting: {test['instruction']}")
        
        try:
            response = requests.post(f"{API_URL}/solve_grid", json={
                "image": f"data:image/jpeg;base64,{image_data}",
                "instruction": test['instruction'],
                "grid_size": test['grid_size']
            })
            
            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    tiles = data['selected_tiles']
                    print(f"✓ Found {len(tiles)} matching tiles")
                    for tile in tiles:
                        print(f"  - Tile {tile['index']} (confidence: {tile['confidence']:.2f})")
                else:
                    print("✗ Solver failed")
            else:
                print(f"✗ API returned error: {response.status_code}")
        
        except Exception as e:
            print(f"✗ Test failed: {e}")
    
    return True

def main():
    print("ReCAPTCHA Solver API Test")
    print("=" * 50)
    
    if not test_health():
        print("\nAPI is not running. Start api_server.py first!")
        return
    
    test_solve_grid()
    
    print("\n" + "=" * 50)
    print("Testing complete!")

if __name__ == "__main__":
    main()
