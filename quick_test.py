import requests
import os
import base64

print("Quick API Test")
print("=" * 50)

print("\n1. Testing API connection...")
try:
    response = requests.get("http://localhost:5000/health", timeout=3)
    if response.status_code == 200:
        print("✓ API is online!")
        data = response.json()
        print(f"  Status: {data['status']}")
        print(f"  Model loaded: {data['model_loaded']}")
    else:
        print("✗ API not responding")
        print("\nStart the API first: python api_server.py")
        exit(1)
except:
    print("✗ Cannot connect to API")
    print("\nStart the API first: python api_server.py")
    exit(1)

print("\n2. Testing image detection...")
image_dir = "recap harvester/recaptcha_images"
if not os.path.exists(image_dir):
    image_dir = "recaptcha_images"

test_image = None
for file in os.listdir(image_dir):
    if file.endswith('.jpg') and 'full_grid' in file:
        test_image = os.path.join(image_dir, file)
        break

if not test_image:
    print("✗ No test images found")
    exit(1)

print(f"  Using: {os.path.basename(test_image)}")

with open(test_image, 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

test_instructions = ['bus', 'car', 'traffic light', 'bike']

for instruction in test_instructions:
    print(f"\n3. Testing detection for: {instruction}")
    try:
        response = requests.post("http://localhost:5000/solve_grid", json={
            "image": f"data:image/jpeg;base64,{image_data}",
            "instruction": instruction,
            "grid_size": 3
        }, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                tiles = data['selected_tiles']
                if len(tiles) > 0:
                    print(f"  ✓ Found {len(tiles)} tiles with {instruction}")
                    for tile in tiles[:3]:
                        print(f"    - Tile {tile['index']} ({tile['confidence']:.2f} confidence)")
                else:
                    print(f"  - No {instruction} detected")
            else:
                print(f"  ✗ Detection failed: {data.get('error')}")
        else:
            print(f"  ✗ API error: {response.status_code}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        break

print("\n" + "=" * 50)
print("Test complete! System is ready.")
print("\nRun START.bat to start the auto solver")
print("=" * 50)
