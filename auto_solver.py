import os
import re
import time
import random
import requests
import json
import base64
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO
from datetime import datetime

SITE_URL = "https://2captcha.com/demo/recaptcha-v2"
DOWNLOAD_DIR = "recaptcha_images"
RECAPTCHA_BASE = "https://www.google.com"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
SOLVER_API = "http://localhost:5000"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

current_v = None
current_co = None
current_challenge_data = None
image_counter = 0
round_counter = 0

session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Referer": SITE_URL,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
})

def extract_sitekey(html):
    match = re.search(r'data-sitekey=["\']([^"\']+)["\']', html)
    if match:
        return match.group(1)
    return None

def get_current_v_param():
    try:
        api_url = "https://www.google.com/recaptcha/api.js"
        response = session.get(api_url)
        if response.status_code != 200:
            return None
        v_match = re.search(r'releases/([^/]+)', response.text)
        if v_match:
            return v_match.group(1)
        return None
    except:
        return None

def get_anchor_token(sitekey):
    global current_v, current_co

    parsed_url = urlparse(SITE_URL)
    if parsed_url.scheme == 'https':
        origin = f"{parsed_url.scheme}://{parsed_url.netloc.split(':')[0]}:443"
    else:
        origin = f"{parsed_url.scheme}://{parsed_url.netloc.split(':')[0]}:80"

    import base64
    co = base64.urlsafe_b64encode(origin.encode('utf-8')).decode('utf-8').rstrip('=')

    v = get_current_v_param()
    if not v:
        v = "r20241024151857"

    cb = str(random.randint(10000000, 99999999))

    current_v = v
    current_co = co

    anchor_url = f"{RECAPTCHA_BASE}/recaptcha/api2/anchor"
    params = {
        "ar": "1",
        "k": sitekey,
        "co": co,
        "hl": "en",
        "v": v,
        "size": "normal",
        "cb": cb
    }

    response = session.get(anchor_url, params=params)
    if response.status_code != 200:
        return None

    token_match = re.search(r'id="recaptcha-token"\s+value="([^"]+)"', response.text)
    if token_match:
        return token_match.group(1)

    return None

def trigger_challenge(sitekey, initial_token):
    global current_challenge_data

    reload_url = f"{RECAPTCHA_BASE}/recaptcha/api2/reload"
    params = {"k": sitekey}

    data = {
        "v": current_v,
        "reason": "i",
        "c": initial_token,
        "k": sitekey,
        "co": current_co,
        "hl": "en",
        "size": "normal",
        "chr": "[89,64,27]",
        "vh": "",
        "bg": ""
    }

    response = session.post(reload_url, params=params, data=data)
    if response.status_code != 200:
        return None

    try:
        if response.text.startswith(')]}\'',):
            resp_text = response.text[5:]
        else:
            resp_text = response.text

        data_json = json.loads(resp_text)

        if isinstance(data_json, list) and len(data_json) > 1 and data_json[1]:
            challenge_token = data_json[1]
            current_challenge_data = data_json
            return challenge_token
        return None
    except:
        return None

def extract_instruction():
    if not current_challenge_data or len(current_challenge_data) < 5:
        return "Unknown"

    challenge_info = current_challenge_data[4]
    if not challenge_info or not isinstance(challenge_info, list) or len(challenge_info) < 2:
        return "Unknown"

    details = challenge_info[1]
    if not details or not isinstance(details, list) or len(details) < 7:
        return "Unknown"

    instruction = details[6] if details[6] else "Unknown"
    return instruction

def solve_challenge_with_api(image_content, instruction, grid_size=3):
    try:
        image_b64 = base64.b64encode(image_content).decode('utf-8')
        
        response = requests.post(f"{SOLVER_API}/solve_grid", json={
            "image": f"data:image/jpeg;base64,{image_b64}",
            "instruction": instruction,
            "grid_size": grid_size
        }, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return [tile['index'] for tile in data.get('selected_tiles', [])]
        
        return []
    except Exception as e:
        print(f"Solver API error: {e}")
        return []

def submit_solution(sitekey, challenge_token, selected_indices):
    reload_url = f"{RECAPTCHA_BASE}/recaptcha/api2/userverify"
    params = {"k": sitekey}
    
    response_array = [[]]
    for idx in selected_indices:
        response_array[0].append(idx)
    
    data = {
        "v": current_v,
        "c": challenge_token,
        "response": json.dumps(response_array),
        "k": sitekey,
        "co": current_co,
        "hl": "en",
        "size": "normal",
        "chr": "[89,64,27]",
        "vh": "",
        "bg": ""
    }
    
    response = session.post(reload_url, params=params, data=data)
    if response.status_code != 200:
        return False, None
    
    try:
        if response.text.startswith(')]}\'',):
            resp_text = response.text[5:]
        else:
            resp_text = response.text
        
        data_json = json.loads(resp_text)
        
        if isinstance(data_json, list) and len(data_json) > 1:
            if data_json[0] == "uvresp":
                token = data_json[1]
                if token:
                    return True, token
        
        return False, None
    except:
        return False, None

def download_and_solve(sitekey, challenge_token):
    global image_counter, round_counter
    instruction = extract_instruction()

    payload_url = f"{RECAPTCHA_BASE}/recaptcha/api2/payload"
    payload_params = {"c": challenge_token, "k": sitekey}

    payload_resp = session.get(payload_url, params=payload_params)
    if payload_resp.status_code != 200:
        return [], instruction, None

    try:
        grid_size = 3
        if current_challenge_data and len(current_challenge_data) > 4:
            challenge_info = current_challenge_data[4]
            if challenge_info and len(challenge_info) > 1:
                details = challenge_info[1]
                if details and len(details) > 2:
                    grid_size = details[2] if details[2] else 3
                if details and len(details) > 3:
                    if details[3] == 4:
                        grid_size = 4

        grid_img = Image.open(BytesIO(payload_resp.content))
        width, height = grid_img.size
        
        if width == height and width == 450:
            grid_size = 3
        elif width == height and width == 300:
            grid_size = 4

        selected_indices = solve_challenge_with_api(payload_resp.content, instruction, grid_size)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_image_path = os.path.join(DOWNLOAD_DIR, f"solved_round{round_counter}_{timestamp}.jpg")
        with open(full_image_path, 'wb') as f:
            f.write(payload_resp.content)

        return selected_indices, instruction, full_image_path

    except Exception as e:
        print(f"Error: {e}")
        return [], instruction, None

def request_new_challenge(sitekey, previous_token):
    global current_challenge_data
    
    reload_url = f"{RECAPTCHA_BASE}/recaptcha/api2/reload"
    params = {"k": sitekey}
    
    data = {
        "v": current_v,
        "reason": "r",
        "c": previous_token,
        "k": sitekey,
        "co": current_co,
        "hl": "en",
        "size": "normal",
        "chr": "[89,64,27]",
        "vh": "",
        "bg": ""
    }
    
    response = session.post(reload_url, params=params, data=data)
    if response.status_code != 200:
        return None
    
    try:
        if response.text.startswith(')]}\'',):
            resp_text = response.text[5:]
        else:
            resp_text = response.text
        
        data_json = json.loads(resp_text)
        
        if isinstance(data_json, list) and len(data_json) > 1 and data_json[1]:
            challenge_token = data_json[1]
            current_challenge_data = data_json
            return challenge_token
        return None
    except:
        return None

def main():
    global round_counter
    
    print("ReCAPTCHA Auto Solver")
    print("Checking solver API...")
    
    try:
        api_check = requests.get(f"{SOLVER_API}/health", timeout=5)
        if api_check.status_code != 200:
            print("Solver API is not running! Start api_server.py first.")
            return
    except:
        print("Cannot connect to solver API! Start api_server.py first.")
        return
    
    print("Solver API is ready!")
    
    resp = session.get(SITE_URL)
    if resp.status_code != 200:
        print("Failed to load demo page")
        return
    
    sitekey = extract_sitekey(resp.text)
    if not sitekey:
        print("Failed to extract sitekey")
        return
    
    initial_token = get_anchor_token(sitekey)
    if not initial_token:
        print("Failed to get anchor token")
        return
    
    time.sleep(1)
    
    challenge_token = trigger_challenge(sitekey, initial_token)
    if not challenge_token:
        print("Failed to trigger initial challenge")
        return
    
    solved_count = 0
    failed_count = 0
    
    try:
        while True:
            print(f"\n{'='*60}")
            print(f"ROUND {round_counter + 1}")
            print(f"{'='*60}")
            
            selected_indices, instruction, full_image = download_and_solve(sitekey, challenge_token)
            
            print(f"Challenge: Select all squares with {instruction}")
            print(f"Selected tiles: {selected_indices}")
            
            if selected_indices:
                success, solution_token = submit_solution(sitekey, challenge_token, selected_indices)
                
                if success and solution_token:
                    print(f"✓ SOLVED! Token: {solution_token[:50]}...")
                    solved_count += 1
                else:
                    print("✗ Solution rejected")
                    failed_count += 1
            else:
                print("✗ No tiles selected (no matches found)")
                failed_count += 1
            
            print(f"Success rate: {solved_count}/{solved_count + failed_count} ({(solved_count/(solved_count + failed_count)*100):.1f}%)")
            
            round_counter += 1
            time.sleep(random.uniform(2, 4))
            
            new_token = request_new_challenge(sitekey, challenge_token)
            if new_token:
                challenge_token = new_token
            else:
                initial_token = get_anchor_token(sitekey)
                if initial_token:
                    challenge_token = trigger_challenge(sitekey, initial_token)
                    if not challenge_token:
                        print("Failed to restart challenge process")
                        break
                else:
                    print("Failed to get new anchor token")
                    break
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    print(f"\n{'='*60}")
    print(f"Total rounds: {round_counter}")
    print(f"Solved: {solved_count}")
    print(f"Failed: {failed_count}")
    if solved_count + failed_count > 0:
        print(f"Success rate: {(solved_count/(solved_count + failed_count)*100):.1f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
