import os
import subprocess
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(1)

def main():
    print("ReCAPTCHA Solver Setup")
    print("=" * 60)
    
    print("\n1. Installing dependencies...")
    run_command(f"{sys.executable} -m pip install -r requirements.txt")
    
    print("\n2. Preparing dataset from collected images...")
    if os.path.exists("recaptcha_images") and os.path.exists("challenge_log.txt"):
        run_command(f"{sys.executable} prepare_dataset.py")
    else:
        print("No training data found. Run h.py first to collect images.")
        return
    
    print("\n3. Training model (this may take a while)...")
    if os.path.exists("dataset/data.yaml"):
        run_command(f"{sys.executable} train_model.py")
    else:
        print("Dataset not prepared. Check prepare_dataset.py")
        return
    
    print("\n" + "=" * 60)
    print("Setup completed!")
    print("\nTo use the solver:")
    print("1. Start the API server: python api_server.py")
    print("2. Open index.html in a browser for manual testing")
    print("3. Run auto_solver.py for automatic solving")

if __name__ == "__main__":
    main()
