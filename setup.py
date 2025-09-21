
import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_requirements():
    try:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[SUCCESS] Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install requirements: {e}")
        return False

def create_directories():
    try:
        print("Creating directories...")
        directories = ['models', 'logs', 'analysis']

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"   Created: {directory}/")

        print("[SUCCESS] Directories created successfully!")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create directories: {e}")
        return False

def generate_data():
    try:
        print("Generating historical data...")
        subprocess.check_call([sys.executable, "generate_data.py"])
        print("[SUCCESS] Historical data generated successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to generate data: {e}")
        return False

def train_models():
    try:
        print("Training ML models...")
        subprocess.check_call([sys.executable, "src/train.py"])
        print("[SUCCESS] Models trained successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to train models: {e}")
        return False

def run_tests():
    try:
        print("Running system tests...")
        subprocess.check_call([sys.executable, "test_installation.py"])
        print("[SUCCESS] All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Tests failed: {e}")
        return False

def main():
    print("Solar Panel ML Application Setup")
    print("=" * 50)

    steps = [
        ("Installing Requirements", install_requirements),
        ("Creating Directories", create_directories),
        ("Generating Data", generate_data),
        ("Training Models", train_models),
        ("Running Tests", run_tests)
    ]

    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        success = step_func()

        if not success:
            print(f"\n[ERROR] Setup failed at step: {step_name}")
            print("Please check the error messages above and try again.")
            return False

    print("\n" + "=" * 50)
    print("[SUCCESS] SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 50)

    print("\nYour Solar Panel ML Application is ready to use!")
    print("\nNext steps:")
    print("1. Configure your API keys in config.ini")
    print("2. Set up Firebase credentials in serviceAccountKey.json")
    print("3. Start the dashboard with: streamlit run src/dashboard.py")
    print("\nFor analysis, you can run:")
    print("- python analysis/01_data_profiling.py")
    print("- python analysis/02_visualizations.py")
    print("- python analysis/03_correlation_analysis.py")

    return True

if __name__ == "__main__":
    main()
