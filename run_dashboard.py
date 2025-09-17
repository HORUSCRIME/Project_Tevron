import subprocess
import sys
import os
import argparse
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(" Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n Install missing packages with:")
        print("   pip install -r dashboard_requirements.txt")
        return False
    
    print(" All required packages are installed!")
    return True

def check_data_files():
    """Check if data files exist"""
    data_files = [
        'solar_training_data.csv',
        'solar_test_data.csv'
    ]
    
    existing_files = []
    for file in data_files:
        if os.path.exists(file):
            existing_files.append(file)
    
    if existing_files:
        print(f" Found data files: {', '.join(existing_files)}")
        return True
    else:
        print("  No data files found. Dashboard will use sample data.")
        return False

def optimize_for_rpi():
    """Set environment variables for Raspberry Pi optimization"""
    os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '50'
    os.environ['STREAMLIT_SERVER_MAX_MESSAGE_SIZE'] = '50'
    
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    
    print(" Applied Raspberry Pi optimizations")

def main():
    parser = argparse.ArgumentParser(description='Launch Solar Panel Dashboard')
    parser.add_argument('--port', type=int, default=8501, 
                       help='Port to run the dashboard on (default: 8501)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--no-browser', action='store_true',
                       help='Do not open browser automatically')
    
    args = parser.parse_args()
    
    print(" Solar Panel Monitoring Dashboard Launcher")
    print("=" * 50)
    
    if not check_requirements():
        sys.exit(1)
    
    check_data_files()
    
    optimize_for_rpi()
    
    dashboard_file = Path(__file__).parent / 'streamlit_dashboard.py'
    
    if not dashboard_file.exists():
        print(" Dashboard file not found: streamlit_dashboard.py")
        sys.exit(1)
    
    cmd = [
        'streamlit', 'run', str(dashboard_file),
        '--server.port', str(args.port),
        '--server.address', args.host,
        '--server.headless', 'true' if args.no_browser else 'false',
        '--server.fileWatcherType', 'none',  
        '--browser.gatherUsageStats', 'false'
    ]
    
    print(f" Starting dashboard on http://{args.host}:{args.port}")
    print(" Access from other devices using your Pi's IP address")
    print("  Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f" Error running dashboard: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(" Streamlit not found. Install with: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()