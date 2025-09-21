import subprocess
import sys
import os

def install_requirements():
    packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy'
    ]

    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def run_dashboard():
    dashboard_path = os.path.join(os.path.dirname(__file__), 'react_dashboard.py')

    print("🚀 Starting Tevron Solar AI Hub Dashboard...")
    print("📱 Dashboard will open in your browser")
    print("🔗 URL: http://localhost:8501")
    print("⚡ Press Ctrl+C to stop the dashboard")

    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', dashboard_path,
            '--server.port=8501',
            '--server.address=localhost',
            '--browser.gatherUsageStats=false'
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

if __name__ == "__main__":
    print("🔧 Installing requirements...")
    install_requirements()
    print("\n" + "="*50)
    run_dashboard()
