import subprocess
import sys
import os
from pathlib import Path

def main():
    print("⚡" * 30)
    print("⚡ SOLARVISION PRO - PROFESSIONAL EDITION ⚡")
    print("⚡" * 30)
    
    dashboard_file = Path(__file__).parent / 'professional_dashboard.py'
    
    if not dashboard_file.exists():
        print("❌ Dashboard file not found")
        sys.exit(1)
    
    # Get local IP
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"
    
    port = 8501
    
    print(f"🚀 Launching Professional Solar Dashboard")
    print(f"   📍 Local: http://localhost:{port}")
    print(f"   🌐 Network: http://{local_ip}:{port}")
    print("   💼 Professional React-like design")
    print("   📊 Complete analytics suite")
    print("   🔧 Full maintenance tracking")
    print("   📈 Advanced reporting system")
    print("   📱 Fully responsive layout")
    print("⌨️  Press Ctrl+C to stop")
    print("=" * 70)
    
    cmd = [
        'streamlit', 'run', str(dashboard_file),
        '--server.port', str(port),
        '--server.address', '0.0.0.0'
    ]
    
    # Open browser
    import webbrowser
    import threading
    import time
    
    def open_browser():
        time.sleep(3)
        webbrowser.open(f'http://localhost:{port}')
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Professional Dashboard stopped")
        print("👋 Thank you for using SolarVision Pro!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()