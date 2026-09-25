import os
import sys
import time
import webbrowser
import threading
import socket

# Ensure Windows terminal doesn't crash on print
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Import the ASGI application from root app.py (works for local and cloud)
from app import app

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def open_browser(port):
    time.sleep(1.5)
    try:
        webbrowser.open(f"http://127.0.0.1:{port}")
    except Exception:
        pass

if __name__ == "__main__":
    # Support dynamic PORT for Render/cloud deployments and default to 8000 locally
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    local_ip = get_local_ip()

    # In cloud environments (Render, Heroku, etc.), do not launch a desktop browser
    is_cloud = "RENDER" in os.environ or "PORT" in os.environ
    if not is_cloud:
        threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    
    import uvicorn
    print("=" * 65)
    print("  VRIXA AI ASSISTANT - CYBER HUD ONLINE")
    print("  HARSH - ROLL NO. 23035004049")
    print("-" * 65)
    print(f"  HOST BINDING: http://{host}:{port}")
    print(f"  LAPTOP / PC LINK : http://127.0.0.1:{port}")
    print(f"  PHONE LINK (Same Wi-Fi): http://{local_ip}:{port}")
    print("=" * 65)
    uvicorn.run(app, host=host, port=port, reload=False)
