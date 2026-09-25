"""Vrixa AI Assistant - Root ASGI Application Entrypoint.

Provides a clean root entrypoint for uvicorn/gunicorn on Render, Heroku, or any cloud platform:
    uvicorn app:app --host 0.0.0.0 --port $PORT
"""
import os
import sys
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(BASE_DIR, "New folder")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Ensure 'app' in sys.modules points to backend app.py so deployed_router can import it cleanly
legacy_app_path = os.path.join(backend_dir, "app.py")
spec = importlib.util.spec_from_file_location("app", legacy_app_path)
backend_app_mod = importlib.util.module_from_spec(spec)
sys.modules["app"] = backend_app_mod
spec.loader.exec_module(backend_app_mod)

import deployed_router
app = deployed_router.app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
