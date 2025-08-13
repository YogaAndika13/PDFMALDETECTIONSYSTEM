import os
# Disable oneDNN optimizations before any other imports, especially tensorflow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from app import app
from waitress import serve

if __name__ == "__main__":
    # Check if we're in development mode
    debug_mode = os.environ.get('FLASK_ENV') == 'development' or os.environ.get('DEBUG', '').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '127.0.0.1')
    
    if debug_mode:
        print("RUN.PY: Starting with Flask's built-in development server for debugging...")
        print(f"⚠️  WARNING: Debug mode is enabled. Do NOT use this in production!")
        app.run(host=host, port=port, debug=True)
    else:
        print("RUN.PY: Starting with Waitress production server...")
        print(f"🚀 Server starting on http://{host}:{port}")
        # Use Waitress for production (as per your memory preference for Windows)
        serve(app, host=host, port=port, threads=4)
