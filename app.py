"""
Campus Navigator: Hugging Face Spaces Entry Point
=================================================
File: app.py

Primary entry point for Hugging Face Spaces web deployment.
Directly invokes the Gradio application defined in web/app.py.
"""

import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from web.app import build_app, KEYBOARD_JS

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Campus Navigator Web App on port {port}...", flush=True)
    demo = build_app()
    try:
        demo.launch(server_name="0.0.0.0", server_port=port, head=KEYBOARD_JS)
    except TypeError:
        demo.launch(server_name="0.0.0.0", server_port=port)
