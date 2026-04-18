"""
Quick launcher for the Speech Emotion Recognition API.
Usage:
    python run_api.py                        # uses models/ensemble_best.pth
    python run_api.py --model path/to/model  # custom model path
    python run_api.py --port 8080            # custom port
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='models/ensemble_best.pth')
parser.add_argument('--port',  type=int, default=5000)
args = parser.parse_args()

os.environ['MODEL_PATH'] = args.model

from backend.app import app
print(f"\n{'='*55}")
print("  Speech Emotion Recognition API")
print(f"  Model : {args.model}")
print(f"  URL   : http://localhost:{args.port}")
print(f"  Docs  : http://localhost:{args.port}/api/v1/health")
print(f"{'='*55}\n")
app.run(host='0.0.0.0', port=args.port, debug=False)
