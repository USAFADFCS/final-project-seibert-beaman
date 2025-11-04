#!/bin/bash
# setup.sh - Environment setup script for CS471 Meeting Summarizer

echo "[setup] Creating virtual environment..."
python3 -m venv .venv

echo "[setup] Activating environment..."
source .venv/bin/activate

echo "[setup] Upgrading pip..."
pip install --upgrade pip

echo "[setup] Installing dependencies..."
pip install -r requirements.txt

echo "[setup] Creating input and output folders (if missing)..."
mkdir -p input output

echo "[setup] Pre-caching model weights (this may take a few minutes)..."
python - <<'PYCODE'
from transformers import pipeline
print("[setup] Downloading summarization model...")
_ = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
print("[setup] Model cached successfully.")
PYCODE


echo "BREAK BREAK BREAK"
echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo "[setup] Setup complete!"

echo "First, run: source .venv/bin/activate"
echo "Run the app with: python main.py"
echo "To clear model cache later, run: huggingface-cli delete-cache"
