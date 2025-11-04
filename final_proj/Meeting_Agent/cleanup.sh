#!/bin/bash
# cleanup.sh - Safely remove environment and cached model files

echo "[cleanup] Deactivating virtual environment if active..."
deactivate 2>/dev/null || true

echo "[cleanup] Removing virtual environment..."
rm -rf .venv

echo "[cleanup] Clearing Hugging Face model cache..."
rm -rf ~/.cache/huggingface

echo "[cleanup] Clearing PyTorch cache..."
rm -rf ~/.cache/torch

echo "[cleanup] Removing __pycache__ folders..."
find . -type d -name "__pycache__" -exec rm -rf {} +

echo "[cleanup] Done!"
echo "Your project directory has been reset to a clean state."
