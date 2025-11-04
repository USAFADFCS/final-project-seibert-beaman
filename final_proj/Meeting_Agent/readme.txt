Meeting Summarizer (CS471 Final Project)

This program reads a meeting transcript (.txt file) and produces:
- An overall meeting summary
- A list of action items (who, what, when)

Setup Instructions:
1. Run: bash setup.sh
   - This creates a virtual environment and installs dependencies.
2. Add a transcript file to the "input" folder.
3. Run: source .venv/bin/activate
4. Run: python main.py
   - The summarized output will appear in the "output" folder.

Dependencies:
- Python 3.12 or newer
- torch
- transformers
- sentencepiece

Optional:
You can manually activate the environment with:
    source .venv/bin/activate
and deactivate it with:
    deactivate

Cleanup instructions:
1. Run: bash cleanup.sh