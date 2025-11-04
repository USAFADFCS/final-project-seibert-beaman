import os
import re
from dataclasses import dataclass
from typing import List

from transformers import pipeline

# -----------------------------
# CONFIG
# -----------------------------

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"  # Hugging Face summarization model
INPUT_DIR = "input"
OUTPUT_DIR = "output"


# -----------------------------
# DATA STRUCTURE
# -----------------------------

@dataclass
class Utterance:
    speaker: str
    text: str


# -----------------------------
# TOOL: SUMMARIZATION
# -----------------------------

class SummarizationTool:
    """
    Thin wrapper around a Hugging Face summarization pipeline.
    This is the 'tool' that our agents will use.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"[tool] Loading summarization model: {model_name}")
        self.pipe = pipeline("summarization", model=model_name)

    def summarize(self, text: str, max_len: int = 150, min_len: int = 40) -> str:
        # For tonight: let transformers handle truncation if the text is long.
        result = self.pipe(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
        )
        return result[0]["summary_text"]


# -----------------------------
# AGENT 1: TRANSCRIPT INGESTION
# -----------------------------

class TranscriptIngestionAgent:
    """
    Reads a .txt file and parses it into a list of Utterance objects.

    Expected line formats:
      Speaker 1: Hello everyone...
      Speaker 2: I wanted to talk about the deadline...
    If a line doesn't match this pattern, it's assigned to 'Unknown'.
    """

    SPEAKER_LINE = re.compile(r"^\s*(?P<speaker>[^:]+):\s*(?P<text>.+)$")

    def run(self, path: str) -> List[Utterance]:
        print(f"[ingest] Reading transcript from {path}")
        utterances: List[Utterance] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                m = self.SPEAKER_LINE.match(line)
                if m:
                    speaker = m.group("speaker").strip()
                    text = m.group("text").strip()
                else:
                    speaker = "Unknown"
                    text = line

                utterances.append(Utterance(speaker=speaker, text=text))

        print(f"[ingest] Parsed {len(utterances)} utterances")
        return utterances


# -----------------------------
# AGENT 2: GLOBAL SUMMARY
# -----------------------------

class GlobalSummaryAgent:
    """
    Produces an overall summary of the meeting using the summarization tool.
    """

    def __init__(self, tool: SummarizationTool):
        self.tool = tool

    def run(self, utterances: List[Utterance]) -> str:
        print("[Global] Generating overall summary...")

        full_text = "\n".join(f"{u.speaker}: {u.text}" for u in utterances)

        return self.tool.summarize(full_text, max_len= 220, min_len= 80)
    


# -----------------------------
# AGENT 3: ACTION ITEM EXTRACTOR
# -----------------------------

class ActionItemAgent:
    """
    Extracts action items (who / what / when) from the meeting.

    This version uses simple heuristics instead of the summarization tool:
    - Looks for commitment keywords (will, I'll, by, tomorrow, next week, etc.)
    - Attaches bare "I'll ..." lines to the last known speaker
    - Skips question-style lines like "Can you get them fixed by Thursday?"
    - Formats results as bullet points.
    """

    def __init__(self, tool: SummarizationTool | None = None):
        # Tool kept for compatibility with existing code, but not used here.
        self.tool = tool

    def run(self, utterances: List[Utterance]) -> str:
        action_lines: List[str] = []
        last_speaker: str | None = None

        # Keywords that usually indicate an action/commitment
        commitment_keywords = [
            "i'll",
            "i will",
            "we will",
            "we'll",
            "will ",
            "going to",
            "plan to",
            "need to",
            "have to",
            "should",
            "must",
            "by ",
            "tomorrow",
            "next week",
            "next monday",
            "next tuesday",
            "next wednesday",
            "next thursday",
            "next friday",
            "deadline",
        ]

        # Patterns that look like questions we *don't* want as action items
        question_start = re.compile(r"(?i)^\s*(can|could|will|should)\s+you\b")

        for u in utterances:
            speaker = (u.speaker or "").strip() or "Unknown"
            text = (u.text or "").strip()
            lower = text.lower()

            # Track last known speaker so "I'll..." lines can inherit it
            if speaker != "Unknown":
                last_speaker = speaker

            # Skip empty lines
            if not text:
                continue

            # Skip clear questions like "Can you get this done by Thursday?"
            if question_start.match(text):
                continue

            # If the line starts with "I'll" and we have a last speaker, use that
            if text.lower().startswith("i'll") and last_speaker is not None:
                speaker = last_speaker

            # Check if this line looks like an action/commitment
            if any(kw in lower for kw in commitment_keywords):
                # Basic cleanup
                text_clean = re.sub(r'[“”"]+', "", text).strip()

                # Ensure it ends with a period for consistency
                if not text_clean.endswith((".", "!", "?")):
                    text_clean += "."

                action_lines.append(f"- {speaker}: {text_clean}")

        if not action_lines:
            return "No action items found."

        # Neaten spacing inside the whole block
        cleaned = "\n".join(action_lines)
        cleaned = re.sub(r"\s+", " ", cleaned)           # collapse extra spaces
        cleaned = re.sub(r"\s+\.", ".", cleaned)         # no space before periods
        cleaned = re.sub(r"\s+\n", "\n", cleaned).strip()

        # Re-split into lines to preserve bullets after space normalization
        final_lines = [line for line in cleaned.split("\n") if line.strip()]
        return "\n".join(final_lines)


# -----------------------------
# COORDINATOR / MAIN PIPELINE
# -----------------------------

def process_file(path: str, tool: SummarizationTool):
    ingestion_agent = TranscriptIngestionAgent()
    summary_agent = GlobalSummaryAgent(tool)
    action_agent = ActionItemAgent(tool)

    utterances = ingestion_agent.run(path)
    overall_summary = summary_agent.run(utterances)
    action_items = action_agent.run(utterances)

    action_items = "\n".join(f"- {line.strip()}" for line in action_items.split("-") if line.strip())
    


    # write output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    out_path = os.path.join(OUTPUT_DIR, f"{name}_summary.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        # f.write(f"Meeting Summary for {name}\n\n")
        f.write("--------------------------------------------------\n")
        f.write(" MEETING SUMMARY REPORT \n")
        f.write("--------------------------------------------------\n\n")


        f.write("=== Overall Summary ===\n")
        f.write(overall_summary.strip() + "\n\n")

        f.write("=== Action Items ===\n")
        f.write(action_items.strip() + "\n")

    for _ in range(10):
        print()
    
    print(f"[main] ✅ Wrote summary to {out_path}")
    print("If you are done, feel free to run: deactviate")
    print("followed by running: bash cleanup.sh")


def main():
    if not os.path.isdir(INPUT_DIR):
        raise SystemExit(
            f"Input directory '{INPUT_DIR}' does not exist. "
            f"Create it and add .txt files."
        )

    txt_files = [
        os.path.join(INPUT_DIR, fn)
        for fn in os.listdir(INPUT_DIR)
        if fn.lower().endswith(".txt")
    ]

    if not txt_files:
        print(f"[main] No .txt files found in '{INPUT_DIR}'. Add a transcript and rerun.")
        return

    tool = SummarizationTool()

    for path in txt_files:
        process_file(path, tool)


if __name__ == "__main__":
    main()
