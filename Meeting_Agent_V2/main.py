import os

from tools import SummarizationTool, TranscriptionTool
from agents import (
  TranscriptionAgent,
  TranscriptIngestionAgent,
  SectioningAgent,
  SectionSummaryAgent,
  GlobalSummaryAgent,
  ActionItemAgent,
  MetadataAgent,
  PlannerAgent,
  Orchestrator,
  write_output,
)
from models import MeetingContext
from config import INPUT_DIR, AUDIO_EXTENSIONS


def main():
  if not os.path.isdir(INPUT_DIR):
    raise SystemExit(f"Input directory '{INPUT_DIR}' does not exist.")

  # Collect both .txt and audio files
  files = []
  for fn in os.listdir(INPUT_DIR):
    path = os.path.join(INPUT_DIR, fn)
    if not os.path.isfile(path):
      continue
    _, ext = os.path.splitext(fn.lower())
    if ext == ".txt" or ext in AUDIO_EXTENSIONS:
      files.append(path)

  if not files:
    print("[main] No .txt or audio files found in 'input/'.")
    return

  # Core tools
  summary_tool = SummarizationTool()

  # Try to enable transcription; fall back gracefully if whisper not installed
  transcription_agent = None
  try:
    ttool = TranscriptionTool()
    transcription_agent = TranscriptionAgent(ttool)
  except RuntimeError as e:
    print(f"[main] Transcription disabled: {e}")

  # Agents
  ingestion = TranscriptIngestionAgent()
  sectioning = SectioningAgent(max_utterances_per_section=40)
  section_summary = SectionSummaryAgent(summary_tool)
  global_summary = GlobalSummaryAgent(summary_tool)
  action_agent = ActionItemAgent(summary_tool)   # pass tool in here
  metadata_agent = MetadataAgent()
  planner = PlannerAgent()

  orchestrator = Orchestrator(
    ingestion=ingestion,
    sectioning=sectioning,
    section_summary=section_summary,
    global_summary=global_summary,
    action_agent=action_agent,
    planner=planner,
    transcription=transcription_agent,
    metadata=metadata_agent,
  )

  for path in files:
    print("\n====================================")
    print(f"[main] Processing: {path}")
    print("====================================")
    ctx = MeetingContext(source_path=path)
    ctx = orchestrator.run(ctx)
    write_output(ctx)


if __name__ == "__main__":
  main()
