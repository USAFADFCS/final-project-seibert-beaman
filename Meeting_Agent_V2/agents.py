import os
import re
from typing import List, Optional

from models import Utterance, MeetingContext
from tools import SummarizationTool, TranscriptionTool
from config import OUTPUT_DIR, AUDIO_EXTENSIONS


# -----------------------------
# // BASE AGENT
# -----------------------------

class BaseAgent:
  def __init__(self, name: str):
    self.name = name

  def run(self, ctx: MeetingContext) -> MeetingContext:
    raise NotImplementedError


# -----------------------------
# // TRANSCRIPTION AGENT (A)
# -----------------------------

class TranscriptionAgent(BaseAgent):
  """
  Uses TranscriptionTool (whisper) to convert audio → utterances.
  """

  def __init__(self, tool: TranscriptionTool):
    super().__init__("TranscriptionAgent")
    self.tool = tool

  def run(self, ctx: MeetingContext) -> MeetingContext:
    if not ctx.source_path:
      return ctx

    _, ext = os.path.splitext(ctx.source_path.lower())
    if ext not in AUDIO_EXTENSIONS:
      return ctx  # nothing to do

    text = self.tool.transcribe(ctx.source_path)
    if not text:
      print("[transcription] Warning: empty transcript from audio")
      return ctx

    # Simple split into pseudo-utterances by line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
      lines = [text.strip()]

    ctx.utterances = [Utterance("Unknown", line) for line in lines]
    ctx.audio_path = ctx.source_path
    ctx.meta["source_type"] = "audio"
    ctx.meta["num_transcript_lines"] = len(lines)
    return ctx


# -----------------------------
# // TRANSCRIPT INGESTION AGENT
# -----------------------------

class TranscriptIngestionAgent(BaseAgent):

  SPEAKER_LINE = re.compile(r"^\s*(?P<speaker>[^:]+):\s*(?P<text>.+)$")

  def __init__(self):
    super().__init__("TranscriptIngestionAgent")

  def run(self, ctx: MeetingContext) -> MeetingContext:
    path = ctx.source_path
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

        utterances.append(Utterance(speaker, text))

    print(f"[ingest] Parsed {len(utterances)} utterances")
    ctx.utterances = utterances
    ctx.meta["source_type"] = ctx.meta.get("source_type", "text")
    return ctx


# -----------------------------
# // SECTIONING AGENT
# -----------------------------

class SectioningAgent(BaseAgent):
  def __init__(self, max_utterances_per_section: int = 40):
    super().__init__("SectioningAgent")
    self.max_utterances = max_utterances_per_section

  def run(self, ctx: MeetingContext) -> MeetingContext:
    print("[section] Splitting transcript into sections...")
    sections = []
    cur = []

    for u in ctx.utterances:
      cur.append(u)
      if len(cur) >= self.max_utterances:
        sections.append(cur)
        cur = []

    if cur:
      sections.append(cur)

    print(f"[section] Created {len(sections)} section(s)")
    ctx.sections = sections
    return ctx


# -----------------------------
# // SECTION SUMMARY AGENT (2A + 2B)
# -----------------------------

class SectionSummaryAgent(BaseAgent):
  """
  Stage 1: For each section, produce 3–5 bullet points capturing
  topics, decisions, and explicit next steps.
  """

  def __init__(self, tool: SummarizationTool):
    super().__init__("SectionSummaryAgent")
    self.tool = tool

  @staticmethod
  def _lengths_for_text(text: str, max_cap: int = 180):
    words = len(text.split())
    max_len = min(max_cap, max(40, int(words * 0.7)))  # slightly verbose bullets
    min_len = max(20, min(int(words * 0.3), max_len - 5))
    return max_len, min_len

  def run(self, ctx: MeetingContext) -> MeetingContext:
    print("[SectionSummary] Summarizing each section into bullet points...")

    if not ctx.sections:
      ctx.sections = [ctx.utterances]

    ctx.section_summaries = []

    for i, section in enumerate(ctx.sections, start=1):
      print(f"[SectionSummary] Section {i}/{len(ctx.sections)}")

      # Just the raw meeting content
      raw_text = "\n".join(f"{u.speaker}: {u.text}" for u in section)

      # No instructions in the input – treat model as a compressor
      full_text = raw_text

      max_len, min_len = self._lengths_for_text(full_text, max_cap=180)
      bullets = self.tool.summarize(full_text, max_len=max_len, min_len=min_len)

      # Optionally enforce bullet formatting ourselves:
      # turn sentences into "- ..." lines
      sentences = [s.strip() for s in bullets.split(".") if s.strip()]
      bulletified = "\n".join(f"- {s}." for s in sentences[:5])
      ctx.section_summaries.append(bulletified.strip())

    return ctx


# -----------------------------
# // GLOBAL SUMMARY AGENT (2A + 2B)
# -----------------------------

class GlobalSummaryAgent(BaseAgent):
  """
  Stage 2: Take all section bullets and generate a short executive
  summary in narrative or bullet form.

  For reliability, we keep this extractive: we just combine section
  summaries instead of asking the model again (which can hallucinate).
  """

  def __init__(self, tool: SummarizationTool):
    super().__init__("GlobalSummaryAgent")
    self.tool = tool  # kept for API compatibility, not used now

  def run(self, ctx: MeetingContext) -> MeetingContext:
    print("[global] Generating global narrative summary (extractive)...")

    if ctx.section_summaries:
      # Just join the section bullets into one block
      ctx.global_summary = "\n\n".join(ctx.section_summaries)
    else:
      # Fallback: take the first ~10 utterances as a coarse summary
      lines = []
      for u in ctx.utterances[:10]:
        lines.append(f"- {u.speaker}: {u.text}")
      ctx.global_summary = "\n".join(lines)

    return ctx



# -----------------------------
# // ACTION ITEM AGENT (2C + 6.3)
# -----------------------------

class ActionItemAgent(BaseAgent):
  """
  Hybrid approach:
    1) Heuristic pass over utterances to find explicit commitments
    2) Optional model-based pass using summarize_for_actions()
    3) Merge + dedupe
  """

  def __init__(self, tool: Optional[SummarizationTool] = None):
    super().__init__("ActionItemAgent")
    self.tool = tool

    # Regex for strong first-person commitments
    self.commit_pattern = re.compile(
      r"\b(i['’]?ll|i will|we['’]?ll|we will)\b",
      re.IGNORECASE,
    )

    # Time/deadline hints (just for nicer recall, not required)
    self.deadline_keywords = [
      "by ", "tomorrow", "next week",
      "next monday", "next tuesday", "next wednesday",
      "next thursday", "next friday", "monday", "tuesday",
      "wednesday", "thursday", "friday",
      "eod", "end of day", "noon", "deadline",
    ]

    # Questions like "Can you... / Should you..." (requests)
    self.question_start = re.compile(r"(?i)^\s*(can|could|will|would|should)\s+you\b")

  def _heuristic_actions(self, ctx: MeetingContext) -> List[str]:
    action_lines: List[str] = []
    last_speaker: Optional[str] = None

    for u in ctx.utterances:
      speaker = (u.speaker or "Unknown").strip() or "Unknown"
      text = (u.text or "").strip()
      lower = text.lower()

      if speaker != "Unknown":
        last_speaker = speaker

      if not text:
        continue

      # Skip generic offers like "I can ..."
      if lower.strip().startswith("i can "):
        continue

      # Detect explicit self-commitment: I'll / I will / We'll / We will
      m = self.commit_pattern.search(lower)
      if not m:
        # Optionally treat some requests as actions if they have deadlines
        if self.question_start.match(text) and any(kw in lower for kw in self.deadline_keywords):
          # This is "Can you ... by Friday?" style; treat as request
          cleaned = re.sub(r'[“”"]+', "", text).strip()
          if not cleaned.endswith((".", "!", "?")):
            cleaned += "."
          action_lines.append(f"- {speaker}: (request) {cleaned}")
        continue

      # If the line starts with "I'll ..." but speaker is Unknown,
      # inherit the last named speaker
      if lower.startswith("i'll") and last_speaker:
        speaker = last_speaker

      # Shorten to just the committed part, starting from the match
      commit_start = m.start()
      snippet = text[commit_start:].strip()

      # Clean up quotes and ensure punctuation
      cleaned = re.sub(r'[“”"]+', "", snippet).strip()
      if not cleaned.endswith((".", "!", "?")):
        cleaned += "."

      action_lines.append(f"- {speaker}: {cleaned}")

    return action_lines

  def _model_actions(self, ctx: MeetingContext) -> List[str]:
    """
    Optional: let the summarization model propose action items.
    We keep this as a secondary signal because it can hallucinate.
    """
    if self.tool is None:
      return []

    full_text = "\n".join(f"{u.speaker}: {u.text}" for u in ctx.utterances)
    if not full_text.strip():
      return []

    ai_raw = self.tool.summarize_for_actions(full_text)
    lines = []
    for line in ai_raw.splitlines():
      l = line.strip()
      if not l:
        continue

      # normalize bullet prefix
      if l.startswith(("-", "*")):
        l = l[1:].strip()
      # skip generic "No action items"
      if l.lower().startswith("no action items"):
        continue

      lines.append(f"- {l}")

    return lines

  def run(self, ctx: MeetingContext) -> MeetingContext:
    heuristic = self._heuristic_actions(ctx)
    model_based = self._model_actions(ctx) if self.tool is not None else []

    merged = sorted(set(heuristic + model_based))

    if not merged:
      ctx.action_items = "No action items found."
    else:
      ctx.action_items = "\n".join(merged)

    return ctx


# -----------------------------
# // METADATA AGENT (F)
# -----------------------------

class MetadataAgent(BaseAgent):
  """
  Adds simple meeting metadata (participants, utterance count, source type).
  """

  def __init__(self):
    super().__init__("MetadataAgent")

  def run(self, ctx: MeetingContext) -> MeetingContext:
    participants = sorted(
      {u.speaker for u in ctx.utterances if u.speaker and u.speaker != "Unknown"}
    )
    ctx.meta.setdefault("participants", participants)
    ctx.meta["num_utterances"] = len(ctx.utterances)
    ctx.meta["source_type"] = ctx.meta.get("source_type", "text")
    return ctx


# -----------------------------
# // PLANNER + ORCHESTRATOR
# -----------------------------

class PlannerAgent:
  """
  Chooses a plan based on file type (.txt vs audio).
  """

  def plan(self, ctx: MeetingContext) -> List[str]:
    _, ext = os.path.splitext(ctx.source_path.lower())
    if ext in AUDIO_EXTENSIONS:
      return [
        "transcribe",
        "section",
        "section_summary",
        "global_summary",
        "actions",
        "meta",
      ]
    else:
      return [
        "ingest",
        "section",
        "section_summary",
        "global_summary",
        "actions",
        "meta",
      ]


class Orchestrator:
  def __init__(
      self,
      ingestion: TranscriptIngestionAgent,
      sectioning: SectioningAgent,
      section_summary: SectionSummaryAgent,
      global_summary: GlobalSummaryAgent,
      action_agent: ActionItemAgent,
      planner: PlannerAgent,
      transcription: Optional[TranscriptionAgent] = None,
      metadata: Optional[MetadataAgent] = None,
  ):
    self.ingestion = ingestion
    self.sectioning = sectioning
    self.section_summary = section_summary
    self.global_summary = global_summary
    self.action_agent = action_agent
    self.planner = planner
    self.transcription = transcription
    self.metadata = metadata

  def run(self, ctx: MeetingContext) -> MeetingContext:
    plan = self.planner.plan(ctx)
    print(f"[planner] Plan: {plan}")

    for step in plan:
      print(f"[orchestrator] Running step: {step}")
      if step == "ingest":
        ctx = self.ingestion.run(ctx)
      elif step == "transcribe":
        if self.transcription is None:
          print("[orchestrator] No transcription agent configured; skipping.")
        else:
          ctx = self.transcription.run(ctx)
      elif step == "section":
        ctx = self.sectioning.run(ctx)
      elif step == "section_summary":
        ctx = self.section_summary.run(ctx)
      elif step == "global_summary":
        ctx = self.global_summary.run(ctx)
      elif step == "actions":
        ctx = self.action_agent.run(ctx)
      elif step == "meta":
        if self.metadata:
          ctx = self.metadata.run(ctx)
    return ctx


# -----------------------------
# // OUTPUT UTILITY
# -----------------------------

def write_output(ctx: MeetingContext):
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  base = os.path.basename(ctx.source_path)
  name, _ = os.path.splitext(base)
  out_path = os.path.join(OUTPUT_DIR, f"{name}_summary.txt")

  with open(out_path, "w", encoding="utf-8") as f:
    f.write("--------------------------------------------------\n")
    f.write(" MEETING SUMMARY REPORT \n")
    f.write("--------------------------------------------------\n\n")

    if ctx.meta:
      f.write("=== Metadata ===\n")
      for key, value in ctx.meta.items():
        f.write(f"- {key}: {ctx.meta[key]}\n")
      f.write("\n")

    f.write("=== Overall Summary ===\n")
    f.write(ctx.global_summary.strip() + "\n\n")

    f.write("=== Action Items ===\n")
    f.write(ctx.action_items.strip() + "\n")

  print(f"[main] ✓ Output written to {out_path}")
