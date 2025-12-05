import transformers
from transformers import pipeline

from config import MODEL_NAME, WHISPER_MODEL_NAME

# Silence most HF warnings
transformers.logging.set_verbosity_error()

# Optional: whisper for audio transcription
try:
  import whisper
except ImportError:
  whisper = None


class SummarizationTool:
  """
  Wrapper around a HuggingFace summarization model.
  Runs on CPU for stability and explicitly enforces the model's
  max position embeddings to avoid IndexError.
  """

  def __init__(self, model_name: str = MODEL_NAME):
    print(f"[tool] Loading summarization model: {model_name}")
    # Keep this on CPU for simplicity/stability
    self.pipe = pipeline(
      "summarization",
      model=model_name,
      device="cpu",
    )

    # Make sure tokenizer knows the real max length
    self.tokenizer = self.pipe.tokenizer
    self.model = self.pipe.model

    max_pos = getattr(self.model.config, "max_position_embeddings", None)
    if max_pos is not None:
      # Hugging Face sometimes sets a huge default; we clamp it.
      self.tokenizer.model_max_length = max_pos
      # Also store in init_kwargs so it's honored internally
      if hasattr(self.tokenizer, "init_kwargs"):
        self.tokenizer.init_kwargs["model_max_length"] = max_pos
      print(f"[tool] Set tokenizer.model_max_length = {max_pos}")

  def _safe_call(self, text: str, max_len: int, min_len: int) -> str:
    """
    Helper that repeatedly truncates the input if the model
    still complains about sequence length.
    """
    # Basic char-level clamp to avoid totally insane inputs
    if len(text) > 8000:
      text = text[:8000]

    # Try up to 3 times, each time cutting the text size if needed
    for attempt in range(3):
      try:
        result = self.pipe(
          text,
          max_length=max_len,
          min_length=min_len,
          do_sample=False,
          truncation=True,  # ensure tokenizer truncates to model_max_length
        )
        return result[0]["summary_text"]
      except IndexError as e:
        # Likely still too long; cut the text in half (by words) and retry
        words = text.split()
        if len(words) <= 50:
          # it's already short; no point retrying
          raise e
        half = max(50, len(words) // 2)
        text = " ".join(words[:half])
        print(f"[tool] Warning: IndexError from model, truncating input and retrying (remaining words: {half})")

    # If we somehow still fail, re-raise the last error
    raise RuntimeError("Summarization failed after repeated truncation attempts.")

  def summarize(self, text: str, max_len: int = 150, min_len: int = 40) -> str:
    if not text.strip():
      return ""
    return self._safe_call(text, max_len=max_len, min_len=min_len)

  def summarize_for_actions(
      self,
      transcript_text: str,
      max_len: int = 220,
      min_len: int = 60,
  ) -> str:
    """
    Specialized helper: ask the model ONLY for action items.
    Returns free-form text (usually bullets) that the ActionItemAgent
    will post-process.
    """
    prompt = (
        "<instruction>\n"
        "Extract all concrete action items from the meeting transcript. "
        "Each bullet must include: responsible person if named, the task, and "
        "any deadlines or time references. Use '-' bullets only. "
        "If there are no action items, output: No action items.\n"
        "</instruction>\n\n"
        "<content>\n"
        f"{transcript_text}\n"
        "</content>"
    )
    if not transcript_text.strip():
      return "No action items."

    return self._safe_call(prompt, max_len=max_len, min_len=min_len).strip()


class TranscriptionTool:
  """
  Wrapper around a local Whisper model for audio → text.
  """

  def __init__(self, model_name: str = WHISPER_MODEL_NAME):
    if whisper is None:
      # This will be caught in main.py so text-only still works.
      raise RuntimeError(
        "whisper package is not installed. "
        "Run 'pip install -U openai-whisper'."
      )
    print(f"[tool] Loading transcription model: whisper-{model_name}")
    self.model = whisper.load_model(model_name)

  def transcribe(self, audio_path: str) -> str:
    print(f"[transcription] Transcribing audio: {audio_path}")
    result = self.model.transcribe(audio_path)
    return result.get("text", "").strip()
