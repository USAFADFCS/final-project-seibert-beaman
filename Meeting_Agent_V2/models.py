from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Utterance:
    speaker: str
    text: str

@dataclass
class MeetingContext:
    """Shared state all agents modify."""
    source_path: str
    audio_path: Optional[str] = None

    utterances: List[Utterance] = field(default_factory=list)
    sections: List[List[Utterance]] = field(default_factory=list)
    section_summaries: List[str] = field(default_factory=list)
    global_summary: str = ""
    action_items: str = ""

    meta: Dict[str, Any] = field(default_factory=dict)
