import os
import re
from typing import List

from models import MeetingContext
from config import OUTPUT_DIR

# Strong commitment phrases like "I'll", "I will", "We'll", "We will"
COMMIT_PATTERN = re.compile(r"\b(i['’]?ll|i will|we['’]?ll|we will)\b", re.IGNORECASE)

# Time/deadline hints
DEADLINE_PATTERN = re.compile(
    r"\b("
    r"monday|tuesday|wednesday|thursday|friday|tomorrow|today|eod|end of day|noon|week|by "
    r")\b",
    re.IGNORECASE,
)

# Words we know are often hallucinated in your use case
HALLUCINATION_SUSPECTS = [
    "github",
    "repository",
    "download",
    "release",
    "end of the month",
    "beta",
    "version",
]


def _get_raw_text(ctx: MeetingContext) -> str:
    return "\n".join(f"{u.speaker}: {u.text}" for u in ctx.utterances)


def _get_action_lines(ctx: MeetingContext) -> List[str]:
    if not ctx.action_items:
        return []
    return [
        line.strip()
        for line in ctx.action_items.splitlines()
        if line.strip().startswith("- ")
    ]


def build_quality_report(ctx: MeetingContext) -> str:
    """
    Build a human-readable quality report for a single MeetingContext.
    This does NOT call any models; it's all rule-based.
    """
    raw_text = _get_raw_text(ctx)
    summary_text = (ctx.global_summary or "").strip()
    actions = _get_action_lines(ctx)

    lines: List[str] = []
    lines.append("MEETING QUALITY CHECK")
    lines.append("---------------------")
    lines.append(f"Source: {ctx.source_path}")
    lines.append(f"Participants: {', '.join(ctx.meta.get('participants', []))}")
    lines.append(f"Utterances: {len(ctx.utterances)}")
    lines.append("")

    # 1. Commitment-like lines in the raw transcript
    commit_like = []
    for u in ctx.utterances:
        if COMMIT_PATTERN.search(u.text.lower()):
            commit_like.append(u)

    lines.append("[Commitment Lines in Transcript]")
    lines.append(f"- Detected commitment-like lines: {len(commit_like)}")
    for u in commit_like:
        lines.append(f"  * {u.speaker}: {u.text}")
    lines.append("")

    # 2. Action items detected
    lines.append("[Action Item Detection]")
    lines.append(f"- Action items in ctx.action_items: {len(actions)}")

    # Rough recall metric
    if commit_like:
        recall = len(actions) / len(commit_like)
        lines.append(f"- Heuristic action recall (actions / commitments): {recall:.2f}")
    else:
        lines.append("- Heuristic action recall: N/A (no commitments found)")
    lines.append("")

    # 3. Weak vs strong actions
    weak_actions = []
    strong_actions = []

    for line in actions:
        lower = line.lower()
        if COMMIT_PATTERN.search(lower):
            strong_actions.append(line)
        else:
            weak_actions.append(line)

    lines.append("[Action Item Quality]")
    lines.append(f"- Strong actions (explicit I'll / I will / we'll): {len(strong_actions)}")
    for s in strong_actions:
        lines.append(f"  * {s}")
    lines.append(f"- Weak actions (no explicit commitment phrase): {len(weak_actions)}")
    for w in weak_actions:
        lines.append(f"  * {w}")
    lines.append("")

    # 4. Hallucination suspects in summary
    lines.append("[Summary Hallucination Check]")

    # simple bag-of-words check
    raw_words = set(w.strip(".,!?()[]").lower() for w in raw_text.split())
    summary_words = set(w.strip(".,!?()[]").lower() for w in summary_text.split())

    # suspicious words = in summary but not in transcript
    diff_words = summary_words - raw_words
    suspects_from_diff = sorted(
        {w for w in diff_words if len(w) > 3 and not w.isnumeric()}
    )

    # overlap with a small list of "known bad" hallucination terms
    suspects_known = sorted(
        {w for w in HALLUCINATION_SUSPECTS if w in summary_text.lower()}
    )

    if not summary_text:
        lines.append("- No summary text present.")
    else:
        if suspects_known:
            lines.append(
                "- Known hallucination suspects found in summary: "
                + ", ".join(suspects_known)
            )
        else:
            lines.append("- No known hallucination keywords found in summary.")

        # Show at most 15 "new" words
        if suspects_from_diff:
            lines.append("- Words present in summary but not in transcript (top 15):")
            for w in suspects_from_diff[:15]:
                lines.append(f"  * {w}")
        else:
            lines.append("- No summary-only words detected (good sign).")

    lines.append("")

    # 5. Simple overall verdict
    lines.append("[Overall Heuristic Verdict]")
    verdict_bits = []

    if commit_like and not actions:
        verdict_bits.append("❌ No action items detected despite commitments.")
    elif commit_like and actions:
        verdict_bits.append("✅ Action items detected for some commitments.")

    if suspects_known:
        verdict_bits.append("⚠️ Summary may contain hallucinated tool/release details.")
    elif summary_text:
        verdict_bits.append("✅ No obvious hallucination keywords in summary.")

    if not verdict_bits:
        verdict_bits.append("ℹ️ Not enough data for a verdict.")

    for v in verdict_bits:
        lines.append(f"- {v}")

    return "\n".join(lines)


def write_quality_report(ctx: MeetingContext):
    """Write the quality report to a separate *_quality.txt file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.basename(ctx.source_path)
    name, _ = os.path.splitext(base)
    out_path = os.path.join(OUTPUT_DIR, f"{name}_quality.txt")

    report = build_quality_report(ctx)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(f"[quality] ✓ Quality report written to {out_path}")
