"""
log_interactions.py — CSV logger for AgriRAG Q&A interactions.

Every call to `log_interaction()` appends one row to:
    data/interaction_log.csv
Columns: timestamp | question | answer | inference_time_s
"""

import csv
import datetime
from pathlib import Path

LOG_PATH = Path(__file__).parent / "data" / "interaction_log.csv"
_HEADERS = ["timestamp", "question", "answer", "inference_time_s"]


def _ensure_file():
    """Create the CSV file with headers if it does not yet exist."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(_HEADERS)


def log_interaction(
    question: str,
    answer: str,
    inference_time_s: float | None = None,
) -> None:
    """
    Append one interaction row to the CSV log.

    Args:
        question:        The user query sent to the system.
        answer:          The AI-generated answer text.
        inference_time_s: Optional image inference time in seconds (None for text-only queries).
    """
    _ensure_file()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, question, answer, inference_time_s if inference_time_s is not None else ""])
