from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_DIR = ROOT_DIR / "cache" / "history"


@dataclass(frozen=True)
class HistorySession:
    path: Path
    label: str
    updated_at: str


def ensure_history_dir(history_dir: Path | None = None) -> Path:
    path = history_dir or DEFAULT_HISTORY_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, default=str)
        handle.write("\n")


def new_session_path(history_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return history_dir / f"session-{timestamp}-{suffix}.jsonl"


def init_session(path: Path, meta: dict | None = None) -> None:
    payload = {"type": "meta", "created_at": _now_iso()}
    if meta:
        payload.update(meta)
    _append_record(path, payload)


def append_message(
    path: Path,
    role: str,
    content: str,
    reasoning: str | None = None,
    tool_calls: list[dict] | None = None,
) -> None:
    record = {
        "type": "message",
        "timestamp": _now_iso(),
        "role": role,
        "content": content,
        "reasoning": reasoning or "",
        "tool_calls": tool_calls or [],
    }
    _append_record(path, record)


def load_messages(path: Path) -> list[dict]:
    if not path.exists():
        return []
    messages: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") != "message":
            continue
        messages.append(
            {
                "role": record.get("role", "assistant"),
                "content": record.get("content", ""),
                "reasoning": record.get("reasoning", ""),
                "tool_calls": record.get("tool_calls", []) or [],
            }
        )
    return messages


def _session_label(path: Path) -> str:
    first_user: str | None = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") == "message" and record.get("role") == "user":
            first_user = record.get("content", "").strip()
            break
    return first_user or "(no questions yet)"


def list_sessions(history_dir: Path) -> list[HistorySession]:
    if not history_dir.exists():
        return []
    paths = sorted(
        (p for p in history_dir.glob("*.jsonl") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    sessions: list[HistorySession] = []
    for path in paths:
        label = _session_label(path)
        updated_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat(
            timespec="seconds"
        )
        sessions.append(HistorySession(path=path, label=label, updated_at=updated_at))
    return sessions


def delete_session(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file():
        path.unlink()


def cleanup_empty_sessions(history_dir: Path) -> int:
    """Delete sessions that have no user messages.

    Returns the number of sessions deleted.
    """
    if not history_dir.exists():
        return 0
    deleted = 0
    for path in list(history_dir.glob("*.jsonl")):
        if not path.is_file():
            continue
        messages = load_messages(path)
        has_user_message = any(m.get("role") == "user" for m in messages)
        if not has_user_message:
            path.unlink()
            deleted += 1
    return deleted
