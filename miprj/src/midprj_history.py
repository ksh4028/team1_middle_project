from __future__ import annotations

from typing import Any

from midprj_sqlite import SQLiteDB


def truncate_text_by_tokens(text: str, max_tokens: int, tokenizer) -> str:
    if not text or max_tokens <= 0 or tokenizer is None:
        return ""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)


def format_history_messages(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    parts = []
    for item in history:
        role = item.get("role", "user")
        label = "사용자" if role == "user" else "어시스턴트"
        content = item.get("content", "")
        if content:
            parts.append(f"[{label}] {content}")
    return "\n".join(parts)


class HistoryManager:
    def __init__(self, param) -> None:
        self._history: list[dict[str, Any]] = []
        self._max_turns = max(int(getattr(param, "history_max_turns", 0)), 0)
        self._persist = bool(getattr(param, "history_persist", False))
        self._budget_ratio = float(getattr(param, "history_budget_ratio", 0.2))
        self._budget_min_tokens = int(getattr(param, "history_budget_min_tokens", 64))
        self._budget_max_tokens = int(getattr(param, "history_budget_max_tokens", 256))
        if self._budget_min_tokens > self._budget_max_tokens:
            self._budget_min_tokens = self._budget_max_tokens
        self._session_id = (
            str(getattr(param, "history_session_id", "")).strip()
            or f"execute_{param.execute_index}"
        )

        if self._persist:
            db = SQLiteDB()
            db.ensure_chat_history_table()
            limit = self._max_turns * 2 if self._max_turns else 20
            self._history = db.load_recent_chat_history(self._session_id, limit)

    def clear(self) -> None:
        self._history = []

    def append(self, role: str, content: str) -> None:
        if not content:
            return
        self._history.append({"role": role, "content": content})
        if self._max_turns:
            max_items = self._max_turns * 2
            if len(self._history) > max_items:
                self._history = self._history[-max_items:]
        if self._persist:
            db = SQLiteDB()
            db.ensure_chat_history_table()
            db.save_chat_history(self._session_id, role, content)

    def format_block(self) -> str:
        history_text = format_history_messages(self._history)
        if not history_text:
            return ""
        return f"History:\n{history_text}\n\n"

    def build_budgeted_block(self, tokenizer, max_available_tokens: int) -> str:
        if max_available_tokens <= 0 or not self._history:
            return ""
        history_block = self.format_block()
        if not history_block:
            return ""
        if tokenizer is None:
            return history_block

        budget = int(max_available_tokens * self._budget_ratio)
        budget = min(budget, self._budget_max_tokens, max_available_tokens)
        budget = max(budget, min(self._budget_min_tokens, max_available_tokens))
        return truncate_text_by_tokens(history_block, budget, tokenizer)
