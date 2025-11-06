# nulla_source\scripts\nulla_game_snake.py
# Snake overlay for Nulla's chat window — import-safe (no GUI at import time)

from __future__ import annotations

import os
import random
import tkinter as tk
from typing import Optional, Tuple, List, Dict, Any

# ============ Module state ============
_overlay: Optional[tk.Frame] = None
_header: Optional[tk.Frame] = None
_canvas: Optional[tk.Canvas] = None
_confirm_sheet: Optional[tk.Frame] = None

_story_win: Optional[tk.Toplevel] = None
_hooks: Dict[str, Any] = {}
_active: bool = False
_paused: bool = False
_game_over: bool = False

_input_prev_state: Optional[str] = None
_tick_job: Optional[str] = None
_toast_job: Optional[str] = None

# Grid + visuals
GRID_COLS = 28
GRID_ROWS = 18
CELL = 24
PADDING = 8

# Colors (dark theme match)
BG = "#0f1115"
PANEL = "#161a20"
BORDER = "#2a2f3a"
FG = "#e6e6e6"
FG_DIM = "#9aa4b2"
ACCENT = "#3b82f6"
APPLE = "#ef4444"  # red
SNAKE = "#22c55e"
SNAKE_HEAD = "#86efac"
PAUSE_DIM = "#000000"
PAUSE_STIPPLE = "gray50"

# Snake state
_snake: List[Tuple[int, int]] = []
_dir: Tuple[int, int] = (1, 0)
_pending_dir: Tuple[int, int] = (1, 0)
_food: Tuple[int, int] = (10, 10)
_score: int = 0
_speed_ms: int = 140

# =========================================================
# AI QUIPS via local OpenAI-compatible endpoint (e.g., llama.cpp server)
# =========================================================

def _strip_outer_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1].strip()
    return s

class _GameAI:
    """
    Tiny client for an OpenAI-compatible chat endpoint.
    Uses NULLA_API_BASE / NULLA_MODEL if set; otherwise defaults to 127.0.0.1:1234
    """
    def __init__(self):
        self.base = (os.getenv("NULLA_API_BASE", "http://127.0.0.1:1234")).rstrip("/")
        self.model = os.getenv("NULLA_MODEL", "openhermes-2.5-mistral-7b")
        try:
            import requests  # type: ignore
            self._requests = requests
        except Exception:
            self._requests = None

        # friendly fallback pools if request fails
        self.fallback: Dict[str, List[str]] = {
            "load": [
                "Snake ready! WASD/Arrows to move. Press P to pause.",
                "All set. Use WASD or arrows; P to pause.",
            ],
            "pause": [
                "Paused. Stretch your fingers.",
                "Taking a tiny break. Press P to resume.",
            ],
            "resume": [
                "Back in motion—good luck!",
                "Unpaused. Let’s slither.",
            ],
            "milestone": [
                "Nice! Score {score}.",
                "Clean play—{score} points.",
            ],
            "restart": [
                "Restarting Snake…",
                "Fresh board—try again!",
            ],
            "exit": [
                "Snake closed.",
                "Closed Snake overlay.",
            ],
            "game_over": [
                "Oof—crashed. Final score: {score}.",
                "GG—run over. Final score: {score}.",
            ],
        }

    def _sys_prompt(self, kind: str, ctx: Dict[str, Any]) -> str:
        base = (
            "You are Nulla: warm, calm, supportive, friendly, and kind.\n"
            "Write EXACTLY ONE short line (<= 120 chars). No emojis. Never mention AI or prompts.\n"
            "Keep it wholesome and encouraging. Vary the wording naturally each time.\n"
        )
        if kind == "load":
            base += "Event: game_load. Briefly state controls: 'WASD/Arrows to move. P to pause.'\n"
        elif kind == "pause":
            base += "Event: pause. Say the game is paused and how to resume (press P).\n"
        elif kind == "resume":
            base += "Event: resume. Say we're back and encourage the player.\n"
        elif kind == "milestone":
            base += "Event: milestone. Congratulate and include the current score.\n"
        elif kind == "restart":
            base += "Event: restart. Say the game is restarting in a friendly way.\n"
        elif kind == "exit":
            base += "Event: exit. Confirm the Snake game overlay closed. Do NOT say goodbye/farewell or imply leaving the app.\n"
        elif kind == "game_over":
            base += (
                "Event: game_over. Encourage the player. You MUST end the sentence with exactly: "
                f"'Final score: {ctx.get('score', 0)}.'\n"
            )
        return base

    def _user_prompt(self, kind: str, ctx: Dict[str, Any]) -> str:
        parts = [f"Kind={kind}"]
        if "score" in ctx:
            parts.append(f"Score={ctx['score']}")
        if "reason" in ctx and ctx["reason"]:
            parts.append(f"Reason={ctx['reason']}")
        return " | ".join(parts)

    def gen(self, kind: str, **ctx) -> str:
        if not self._requests:
            return random.choice(self.fallback.get(kind, ["Okay."])).format(**ctx)

        sys_prompt = self._sys_prompt(kind, ctx)
        user_prompt = self._user_prompt(kind, ctx)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.9,
            "top_p": 0.95,
            "max_tokens": 60,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}

        try:
            r = self._requests.post(f"{self.base}/v1/chat/completions",
                                    json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            text = (data["choices"][0]["message"]["content"] or "").strip()
            line = _strip_outer_quotes(text.split("\n")[0])[:200]
            if kind == "game_over":
                tail = f"Final score: {ctx.get('score', 0)}."
                if not line.endswith(tail):
                    if "Final score:" in line:
                        idx = line.rfind("Final score:")
                        line = line[:idx].rstrip() + " " + tail
                    else:
                        if not line.endswith("."):
                            line += "."
                        line += " " + tail
            return line or random.choice(self.fallback.get(kind, ["Okay."])).format(**ctx)
        except Exception:
            return random.choice(self.fallback.get(kind, ["Okay."])).format(**ctx)

_AI = _GameAI()

def _announce(msg: str, speak: bool = False):
    """
    Send raw text to the host; nulla_window adds the single
    'Nulla (game):' prefix and newline.
    """
    try:
        if callable(_hooks.get("story_push")):
            _hooks["story_push"](msg)  # raw, no prefix/newline
        if speak and callable(_hooks.get("enqueue_sentence_if_ready")):
            _hooks["enqueue_sentence_if_ready"](msg)
        if callable(_hooks.get("mark_nulla_spoke")):
            _hooks["mark_nulla_spoke"]()
        if callable(_hooks.get("idle_touch")):
            _hooks["idle_touch"]()
    except Exception:
        pass

def _say_ai(kind: str, speak: bool = False, **kw):
    line = _AI.gen(kind, **kw)
    _announce(line, speak=speak)

# =========================================================

def mount_overlay(story_win: tk.Toplevel, hooks: Dict[str, Any]) -> None:
    """Required entrypoint. Creates the overlay and starts the game loop."""
    global _overlay, _header, _canvas, _story_win, _hooks, _active, _paused, _game_over

    if _active and _overlay and _overlay.winfo_exists():
        _overlay.lift()
        return

    _story_win = story_win
    _hooks = hooks or {}

    # pause idle/short-idle while the game is active
    try:
        if callable(_hooks.get("idle_block_push")):
            _hooks["idle_block_push"]()
    except Exception:
        pass

    _disable_chat_input()

    _overlay = tk.Frame(_story_win, bg=BG, highlightthickness=1, highlightbackground=BORDER)
    _overlay.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)
    _overlay.lift()

    _header = tk.Frame(_overlay, bg=PANEL, height=44, highlightthickness=0)
    _header.pack(side="top", fill="x")

    back_btn = tk.Button(
        _header, text="← Back", relief="flat",
        bg=BORDER, fg=FG, activebackground=BORDER, activeforeground=FG,
        command=_on_back
    )
    back_btn.pack(side="left", padx=8, pady=8)

    title = tk.Label(_header, text="Snake", bg=PANEL, fg=FG, font=("Segoe UI", 11, "bold"))
    title.pack(side="left", padx=8)

    score_lbl = tk.Label(_header, text="Score: 0", bg=PANEL, fg=FG_DIM, font=("Consolas", 11))
    score_lbl.pack(side="right", padx=12)
    _header.score_lbl = score_lbl  # type: ignore[attr-defined]

    _canvas = tk.Canvas(_overlay, bg=BG, highlightthickness=0)
    _canvas.pack(side="top", fill="both", expand=True)

    _overlay.bind_all("<Key>", _on_key)
    _overlay.bind("<FocusIn>", lambda e: _overlay.focus_set())
    _overlay.focus_set()
    _overlay.update_idletasks()

    _canvas.bind("<Configure>", _on_canvas_resize)

    _init_game()

    # Defer the spoken controls line so "Launching Snake…" prints first.
    # (GameManager prints that after mount_overlay returns.)
    if _overlay:
        _overlay.after(200, lambda: _say_ai("load", speak=True))

    _active = True
    _paused = False
    _game_over = False
    _resume_loop()

def is_active() -> bool:
    return bool(_active and _overlay and _overlay.winfo_exists())

def exit_game(confirm: bool = True) -> None:
    if confirm:
        _show_confirm_sheet(
            text="Are you sure you want to exit?",
            yes_cb=_teardown,
            no_cb=_hide_confirm_sheet
        )
    else:
        _teardown()

def hide_overlay() -> None:
    _teardown()

def destroy_overlay() -> None:
    _teardown()

# ============ Internal helpers ============

def _get_chat_entry_widget() -> Optional[tk.Entry]:
    """
    Try NullaChatWindow.input_entry first; fall back to .entry (nulla_window.py).
    """
    entry = getattr(_story_win, "input_entry", None)
    if not isinstance(entry, tk.Entry):
        entry = getattr(_story_win, "entry", None)
    return entry if isinstance(entry, tk.Entry) else None

def _disable_chat_input():
    global _input_prev_state
    entry = _get_chat_entry_widget()
    if entry is not None:
        try:
            _input_prev_state = str(entry.cget("state"))
            entry.configure(state="disabled")
        except Exception:
            pass

def _restore_chat_input():
    entry = _get_chat_entry_widget()
    if entry is not None:
        try:
            entry.configure(state=_input_prev_state if _input_prev_state else "normal")
            entry.focus_set()
        except Exception:
            pass

def _toast(msg: str, ms: int = 1800):
    global _toast_job
    if not _canvas:
        return
    _clear_toast()
    w = _canvas.winfo_width()
    _canvas.toaster = _canvas.create_text(  # type: ignore[attr-defined]
        w // 2, 34, text=msg, fill=FG, font=("Segoe UI", 11, "bold")
    )
    def _kill():
        _clear_toast()
    _toast_job = _canvas.after(ms, _kill)

def _clear_toast():
    global _toast_job
    if not _canvas:
        return
    if hasattr(_canvas, "toaster"):
        try:
            _canvas.delete(_canvas.toaster)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            delattr(_canvas, "toaster")
        except Exception:
            pass
    if _toast_job:
        try:
            _canvas.after_cancel(_toast_job)
        except Exception:
            pass
        _toast_job = None

def _on_back():
    _show_confirm_sheet(
        text="Are you sure you want to exit?",
        yes_cb=_teardown,
        no_cb=_hide_confirm_sheet
    )

def _show_confirm_sheet(text: str, yes_cb, no_cb):
    global _confirm_sheet
    if not _overlay:
        return
    if _confirm_sheet and _confirm_sheet.winfo_exists():
        return

    if _canvas:
        w, h = _canvas.winfo_width(), _canvas.winfo_height()
        _canvas.dim = _canvas.create_rectangle(  # type: ignore[attr-defined]
            0, 0, w, h, fill=PAUSE_DIM, outline="", stipple=PAUSE_STIPPLE
        )

    _confirm_sheet = tk.Frame(_overlay, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
    _confirm_sheet.place(relx=0.5, rely=0.5, anchor="center")

    lbl = tk.Label(_confirm_sheet, text=text, bg=PANEL, fg=FG, font=("Segoe UI", 11, "bold"))
    lbl.pack(padx=16, pady=(16, 8))

    btns = tk.Frame(_confirm_sheet, bg=PANEL)
    btns.pack(fill="x", padx=16, pady=(0, 16))

    yes = tk.Button(btns, text="Yes", relief="flat", bg=ACCENT, fg="white",
                    activebackground=ACCENT, activeforeground="white",
                    command=yes_cb)
    yes.pack(side="left", padx=(0, 8))

    no = tk.Button(btns, text="No", relief="flat", bg=BORDER, fg=FG,
                   activebackground=BORDER, activeforeground=FG,
                   command=no_cb)
    no.pack(side="left")

def _hide_confirm_sheet():
    global _confirm_sheet
    if _confirm_sheet and _confirm_sheet.winfo_exists():
        try:
            _confirm_sheet.destroy()
        except Exception:
            pass
    _confirm_sheet = None
    if _canvas and hasattr(_canvas, "dim"):
        try:
            _canvas.delete(_canvas.dim)  # type: ignore[attr-defined]
            delattr(_canvas, "dim")
        except Exception:
            pass

def _teardown():
    # resume idle after leaving the game
    try:
        if callable(_hooks.get("idle_block_pop")):
            _hooks["idle_block_pop"]()
    except Exception:
        pass

    global _active, _paused, _game_over
    _cancel_loop()
    _hide_confirm_sheet()
    _clear_toast()

    if _overlay and _overlay.winfo_exists():
        try:
            _overlay.place_forget()
            _overlay.destroy()
        except Exception:
            pass

    _restore_chat_input()

    _active = False
    _paused = False
    _game_over = False

    # 🔊 Make XTTS read the neutral close line
    _say_ai("exit", speak=True)

# ============ Game core ============

def _init_game():
    global _snake, _dir, _pending_dir, _food, _score, _speed_ms, _game_over
    _game_over = False
    _score = 0
    _speed_ms = 140
    cx, cy = GRID_COLS // 2, GRID_ROWS // 2
    _snake = [(cx - 1, cy), (cx - 2, cy), (cx - 3, cy)]
    _dir = (1, 0)
    _pending_dir = (1, 0)
    _food = _spawn_food()
    _update_score_label()
    _redraw()

def _spawn_food() -> Tuple[int, int]:
    while True:
        fx = random.randint(0, GRID_COLS - 1)
        fy = random.randint(0, GRID_ROWS - 1)
        if (fx, fy) not in _snake:
            return (fx, fy)

def _update_score_label():
    if _header and hasattr(_header, "score_lbl"):
        try:
            _header.score_lbl.config(text=f"Score: {_score}")  # type: ignore[attr-defined]
        except Exception:
            pass

def _on_canvas_resize(event):
    _recompute_cell()
    _redraw()

def _recompute_cell():
    global CELL, GRID_COLS, GRID_ROWS
    if not _canvas:
        return
    w = max(100, _canvas.winfo_width() - 2 * PADDING)
    h = max(100, _canvas.winfo_height() - 2 * PADDING)
    target_cols, target_rows = 28, 18
    cell_w = w // target_cols
    cell_h = h // target_rows
    CELL = max(12, min(36, cell_w, cell_h))
    GRID_COLS = max(10, w // CELL)
    GRID_ROWS = max(10, h // CELL)

def _grid_to_px(x: int, y: int) -> Tuple[int, int, int, int]:
    if not _canvas:
        return (0, 0, 0, 0)
    w = _canvas.winfo_width()
    h = _canvas.winfo_height()
    gw = GRID_COLS * CELL
    gh = GRID_ROWS * CELL
    ox = (w - gw) // 2
    oy = (h - gh) // 2
    x0 = ox + x * CELL
    y0 = oy + y * CELL
    return (x0 + 1, y0 + 1, x0 + CELL - 1, y0 + CELL - 1)

def _redraw():
    if not _canvas:
        return
    _canvas.delete("all")

    w = _canvas.winfo_width()
    h = _canvas.winfo_height()
    gw = GRID_COLS * CELL
    gh = GRID_ROWS * CELL
    ox = (w - gw) // 2
    oy = (h - gh) // 2
    _canvas.create_rectangle(ox - 2, oy - 2, ox + gw + 2, oy + gh + 2, outline=BORDER, width=2)

    fx, fy = _food
    x0, y0, x1, y1 = _grid_to_px(fx, fy)
    _canvas.create_oval(x0 + 2, y0 + 2, x1 - 2, y1 - 2, fill=APPLE, outline="")

    for i, (sx, sy) in enumerate(_snake):
        x0, y0, x1, y1 = _grid_to_px(sx, sy)
        color = SNAKE_HEAD if i == 0 else SNAKE
        _canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=BG)

    if _paused:
        _draw_pause_overlay()
    if _game_over:
        _draw_game_over_overlay()

def _draw_pause_overlay():
    if not _canvas:
        return
    w, h = _canvas.winfo_width(), _canvas.winfo_height()
    _canvas.create_rectangle(0, 0, w, h, fill=PAUSE_DIM, stipple=PAUSE_STIPPLE, outline="")
    _canvas.create_text(w // 2, h // 2 - 8, text="Paused", fill=FG, font=("Segoe UI", 16, "bold"))
    _canvas.create_text(w // 2, h // 2 + 14, text="Press P to resume", fill=FG_DIM, font=("Segoe UI", 11))

def _draw_game_over_overlay():
    if not _canvas:
        return
    w, h = _canvas.winfo_width(), _canvas.winfo_height()
    _canvas.create_rectangle(0, 0, w, h, fill=PAUSE_DIM, stipple=PAUSE_STIPPLE, outline="")
    _canvas.create_text(w // 2, h // 2 - 16, text="Game Over", fill=FG, font=("Segoe UI", 18, "bold"))
    _canvas.create_text(w // 2, h // 2 + 8, text=f"Score: {_score}", fill=FG_DIM, font=("Segoe UI", 12))
    _canvas.play_again = _canvas.create_text(  # type: ignore[attr-defined]
        w // 2 - 60, h // 2 + 40, text="[ Play Again ]", fill=ACCENT, font=("Segoe UI", 11, "bold")
    )
    _canvas.exit_btn = _canvas.create_text(  # type: ignore[attr-defined]
        w // 2 + 64, h // 2 + 40, text="[ Exit ]", fill=FG, font=("Segoe UI", 11, "bold")
    )
    _canvas.tag_bind(_canvas.play_again, "<Button-1>", lambda e: _restart())  # type: ignore[attr-defined]
    _canvas.tag_bind(_canvas.exit_btn, "<Button-1>", lambda e: _on_back())   # type: ignore[attr-defined]

def _resume_loop():
    global _tick_job
    _cancel_loop()
    _tick_job = _canvas.after(_speed_ms, _tick) if _canvas else None

def _cancel_loop():
    global _tick_job
    if _canvas and _tick_job:
        try:
            _canvas.after_cancel(_tick_job)
        except Exception:
            pass
    _tick_job = None

def _tick():
    if not _canvas or _paused or _game_over:
        _resume_loop()
        return
    _step()
    _redraw()
    _resume_loop()

def _step():
    global _snake, _dir, _pending_dir, _food, _score, _speed_ms, _game_over

    if (_pending_dir[0] != -_dir[0]) or (_pending_dir[1] != -_dir[1]):
        _dir = _pending_dir

    head_x, head_y = _snake[0]
    nx, ny = head_x + _dir[0], head_y + _dir[1]

    if nx < 0 or ny < 0 or nx >= GRID_COLS or ny >= GRID_ROWS:
        _handle_game_over()
        return

    if (nx, ny) in _snake[:]:
        _handle_game_over()
        return

    _snake.insert(0, (nx, ny))

    if (nx, ny) == _food:
        _score += 1
        _update_score_label()
        _food = _spawn_food()
        _toast(f"+1  |  Score: {_score}", ms=900)
        _speed_ms = max(70, int(_speed_ms * 0.97))
    else:
        _snake.pop()

def _handle_game_over():
    global _game_over
    if _game_over:
        return
    _game_over = True
    _say_ai("game_over", score=_score, speak=True)

def _restart():
    """Reset the board and announce the restart (text + XTTS)."""
    global _paused
    _init_game()
    _paused = False
    _resume_loop()
    _say_ai("restart", speak=True)

def _toggle_pause():
    global _paused
    _paused = not _paused
    _redraw()

# ============ Input handling ============

def _on_key(event):
    global _pending_dir
    key = (event.keysym or "").lower()
    if key in ("left", "a"):
        _pending_dir = (-1, 0)
    elif key in ("right", "d"):
        _pending_dir = (1, 0)
    elif key in ("up", "w"):
        _pending_dir = (0, -1)
    elif key in ("down", "s"):
        _pending_dir = (0, 1)
    elif key in ("p",):
        _toggle_pause()
    elif key in ("escape",):
        _on_back()
    elif key == "return" and _game_over:
        _restart()

# =========================================================
# End of module
# =========================================================
