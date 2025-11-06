# nulla_source\scripts\nulla_game_rps.py
# Rock Paper Scissors overlay for Nulla's chat window — import-safe (no GUI at import time)

from __future__ import annotations

import os
import random
import threading
import time
import tkinter as tk
import shutil
import subprocess
import sys
from queue import Queue
from typing import Optional, Tuple, List, Dict, Any

# ============ Module state ============
_overlay: Optional[tk.Frame] = None
_header: Optional[tk.Frame] = None
_canvas: Optional[tk.Canvas] = None
_confirm_sheet: Optional[tk.Frame] = None
_round_sheet: Optional[tk.Frame] = None

_story_win: Optional[tk.Toplevel] = None
_hooks: Dict[str, Any] = {}
_active: bool = False
_game_over: bool = False

_input_prev_state: Optional[str] = None

# jobs
_spin_step_job: Optional[str] = None
_spin_end_job: Optional[str] = None
_toast_job: Optional[str] = None

# Canvas item ids / refs for flicker-free updates
_ids: Dict[str, int] = {}          # canvas item ids
_imgrefs: Dict[str, tk.PhotoImage] = {}  # keep strong refs so Tk doesn't GC images

# Theme (match Runner)
BG = "#0f1115"
PANEL = "#161a20"
BORDER = "#2a2f3a"
FG = "#e6e6e6"
FG_DIM = "#9aa4b2"
ACCENT = "#3b82f6"
PAUSE_DIM = "#000000"
PAUSE_STIPPLE = "gray50"

# RPS visuals
REVEAL_STATIC = "#1b2030"   # steady bg used for Nulla's spinning tile
COLOR_DISABLED = BORDER

CHOICES = ("rock", "paper", "scissors")

# Animation/Sound
SPIN_DURATION_MS = 3000
SPIN_STEP_MS_MIN = 90
SPIN_STEP_MS_MAX = 130

# ---------- assets ----------

def _resolve_asset(rel_name: str) -> str:
    """Resolve asset path for portable folder; also works with PyInstaller onefile (_MEIPASS)."""
    # 1) PyInstaller onefile temp dir
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cand = os.path.join(meipass, "nulla_source", "assets", rel_name)
        if os.path.isfile(cand):
            return cand
    # 2) Side-by-side portable folder (this file lives in nulla_source/scripts/)
    here = os.path.abspath(os.path.dirname(__file__))
    base = os.path.abspath(os.path.join(here, os.pardir))
    cand2 = os.path.join(base, "assets", rel_name)
    if os.path.isfile(cand2):
        return cand2
    # 3) CWD fallback (dev)
    cand3 = os.path.join(os.getcwd(), "nulla_source", "assets", rel_name)
    return cand3

SFX_CLICK_PATH   = _resolve_asset("dice_roll.wav")
SFX_VICTORY_PATH = _resolve_asset("rps_victory.wav")
SFX_DEFEAT_PATH  = _resolve_asset("rps_defeat.wav")

# Base icon images (PNG). Tk 8.6+ supports PNG out of the box.
_icon_base: Dict[str, Optional[tk.PhotoImage]] = {"rock": None, "paper": None, "scissors": None}
# Cache of scaled variants per (name, factor)
_icon_scaled: Dict[Tuple[str, int], tk.PhotoImage] = {}

def _ensure_base_icons():
    for name in ("rock", "paper", "scissors"):
        if _icon_base[name] is None:
            path = _resolve_asset(f"{name}.png")
            try:
                _icon_base[name] = tk.PhotoImage(file=path)
            except Exception:
                _icon_base[name] = None  # fall back to text if missing

def _scaled_icon(name: str, max_w: int, max_h: int) -> Optional[tk.PhotoImage]:
    """
    Return a PhotoImage scaled down to fit inside (max_w, max_h) using integer
    subsample (no Pillow dependency). Keeps a cache for reuse.
    """
    _ensure_base_icons()
    base = _icon_base.get(name)
    if not base:
        return None
    # Compute integer subsample factor (>=1). Ceil division to ensure it fits.
    def ceil_div(a, b): return (a + b - 1) // b
    factor_w = ceil_div(base.width(), max_w) if max_w > 0 else 1
    factor_h = ceil_div(base.height(), max_h) if max_h > 0 else 1
    factor = max(1, factor_w, factor_h)
    key = (name, factor)
    if key in _icon_scaled:
        return _icon_scaled[key]
    img = base.subsample(factor, factor)
    _icon_scaled[key] = img
    return img

# Game state
_target_wins: int = 3
_you_score: int = 0
_nulla_score: int = 0

_your_choice: Optional[str] = None
_nulla_choice: Optional[str] = None

_spinning: bool = False
_spin_display: Optional[str] = None  # what the right tile shows while spinning

# strategy memory
_history: List[Tuple[str, str, str]] = []  # (your_move, nulla_move, outcome)
_last_outcome: Optional[str] = None        # "win"/"lose"/"draw" from YOUR perspective

# layout cache (computed on redraw)
_layout: Dict[str, Tuple[int, int, int, int]] = {}

# =========================================================
# Async AI QUIPS (same style as Runner, short + wholesome)
# =========================================================

def _strip_outer_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1].strip()
    return s

class _GameAI:
    """
    Tiny client for an OpenAI-compatible chat endpoint.
    Uses NULLA_* envs; defaults to http://127.0.0.1:1234
    """
    def __init__(self):
        base_env = os.getenv("NULLA_API_BASE") or "http://127.0.0.1:1234"
        self.base = base_env.rstrip("/")
        self.model = os.getenv("NULLA_MODEL") or "openhermes-2.5-mistral-7b"
        try:
            import requests  # type: ignore
            self._requests = requests
        except Exception:
            self._requests = None

        self.fallback: Dict[str, List[str]] = {
            "load": [
                "Click a tile, then Shoot! First to N wins.",
                "Pick rock, paper, or scissors—then Shoot! First to N.",
            ],
            "restart": [
                "Fresh set—let’s play.",
                "New match. Ready when you are.",
            ],
            "exit": [
                "RPS closed.",
                "Closed the game overlay.",
            ],
            # Different tone depending on result; we still append the exact final-score tail.
            "game_over_win": [
                "Nice win. Final score: You {you} — {bot} Nulla.",
                "You took the set. Final score: You {you} — {bot} Nulla.",
            ],
            "game_over_lose": [
                "Good fight—we’ll get the next one. Final score: You {you} — {bot} Nulla.",
                "Well played. Final score: You {you} — {bot} Nulla.",
            ],
        }

    def _sys_prompt(self, kind: str, ctx: Dict[str, Any]) -> str:
        base = (
            "You are Nulla: warm, calm, supportive, friendly, and kind.\n"
            "Write EXACTLY ONE short line (<= 120 chars). No emojis. Never mention AI or prompts.\n"
            "Keep it wholesome and encouraging. Vary the wording naturally each time.\n"
        )
        if kind == "load":
            base += "Event: game_load (rps). Briefly remind: 'Click a tile, then Shoot! First to N wins.'\n"
        elif kind == "restart":
            base += "Event: restart. Short and friendly.\n"
        elif kind == "exit":
            base += "Event: exit. Confirm the game overlay closed. Do NOT say goodbye or imply leaving the app.\n"
        elif kind == "game_over":
            res = ctx.get("result", "")
            if res == "win":
                base += "Event: game_over. Player WON. Congratulate lightly.\n"
            elif res == "lose":
                base += "Event: game_over. Player LOST. Encourage gently.\n"
            base += (
                "You MUST end the sentence with exactly: "
                f"'Final score: You {ctx.get('you', 0)} — {ctx.get('bot', 0)} Nulla.'\n"
            )
        return base

    def _user_prompt(self, kind: str, ctx: Dict[str, Any]) -> str:
        parts = [f"Kind={kind}"]
        if "you" in ctx:
            parts.append(f"You={ctx['you']}")
        if "bot" in ctx:
            parts.append(f"Nulla={ctx['bot']}")
        if "target" in ctx:
            parts.append(f"Target={ctx['target']}")
        if "result" in ctx:
            parts.append(f"Result={ctx['result']}")
        return " | ".join(parts)

    def gen(self, kind: str, **ctx) -> str:
        if not self._requests:
            if kind == "game_over":
                tail = f"Final score: You {ctx.get('you', 0)} — {ctx.get('bot', 0)} Nulla."
                if ctx.get("result") == "win":
                    return random.choice(self.fallback["game_over_win"]).format(**ctx)
                return random.choice(self.fallback["game_over_lose"]).format(**ctx)
            return random.choice(self.fallback.get(kind, ["Okay."]))

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._sys_prompt(kind, ctx)},
                {"role": "user", "content": self._user_prompt(kind, ctx)},
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
            line = (text.split("\n")[0] if text else "").strip()[:200]
            if kind == "game_over":
                tail = f"Final score: You {ctx.get('you', 0)} — {ctx.get('bot', 0)} Nulla."
                if not line.endswith(tail):
                    if "Final score:" in line:
                        idx = line.rfind("Final score:")
                        line = line[:idx].rstrip() + " " + tail
                    else:
                        if not line.endswith("."):
                            line += "."
                        line += " " + tail
            return line or random.choice(self.fallback.get(kind, ["Okay."]))
        except Exception:
            if kind == "game_over":
                if ctx.get("result") == "win":
                    return random.choice(self.fallback["game_over_win"]).format(**ctx)
                return random.choice(self.fallback["game_over_lose"]).format(**ctx)
            return random.choice(self.fallback.get(kind, ["Okay."]))

_AI = _GameAI()

# ---------- async announce pipeline ----------
_ai_q: "Queue[Tuple[str, bool, Dict[str, Any]]]" = Queue()
_ai_worker_started = False

def _ensure_ai_worker():
    global _ai_worker_started
    if _ai_worker_started:
        return
    _ai_worker_started = True

    def _worker():
        while True:
            try:
                kind, speak, ctx = _ai_q.get()
            except Exception:
                continue
            try:
                line = _AI.gen(kind, **ctx)
                _announce(line, speak=speak)
            except Exception:
                pass

    threading.Thread(target=_worker, daemon=True).start()

def _queue_ai(kind: str, speak: bool = False, **kw):
    try:
        _ai_q.put_nowait((kind, speak, kw))
    except Exception:
        pass

def _announce(msg: str, speak: bool = False):
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
    _queue_ai(kind, speak=speak, **kw)

# =========================================================
# Public API expected by GameManager
# =========================================================

def mount_overlay(story_win: tk.Toplevel, hooks: Dict[str, Any]) -> None:
    """Required entrypoint. Creates the overlay and starts the game."""
    global _overlay, _header, _canvas, _story_win, _hooks, _active, _game_over, _ids

    if _active and _overlay and _overlay.winfo_exists():
        _overlay.lift()
        return

    _ensure_ai_worker()

    _story_win = story_win
    _hooks = hooks or {}

    # Pause idle/short-idle while the game is active
    try:
        if callable(_hooks.get("idle_block_push")):
            _hooks["idle_block_push"]()
    except Exception:
        pass

    _disable_chat_input()

    _overlay = tk.Frame(_story_win, bg=BG, highlightthickness=1, highlightbackground=BORDER)
    _overlay.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)
    _overlay.lift()

    _build_header()

    _canvas = tk.Canvas(_overlay, bg=BG, highlightthickness=0)
    _canvas.pack(side="top", fill="both", expand=True)
    _canvas.bind("<Configure>", lambda e: _redraw())

    _overlay.bind("<KeyPress>", _on_key_down)
    _overlay.bind("<KeyRelease>", _on_key_up)
    _overlay.bind("<FocusIn>", lambda e: (_overlay.focus_set()))
    _overlay.focus_set()
    _overlay.update_idletasks()

    _reset_match()
    _active = True
    _game_over = False
    _ids.clear()

    _show_round_picker()
    _redraw()

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

# ============ Build header / helpers ============

def _build_header():
    global _header
    _header = tk.Frame(_overlay, bg=PANEL, height=44, highlightthickness=0)
    _header.pack(side="top", fill="x")

    back_btn = tk.Button(
        _header, text="← Back", relief="flat",
        bg=BORDER, fg=FG, activebackground=BORDER, activeforeground=FG,
        command=_on_back
    )
    back_btn.pack(side="left", padx=8, pady=8)

    title = tk.Label(_header, text="Rock Paper Scissors", bg=PANEL, fg=FG, font=("Segoe UI", 11, "bold"))
    title.pack(side="left", padx=8)

    score_lbl = tk.Label(_header, text=_score_text(), bg=PANEL, fg=FG_DIM, font=("Consolas", 11))
    score_lbl.pack(side="right", padx=12)
    _header.score_lbl = score_lbl  # type: ignore[attr-defined]

def _score_text() -> str:
    return f"You {_you_score} — {_nulla_score} Nulla  (First to {_target_wins})"

def _update_score_label():
    if _header and hasattr(_header, "score_lbl"):
        try:
            _header.score_lbl.config(text=_score_text())  # type: ignore[attr-defined]
        except Exception:
            pass

def _now_ms() -> float:
    return time.perf_counter() * 1000.0

def _get_chat_entry_widget() -> Optional[tk.Entry]:
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

def _toast(msg: str, ms: int = 1100):
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
        w, h = _size()
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

def _unbind_keys():
    try:
        if _overlay and _overlay.winfo_exists():
            _overlay.unbind("<KeyPress>")
            _overlay.unbind("<KeyRelease>")
            _overlay.unbind("<FocusIn>")
    except Exception:
        pass

def _teardown():
    _say_ai("exit", speak=True)
    try:
        if callable(_hooks.get("idle_block_pop")):
            _hooks["idle_block_pop"]()
    except Exception:
        pass

    _cancel_spin()
    _hide_round_picker()
    _hide_confirm_sheet()
    _clear_toast()
    _unbind_keys()

    if _overlay and _overlay.winfo_exists():
        try:
            _overlay.place_forget()
            _overlay.destroy()
        except Exception:
            pass

    _restore_chat_input()

    global _active, _game_over
    _active = False
    _game_over = False
    _ids.clear()
    _imgrefs.clear()
    _history.clear()

# ============ Game core ============

def _reset_match(target: Optional[int] = None):
    global _you_score, _nulla_score, _your_choice, _nulla_choice, _game_over, _target_wins
    if target:
        _target_wins = target
    _you_score = 0
    _nulla_score = 0
    _your_choice = None
    _nulla_choice = None
    _history.clear()
    global _last_outcome
    _last_outcome = None
    _game_over = False
    _update_score_label()

def _size() -> Tuple[int, int]:
    if not _canvas:
        return (800, 600)
    return max(240, _canvas.winfo_width()), max(240, _canvas.winfo_height())

def _show_round_picker():
    global _round_sheet
    if not _overlay or (_round_sheet and _round_sheet.winfo_exists()):
        return

    if _canvas:
        w, h = _size()
        _canvas.dim_round = _canvas.create_rectangle(  # type: ignore[attr-defined]
            0, 0, w, h, fill=PAUSE_DIM, outline="", stipple=PAUSE_STIPPLE
        )

    _round_sheet = tk.Frame(_overlay, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
    _round_sheet.place(relx=0.5, rely=0.5, anchor="center")

    lbl = tk.Label(_round_sheet, text="Select target wins", bg=PANEL, fg=FG, font=("Segoe UI", 12, "bold"))
    lbl.pack(padx=18, pady=(16, 10))

    btns = tk.Frame(_round_sheet, bg=PANEL)
    btns.pack(padx=16, pady=(0, 16))

    def pick(n: int):
        _hide_round_picker()
        _reset_match(target=n)
        _say_ai("load", speak=True, target=n)
        _redraw()

    for n in (3, 5, 10):
        b = tk.Button(btns, text=f"First to {n}", relief="flat",
                      bg=ACCENT, fg="white",
                      activebackground=ACCENT, activeforeground="white",
                      command=lambda nn=n: pick(nn))
        b.pack(side="left", padx=6)

def _hide_round_picker():
    global _round_sheet
    if _round_sheet and _round_sheet.winfo_exists():
        try:
            _round_sheet.destroy()
        except Exception:
            pass
    _round_sheet = None
    if _canvas and hasattr(_canvas, "dim_round"):
        try:
            _canvas.delete(_canvas.dim_round)  # type: ignore[attr-defined]
            delattr(_canvas, "dim_round")
        except Exception:
            pass

def _start_spin():
    global _spinning, _spin_display
    if _spinning or not _canvas:
        return
    _spinning = True
    _spin_display = random.choice(CHOICES)
    # initial draw ensures right tile + image id exist
    _redraw()
    _schedule_spin_step()
    _schedule_spin_end()

def _schedule_spin_step():
    global _spin_step_job, _spin_display
    if not _canvas or not _spinning:
        return
    _spin_display = random.choice(CHOICES)
    _update_spin_image_only()
    delay = random.randint(SPIN_STEP_MS_MIN, SPIN_STEP_MS_MAX)
    _spin_step_job = _canvas.after(delay, _schedule_spin_step)

def _schedule_spin_end():
    global _spin_end_job
    if not _canvas:
        return
    _spin_end_job = _canvas.after(SPIN_DURATION_MS, _resolve_round)

def _cancel_spin():
    global _spin_step_job, _spin_end_job, _spinning
    if _canvas and _spin_step_job:
        try:
            _canvas.after_cancel(_spin_step_job)
        except Exception:
            pass
    if _canvas and _spin_end_job:
        try:
            _canvas.after_cancel(_spin_end_job)
        except Exception:
            pass
    _spin_step_job = None
    _spin_end_job = None
    _spinning = False

# ---------- strategy helpers ----------

def _beats(move: str) -> str:
    return {"rock": "paper", "paper": "scissors", "scissors": "rock"}[move]

def _loses_to(move: str) -> str:
    # move that is beaten by 'move'
    return {"rock": "scissors", "paper": "rock", "scissors": "paper"}[move]

def _predict_player_next() -> str:
    """
    Heuristic predictor:
    - If you just WON, assume 55% you repeat last move.
    - If you just LOST, assume 55% you switch to the move that beats what beat you.
    - Otherwise use a weighted frequency over the last 6 moves (recent weighted higher).
    Fallback: random.
    """
    if not _history:
        return random.choice(CHOICES)

    your_last, nulla_last, outcome = _history[-1]
    # 55% outcome-based bias
    r = random.random()
    if outcome == "win" and r < 0.55:
        return your_last
    if outcome == "lose" and r < 0.55:
        # likely to counter what beat you -> choose the move that beats nulla_last
        return _beats(nulla_last)

    # Weighted frequency (last 6, weights 1.0,0.8,0.6,0.5,0.4,0.3)
    weights = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3]
    counts = {"rock": 0.0, "paper": 0.0, "scissors": 0.0}
    take = _history[-6:]
    for i, (ym, _, _) in enumerate(reversed(take)):
        w = weights[i] if i < len(weights) else 0.2
        counts[ym] += w
    # If tie, random choice among max
    maxv = max(counts.values())
    cands = [m for m, v in counts.items() if abs(v - maxv) < 1e-6]
    return random.choice(cands)

def _choose_nulla_move() -> str:
    """
    Choose a move that counters the predicted player move,
    with some randomness so she isn’t predictable.
    """
    predicted = _predict_player_next()
    counter = _beats(predicted)

    r = random.random()
    if r < 0.15:
        # 15%: true random to stay spicy
        return random.choice(CHOICES)
    if r < 0.35:
        # 20%: mirror predicted (mind game)
        return predicted
    # otherwise play the counter
    return counter

def _add_history(your: str, nulla: str, outcome: str):
    _history.append((your, nulla, outcome))
    if len(_history) > 50:
        del _history[:-50]
    global _last_outcome
    _last_outcome = outcome

# ---------- round resolution ----------

def _resolve_round():
    global _spinning, _you_score, _nulla_score, _nulla_choice, _your_choice
    _cancel_spin()
    _nulla_choice = _choose_nulla_move()
    outcome = _judge(_your_choice, _nulla_choice)

    if outcome == "win":
        _you_score += 1
        _update_score_label()
        _toast("You win the round!", ms=900)
    elif outcome == "lose":
        _nulla_score += 1
        _update_score_label()
        _toast("Nulla wins the round!", ms=900)
    else:
        _toast("Draw.", ms=700)

    # remember behavior
    if _your_choice:
        _add_history(_your_choice, _nulla_choice, outcome)

    _redraw()

    if _you_score >= _target_wins or _nulla_score >= _target_wins:
        _end_match()
        return

    _your_choice = None

def _judge(a: Optional[str], b: Optional[str]) -> str:
    if a is None or b is None:
        return "draw"
    if a == b:
        return "draw"
    if (a == "rock" and b == "scissors") or \
       (a == "paper" and b == "rock") or \
       (a == "scissors" and b == "paper"):
        return "win"
    return "lose"

def _end_match():
    """Set game-over, draw overlay, play result SFX once, and have Nulla speak."""
    global _game_over
    _game_over = True

    # Draw the overlay first so the SFX feels tied to the screen appearing.
    _redraw()

    # Play one-shot victory/defeat sound immediately.
    if _you_score > _nulla_score:
        _play_wav(SFX_VICTORY_PATH)
        result = "win"
    else:
        _play_wav(SFX_DEFEAT_PATH)
        result = "lose"

    # Nulla speaks a final one-liner (already handled, now with result context).
    _say_ai("game_over", speak=True, you=_you_score, bot=_nulla_score, result=result)

def _restart_same_target():
    global _your_choice, _nulla_choice, _game_over
    _your_choice = None
    _nulla_choice = None
    _game_over = False
    _cancel_spin()
    _reset_match(target=_target_wins)
    _say_ai("restart", speak=True)
    _redraw()

# ============ Drawing & layout ============

def _on_canvas_click(event):
    if not _canvas:
        return
    x, y = event.x, event.y

    # When game is over, check overlay buttons FIRST
    if _game_over:
        r = _layout.get("play_again")
        if r and r[0] <= x <= r[2] and r[1] <= y <= r[3]:
            _restart_same_target()
            return
        r = _layout.get("exit_btn")
        if r and r[0] <= x <= r[2] and r[1] <= y <= r[3]:
            _on_back()
            return
        # if overlay is up but miss-clicked, ignore
        return

    # Normal play: options + shoot
    for name in ("opt_rock", "opt_paper", "opt_scissors"):
        r = _layout.get(name)
        if r and r[0] <= x <= r[2] and r[1] <= y <= r[3]:
            _select_choice(name.split("_", 1)[1])
            return

    r = _layout.get("shoot_btn")
    if r and r[0] <= x <= r[2] and r[1] <= y <= r[3]:
        _on_shoot()
        return

def _on_canvas_resize(event):
    _redraw()

def _select_choice(choice: str):
    global _your_choice
    if _spinning or _game_over:
        return
    if choice in CHOICES:
        _your_choice = choice
        _redraw()

def _on_shoot():
    if _game_over or _spinning:
        return
    if _your_choice is None:
        _toast("Pick rock, paper, or scissors.", ms=900)
        return
    _play_sound_once()     # play dice_roll.wav once per click
    _start_spin()

def _size_rects(w: int, h: int):
    pad = 12
    top_h = int(max(100, h * 0.26))
    mid_h = int(max(120, h * 0.38))

    tile_gap = 12
    tile_w = int((w - pad * 2 - tile_gap * 2) / 3)
    tile_h = top_h - pad * 2
    y0 = pad
    x0 = pad
    _layout["opt_rock"] = (x0, y0, x0 + tile_w, y0 + tile_h)
    _layout["opt_paper"] = (x0 + tile_w + tile_gap, y0, x0 + 2 * tile_w + tile_gap, y0 + tile_h)
    _layout["opt_scissors"] = (x0 + 2 * (tile_w + tile_gap), y0, x0 + 3 * tile_w + 2 * tile_gap, y0 + tile_h)

    mid_pad = pad
    mid_y0 = top_h + mid_pad
    mid_y1 = top_h + mid_h - mid_pad
    gap = 20
    box_w = int((w - mid_pad * 2 - gap) / 2)
    _layout["reveal_you"] = (mid_pad, mid_y0, mid_pad + box_w, mid_y1)
    _layout["reveal_nulla"] = (mid_pad + box_w + gap, mid_y0, w - mid_pad, mid_y1)

    btn_h = 42
    btn_w = 160
    btn_y0 = h - btn_h - pad
    btn_y1 = h - pad
    btn_x0 = int((w - btn_w) / 2)
    _layout["shoot_btn"] = (btn_x0, btn_y0, btn_x0 + btn_w, btn_y1)

    go_btn_w = 140
    go_btn_h = 36
    go_y = mid_y1 - go_btn_h - 8
    go_gap = 18
    go_x0 = int(w / 2 - go_gap / 2 - go_btn_w)
    go_x1 = int(w / 2 + go_gap / 2)
    _layout["play_again"] = (go_x0, go_y, go_x0 + go_btn_w, go_y + go_btn_h)
    _layout["exit_btn"] = (go_x1, go_y, go_x1 + go_btn_w, go_y + go_btn_h)

def _draw_button(rect: Tuple[int, int, int, int], label: str, enabled: bool = True, primary: bool = True):
    x0, y0, x1, y1 = rect
    fill = ACCENT if primary and enabled else (COLOR_DISABLED if not enabled else BORDER)
    fg = "white" if primary and enabled else FG
    _canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=BORDER, width=1)
    _canvas.create_text((x0 + x1) // 2, (y0 + y1) // 2, text=label, fill=fg, font=("Segoe UI", 11, "bold"))

def _draw_icon_tile(rect: Tuple[int, int, int, int], name: str, selected: bool = False):
    """Top selectable tiles with icons."""
    x0, y0, x1, y1 = rect
    border = ACCENT if selected else BORDER
    _canvas.create_rectangle(x0, y0, x1, y1, fill=BG, outline=border, width=2)
    max_w = int((x1 - x0) * 0.9)
    max_h = int((y1 - y0) * 0.9)
    img = _scaled_icon(name, max_w, max_h)
    if img:
        iid = _canvas.create_image((x0 + x1) // 2, (y0 + y1) // 2, image=img)
        _imgrefs[f"opt_{name}"] = img  # keep ref
        _ids[f"opt_{name}_img"] = iid
    else:
        # fallback text
        _canvas.create_text((x0 + x1) // 2, (y0 + y1) // 2, text=name.upper(), fill=FG, font=("Segoe UI", 12, "bold"))

def _draw_reveal_icon(rect: Tuple[int, int, int, int], choice: Optional[str], *, spin_static: bool = False, side_key: str = ""):
    """Reveal tiles. If spin_static=True, keep steady bg and just swap image."""
    x0, y0, x1, y1 = rect
    fill = REVEAL_STATIC if (spin_static or choice is None) else BG
    _canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=BG, width=1)

    if choice in CHOICES:
        max_w = int((x1 - x0) * 0.9)
        max_h = int((y1 - y0) * 0.9)
        img = _scaled_icon(choice, max_w, max_h)
        if img:
            iid = _canvas.create_image((x0 + x1) // 2, (y0 + y1) // 2, image=img)
            if side_key:
                _imgrefs[f"{side_key}_imgref"] = img
                _ids[f"{side_key}_img"] = iid
            return
    # fallback placeholder
    _canvas.create_text((x0 + x1) // 2, (y0 + y1) // 2, text="?", fill=FG, font=("Segoe UI", 16, "bold"))

def _redraw():
    if not _canvas:
        return
    _canvas.delete("all")
    _ids.clear()
    _canvas.unbind("<Button-1>")
    _canvas.bind("<Button-1>", _on_canvas_click)

    w, h = _size()
    _size_rects(w, h)

    # Top options
    _draw_icon_tile(_layout["opt_rock"], "rock", selected=(_your_choice == "rock"))
    _draw_icon_tile(_layout["opt_paper"], "paper", selected=(_your_choice == "paper"))
    _draw_icon_tile(_layout["opt_scissors"], "scissors", selected=(_your_choice == "scissors"))

    # Labels under reveal boxes
    rx0, ry0, rx1, ry1 = _layout["reveal_you"]
    _canvas.create_text((rx0 + rx1) // 2, ry0 - 10, text="You", fill=FG_DIM, font=("Segoe UI", 10))
    nx0, ny0, nx1, ny1 = _layout["reveal_nulla"]
    _canvas.create_text((nx0 + nx1) // 2, ny0 - 10, text="Nulla", fill=FG_DIM, font=("Segoe UI", 10))

    # Left reveal (your pick or '?')
    _draw_reveal_icon(_layout["reveal_you"], _your_choice, spin_static=False, side_key="you")

    # Right reveal (spin or result)
    if _spinning:
        # steady bg, icon swaps
        _draw_reveal_icon(_layout["reveal_nulla"], _spin_display, spin_static=True, side_key="nulla")
    else:
        _draw_reveal_icon(_layout["reveal_nulla"], _nulla_choice, spin_static=False, side_key="nulla")

    # Shoot button
    enabled = (not _spinning) and (not _game_over) and (_your_choice is not None)
    _draw_button(_layout["shoot_btn"], "Shoot!", enabled=enabled, primary=True)

    # Game over overlay
    if _game_over:
        _canvas.create_rectangle(0, 0, w, h, fill=PAUSE_DIM, stipple=PAUSE_STIPPLE, outline="")
        _canvas.create_text(w // 2, h // 2 - 40, text="Match Over", fill=FG, font=("Segoe UI", 18, "bold"))
        _canvas.create_text(w // 2, h // 2 - 14, text=f"Score: You {_you_score} — {_nulla_score} Nulla",
                            fill=FG_DIM, font=("Segoe UI", 12))
        _draw_button(_layout["play_again"], "Play Again", enabled=True, primary=True)
        _draw_button(_layout["exit_btn"], "Exit", enabled=True, primary=False)

def _update_spin_image_only():
    """While spinning, avoid full redraw; just swap the image inside Nulla's tile."""
    if not _canvas or not _spinning:
        return
    img_id = _ids.get("nulla_img")
    rect = _layout.get("reveal_nulla")
    if not rect:
        _redraw()
        rect = _layout.get("reveal_nulla")
    if not rect:
        return
    max_w = int((rect[2] - rect[0]) * 0.9)
    max_h = int((rect[3] - rect[1]) * 0.9)
    img = _scaled_icon(_spin_display or "rock", max_w, max_h)
    if img:
        if not img_id:
            # create if missing
            iid = _canvas.create_image((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2, image=img)
            _ids["nulla_img"] = iid
        else:
            try:
                _canvas.itemconfigure(img_id, image=img)
            except Exception:
                pass
        _imgrefs["nulla_imgref"] = img  # keep ref

# ============ Input handling ============

def _on_key_down(event):
    key = (event.keysym or "").lower()
    if key == "escape":
        _on_back()
    elif key in ("r",):
        _select_choice("rock")
    elif key in ("p",):
        _select_choice("paper")
    elif key in ("s",):
        _select_choice("scissors")
    elif key in ("return", "enter"):
        _on_shoot()

def _on_key_up(event):
    pass

# ============ Sound ============

def _play_wav(path: str):
    """Play a single WAV once, with multiple fallbacks. Set NULLA_DISABLE_SFX=1 to skip."""
    if os.getenv("NULLA_DISABLE_SFX") == "1":
        return
    if not path or not os.path.isfile(path):
        return

    def worker():
        # 1) simpleaudio if available
        try:
            import simpleaudio  # type: ignore
            try:
                wave_obj = simpleaudio.WaveObject.from_wave_file(path)  # type: ignore
                wave_obj.play()
                return
            except Exception:
                pass
        except Exception:
            pass

        # 2) Windows winsound
        try:
            import platform
            if platform.system().lower().startswith("win"):
                import winsound  # type: ignore
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)  # type: ignore
                return
        except Exception:
            pass

        # 3) CLI players (non-blocking)
        try:
            DEVNULL = subprocess.DEVNULL
            if shutil.which("ffplay"):
                subprocess.Popen(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                    stdout=DEVNULL, stderr=DEVNULL
                )
                return
            if shutil.which("aplay"):
                subprocess.Popen(["aplay", path], stdout=DEVNULL, stderr=DEVNULL)
                return
            if shutil.which("paplay"):
                subprocess.Popen(["paplay", path], stdout=DEVNULL, stderr=DEVNULL)
                return
            if shutil.which("afplay"):  # macOS
                subprocess.Popen(["afplay", path], stdout=DEVNULL, stderr=DEVNULL)
                return
        except Exception:
            pass
        # otherwise silent

    try:
        threading.Thread(target=worker, daemon=True).start()
    except Exception:
        pass

def _play_sound_once():
    """Back-compat click sound for 'Shoot!' button."""
    _play_wav(SFX_CLICK_PATH)

# ============ End / confirm ============

def _on_canvas_destroy():
    _cancel_spin()

def _on_back():
    _show_confirm_sheet(
        text="Are you sure you want to exit?",
        yes_cb=_teardown,
        no_cb=_hide_confirm_sheet
    )
