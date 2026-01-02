# nulla_source\scripts\nulla_game_ttt.py
# Tic Tac Toe game

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
_game_over: bool = False   # match over (first to N reached)

_input_prev_state: Optional[str] = None

_toast_job: Optional[str] = None

# Game / board state
_target_wins: int = 3
_you_score: int = 0
_nulla_score: int = 0

_board: List[Optional[str]] = [None] * 9   # 3x3 board, row-major, values: "X", "O", or None
_player_turn: bool = True                  # True => player's turn (X), False => Nulla's turn (O)
_round_over: bool = False                  # True while showing finished board before next one
_winning_line: Optional[Tuple[int, int, int]] = None  # indices of winning line for current board
_status_msg: str = "Select target wins to start."

# Layout cache
_layout: Dict[str, Tuple[int, int, int, int]] = {}

# Colors / theme (aligned with RPS theme)
BG = "#0f1115"
PANEL = "#161a20"
BORDER = "#2a2f3a"
FG = "#e6e6e6"
FG_DIM = "#9aa4b2"
ACCENT = "#3b82f6"
PAUSE_DIM = "#000000"
PAUSE_STIPPLE = "gray50"

X_COLOR = "#ef4444"   # red
O_COLOR = "#3b82f6"   # blue
GRID_COLOR = "#e5e7eb"  # light gray
WIN_LINE_COLOR = "#ffffff"

# Fumble chance for Nulla "forgetting" to block a winning player move
def _parse_fumble_env(default: float) -> float:
    val = os.getenv("NULLA_TTT_FUMBLE")
    if not val:
        return default
    try:
        f = float(val)
        if f < 0.0:
            return 0.0
        if f > 1.0:
            return 1.0
        return f
    except Exception:
        return default

FUMBLE_CHANCE: float = _parse_fumble_env(0.1769)  # Base fumble chance (~17.69%). Can be overridden via env NULLA_TTT_FUMBLE (0.0–1.0).

# ---------- asset resolution / SFX (same style as RPS) ----------

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

SFX_SELECT_PATH = _resolve_asset("ttt_select.wav")
SFX_VICTORY_PATH = _resolve_asset("victory1.wav")
SFX_DEFEAT_PATH = _resolve_asset("defeat1.wav")

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

def _play_select_sfx():
    _play_wav(SFX_SELECT_PATH)

def _play_victory_sfx():
    _play_wav(SFX_VICTORY_PATH)

def _play_defeat_sfx():
    _play_wav(SFX_DEFEAT_PATH)

# =========================================================
# Async AI QUIPS (same pattern as Runner/RPS)
# =========================================================

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
                "Tic Tac Toe loaded. Click a square to place X. First to N boards wins.",
                "Game on. You’re X, I’m O. First to N boards.",
            ],
            "restart": [
                "New Tic Tac Toe match. Same target, fresh score.",
                "We reset the match. Let’s see how this set goes.",
            ],
            "exit": [
                "Closed the Tic Tac Toe overlay.",
                "Tic Tac Toe closed. We can play again anytime.",
            ],
            "game_over_win": [
                "You won the match. Final score: You {you} — {bot} Nulla.",
                "Nice match. You took it. Final score: You {you} — {bot} Nulla.",
            ],
            "game_over_lose": [
                "Good set. I edged this one. Final score: You {you} — {bot} Nulla.",
                "That was fun. I won this match. Final score: You {you} — {bot} Nulla.",
            ],
            "game_over_draw": [
                "Match ended even. Final score: You {you} — {bot} Nulla.",
                "We tied this one. Final score: You {you} — {bot} Nulla.",
            ],
        }

    def _sys_prompt(self, kind: str, ctx: Dict[str, Any]) -> str:
        base = (
            "You are Nulla: warm, calm, supportive, friendly, and kind.\n"
            "Write EXACTLY ONE short line (<= 120 chars). No emojis. Never mention AI or prompts.\n"
            "Keep it wholesome and encouraging. Vary the wording naturally each time.\n"
        )
        if kind == "load":
            base += "Event: game_load (tic_tac_toe). Briefly remind the rules: player is X, Nulla is O, first to N boards wins.\n"
        elif kind == "restart":
            base += "Event: restart. New match with same target. Short and friendly.\n"
        elif kind == "exit":
            base += "Event: exit. Confirm the game overlay closed. Do NOT say goodbye or imply leaving the app.\n"
        elif kind == "game_over":
            res = ctx.get("result", "")
            if res == "win":
                base += "Event: game_over. Player WON the match. Congratulate lightly.\n"
            elif res == "lose":
                base += "Event: game_over. Player LOST the match. Encourage gently.\n"
            elif res == "draw":
                base += "Event: game_over. The match ended in a DRAW. Be lightly playful or reflective.\n"
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
            # Fallbacks
            if kind == "game_over":
                res = ctx.get("result")
                if res == "win":
                    key = "game_over_win"
                elif res == "lose":
                    key = "game_over_lose"
                else:
                    key = "game_over_draw"
                return random.choice(self.fallback.get(key, ["Match over."])).format(**ctx)
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
                res = ctx.get("result")
                if res == "win":
                    key = "game_over_win"
                elif res == "lose":
                    key = "game_over_lose"
                else:
                    key = "game_over_draw"
                return random.choice(self.fallback.get(key, ["Match over."])).format(**ctx)
            return random.choice(self.fallback.get(kind, ["Okay."]))

_AI = _GameAI()

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
            _hooks["story_push"](msg)
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
    global _overlay, _header, _canvas, _story_win, _hooks, _active, _game_over, _layout

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

    _active = True
    _game_over = False
    _layout.clear()

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

    title = tk.Label(_header, text="Tic Tac Toe", bg=PANEL, fg=FG, font=("Segoe UI", 11, "bold"))
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

    global _active, _game_over, _board, _round_over, _winning_line
    _active = False
    _game_over = False
    _board = [None] * 9
    _round_over = False
    _winning_line = None
    _layout.clear()

# ============ Game core ============

def _size() -> Tuple[int, int]:
    if not _canvas:
        return (800, 600)
    return max(240, _canvas.winfo_width()), max(260, _canvas.winfo_height())

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

def _reset_match(target: Optional[int] = None):
    global _you_score, _nulla_score, _target_wins, _game_over
    if target:
        _target_wins = target
    _you_score = 0
    _nulla_score = 0
    _game_over = False
    _update_score_label()
    _reset_board()

def _reset_board():
    global _board, _player_turn, _round_over, _winning_line, _status_msg
    _board = [None] * 9
    _player_turn = True
    _round_over = False
    _winning_line = None
    _status_msg = "Your turn. Click a square to place X."
    _redraw()

WINNING_COMBOS: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)

def _board_full() -> bool:
    return all(c is not None for c in _board)

def _find_winner() -> Optional[Tuple[str, Tuple[int, int, int]]]:
    for a, b, c in WINNING_COMBOS:
        v = _board[a]
        if v and v == _board[b] == _board[c]:
            return v, (a, b, c)
    return None

def _find_winning_move(mark: str) -> Optional[int]:
    for a, b, c in WINNING_COMBOS:
        line = [_board[a], _board[b], _board[c]]
        if line.count(mark) == 2 and line.count(None) == 1:
            if _board[a] is None:
                return a
            if _board[b] is None:
                return b
            if _board[c] is None:
                return c
    return None

def _nulla_best_move() -> Optional[int]:
    # 1) Can Nulla win now?
    win_move = _find_winning_move("O")
    if win_move is not None:
        return win_move

    # 2) Block player win, with small chance to fumble
    block_move = _find_winning_move("X")
    if block_move is not None:
        if FUMBLE_CHANCE > 0.0 and random.random() < FUMBLE_CHANCE:
            # Try to pick any *other* free square so she "forgets" to block
            free_idxs = [i for i, v in enumerate(_board) if v is None and i != block_move]
            if free_idxs:
                return random.choice(free_idxs)
            # If block_move is literally the only move left, she has to take it
        return block_move

    # 3) Take center
    if _board[4] is None:
        return 4

    # 4) Take opposite corner
    corners = [(0, 8), (2, 6)]
    for c, o in corners:
        if _board[c] == "X" and _board[o] is None:
            return o

    # 5) Any corner
    for i in (0, 2, 6, 8):
        if _board[i] is None:
            return i

    # 6) Any side
    for i in (1, 3, 5, 7):
        if _board[i] is None:
            return i

    return None

def _handle_player_move(idx: int):
    global _status_msg, _player_turn
    if _game_over or _round_over or not _player_turn:
        return
    if idx < 0 or idx >= 9 or _board[idx] is not None:
        return

    _board[idx] = "X"
    _player_turn = False
    _status_msg = "Nulla's turn..."
    _play_select_sfx()
    _redraw()

    winner = _find_winner()
    if winner:
        mark, line = winner
        _finish_round(mark, line)
        return
    if _board_full():
        _finish_round(None, None)
        return

    # Delay Nulla's move by 1 second (1000 ms)
    if _canvas:
        _canvas.after(1000, _nulla_move)
    else:
        _nulla_move()

def _nulla_move():
    global _status_msg, _player_turn
    if _game_over or _round_over:
        return

    move = _nulla_best_move()
    if move is None:
        # should not happen, but treat as draw
        _finish_round(None, None)
        return

    _board[move] = "O"
    _player_turn = True
    _status_msg = "Your turn."
    _redraw()

    winner = _find_winner()
    if winner:
        mark, line = winner
        _finish_round(mark, line)
        return
    if _board_full():
        _finish_round(None, None)
        return

def _finish_round(mark: Optional[str], line: Optional[Tuple[int, int, int]]):
    global _round_over, _you_score, _nulla_score, _winning_line, _status_msg
    _round_over = True
    _winning_line = line

    if mark == "X":
        _you_score += 1
        _status_msg = "You won this board."
        _toast("You win this board!", ms=900)
        # Match SFX is handled at match-end only.
    elif mark == "O":
        _nulla_score += 1
        _status_msg = "Nulla wins this board."
        _toast("Nulla wins this board!", ms=900)
        # Match SFX is handled at match-end only.
    else:
        _status_msg = "Board ended in a draw."
        _toast("Draw.", ms=900)

    _update_score_label()
    _redraw()

    # Check for match end
    if _you_score >= _target_wins or _nulla_score >= _target_wins:
        if _you_score > _nulla_score:
            result = "win"
        elif _nulla_score > _you_score:
            result = "lose"
        else:
            result = "draw"
        _end_match(result)
    else:
        # schedule next board
        if _canvas:
            _canvas.after(1200, _reset_board)

def _end_match(result: str):
    global _game_over
    _game_over = True
    _redraw()

    # Play match-level SFX only when the entire match is decided
    if result == "win":
        _play_victory_sfx()
    elif result == "lose":
        _play_defeat_sfx()
    # Draw gets no SFX

    _say_ai("game_over", speak=True, you=_you_score, bot=_nulla_score, result=result)

def _restart_same_target():
    _say_ai("restart", speak=True)
    _reset_match(target=_target_wins)
    _redraw()

# ============ Drawing & layout ============

def _on_canvas_click(event):
    if not _canvas:
        return
    x, y = event.x, event.y

    # If match is over, check buttons
    if _game_over:
        r = _layout.get("play_again")
        if r and r[0] <= x <= r[2] and r[1] <= y <= r[3]:
            _restart_same_target()
            return
        r = _layout.get("exit_btn")
        if r and r[0] <= x <= r[2] and r[1] <= y <= r[3]:
            _on_back()
            return
        return

    # Normal play: click on board cells
    for idx in range(9):
        key = f"cell_{idx}"
        r = _layout.get(key)
        if r and r[0] <= x <= r[2] and r[1] <= y <= r[3]:
            _handle_player_move(idx)
            return

def _size_rects(w: int, h: int):
    pad = 20
    board_size = min(w - pad * 2, h - pad * 3 - 80)
    board_size = max(180, board_size)
    x0 = (w - board_size) // 2
    y0 = pad
    x1 = x0 + board_size
    y1 = y0 + board_size
    _layout["board"] = (x0, y0, x1, y1)

    cell_w = board_size / 3
    cell_h = board_size / 3
    for r in range(3):
        for c in range(3):
            idx = r * 3 + c
            cx0 = int(x0 + c * cell_w)
            cy0 = int(y0 + r * cell_h)
            cx1 = int(x0 + (c + 1) * cell_w)
            cy1 = int(y0 + (r + 1) * cell_h)
            _layout[f"cell_{idx}"] = (cx0, cy0, cx1, cy1)

    # Status text area
    status_y0 = y1 + 20
    status_y1 = status_y0 + 30
    _layout["status"] = (pad, status_y0, w - pad, status_y1)

    # Game over buttons (overlay)
    btn_h = 40
    btn_w = 150
    btn_y0 = status_y1 + 40
    btn_y1 = btn_y0 + btn_h
    gap = 20
    total_w = btn_w * 2 + gap
    start_x = (w - total_w) // 2
    _layout["play_again"] = (start_x, btn_y0, start_x + btn_w, btn_y1)
    _layout["exit_btn"] = (start_x + btn_w + gap, btn_y0, start_x + btn_w * 2 + gap, btn_y1)

def _draw_board():
    if not _canvas:
        return
    board_rect = _layout.get("board")
    if not board_rect:
        return
    x0, y0, x1, y1 = board_rect
    _canvas.create_rectangle(x0, y0, x1, y1, outline=BORDER, width=2, fill=BG)

    cell_w = (x1 - x0) / 3
    cell_h = (y1 - y0) / 3

    # Grid lines
    for i in range(1, 3):
        # vertical
        vx = x0 + i * cell_w
        _canvas.create_line(vx, y0, vx, y1, fill=GRID_COLOR, width=2)
        # horizontal
        hy = y0 + i * cell_h
        _canvas.create_line(x0, hy, x1, hy, fill=GRID_COLOR, width=2)

    # Marks
    for idx, mark in enumerate(_board):
        if not mark:
            continue
        r = idx // 3
        c = idx % 3
        cx = x0 + (c + 0.5) * cell_w
        cy = y0 + (r + 0.5) * cell_h
        color = X_COLOR if mark == "X" else O_COLOR
        _canvas.create_text(
            cx, cy, text=mark, fill=color,
            font=("Segoe UI", int(cell_w * 0.5), "bold")
        )

    # Winning line overlay
    if _winning_line:
        a, b, c = _winning_line
        ax, ay = _cell_center(a, x0, y0, cell_w, cell_h)
        cx2, cy2 = _cell_center(c, x0, y0, cell_w, cell_h)
        _canvas.create_line(ax, ay, cx2, cy2,
                            fill=WIN_LINE_COLOR, width=4, capstyle="round")

def _cell_center(idx: int, x0: float, y0: float, cell_w: float, cell_h: float) -> Tuple[float, float]:
    r = idx // 3
    c = idx % 3
    cx = x0 + (c + 0.5) * cell_w
    cy = y0 + (r + 0.5) * cell_h
    return cx, cy

def _draw_status():
    if not _canvas:
        return
    rect = _layout.get("status")
    if not rect:
        return
    x0, y0, x1, y1 = rect
    _canvas.create_text(
        (x0 + x1) // 2, (y0 + y1) // 2,
        text=_status_msg, fill=FG_DIM,
        font=("Segoe UI", 11)
    )

def _draw_button(rect: Tuple[int, int, int, int], label: str, primary: bool = True):
    x0, y0, x1, y1 = rect
    fill = ACCENT if primary else BORDER
    fg = "white" if primary else FG
    _canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=BORDER, width=1)
    _canvas.create_text(
        (x0 + x1) // 2, (y0 + y1) // 2,
        text=label, fill=fg,
        font=("Segoe UI", 11, "bold")
    )

def _redraw():
    if not _canvas:
        return
    _canvas.delete("all")
    _canvas.unbind("<Button-1>")
    _canvas.bind("<Button-1>", _on_canvas_click)

    w, h = _size()
    _size_rects(w, h)

    _draw_board()
    _draw_status()

    # Match over overlay
    if _game_over:
        _canvas.create_rectangle(0, 0, w, h, fill=PAUSE_DIM, stipple=PAUSE_STIPPLE, outline="")
        _canvas.create_text(
            w // 2, h // 2 - 40,
            text="Match Over", fill=FG,
            font=("Segoe UI", 18, "bold")
        )
        _canvas.create_text(
            w // 2, h // 2 - 14,
            text=f"Score: You {_you_score} — {_nulla_score} Nulla",
            fill=FG_DIM, font=("Segoe UI", 12)
        )
        _draw_button(_layout["play_again"], "Play Again", primary=True)
        _draw_button(_layout["exit_btn"], "Exit", primary=False)

# ============ Input handling ============

def _on_key_down(event):
    key = (event.keysym or "").lower()
    if key == "escape":
        _on_back()
    elif key == "r":
        if not _game_over:
            _reset_board()
    elif key in ("n",):
        if _game_over:
            _restart_same_target()

def _on_key_up(event):
    pass
