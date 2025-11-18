# nulla_source\scripts\nulla_game_bounce.py
# Bounce overlay for Nulla's chat window — import-safe (no GUI at import time)

from __future__ import annotations

import os
import math
import random
import threading
import shutil
import subprocess
import sys
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

_move_left: bool = False
_move_right: bool = False

# Playfield + visuals
BG = "#0f1115"
PANEL = "#161a20"
BORDER = "#2a2f3a"
FG = "#e6e6e6"
FG_DIM = "#9aa4b2"
ACCENT = "#3b82f6"
PAUSE_DIM = "#000000"
PAUSE_STIPPLE = "gray50"

BALL_COLOR = "#f97316"      # orange
PADDLE_COLOR = "#22c55e"    # green
WALL_COLOR = "#3b82f6"      # blue accent

PLAYFIELD_MARGIN_X = 40
PLAYFIELD_MARGIN_TOP = 40
PLAYFIELD_MARGIN_BOTTOM = 70

BALL_RADIUS = 9
BALL_DIAMETER = BALL_RADIUS * 2

# Paddle width expressed in "ball units" (2x ball diameter up to 4x).
PADDLE_UNITS_MIN = 2
PADDLE_UNITS_MAX = 4
PADDLE_UNITS_INITIAL = 4

PADDLE_HEIGHT = 12.0
PADDLE_SPEED = 11.0  # px per tick

TICK_MS = 16  # ~60 FPS

# Random mechanics
BASE_BALL_SPEED = 6.0
MAX_BALL_SPEED = 14.0
SPEED_RAMP_MULT = 1.02
SPEED_BURST_CHANCE = 0.18
SPEED_BURST_MULT = 1.12

# Game state
_ball_pos: Tuple[float, float] = (0.0, 0.0)
_ball_vel: Tuple[float, float] = (0.0, 0.0)
_ball_speed: float = BASE_BALL_SPEED

_paddle_x: float = 0.0
_paddle_width: float = BALL_DIAMETER * PADDLE_UNITS_INITIAL
_paddle_units: int = PADDLE_UNITS_INITIAL

_score: int = 0

# =========================================================
# Asset resolution / SFX (portable, same style as TTT)
# =========================================================

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

SFX_BOUNCE_PATH = _resolve_asset("bounce1.wav")


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


def _play_bounce_sfx():
    _play_wav(SFX_BOUNCE_PATH)


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

        # fallback pools if request fails
        self.fallback: Dict[str, List[str]] = {
            "load": [
                "Bounce ready. Left/Right or A/D to move. P to pause.",
                "Game loaded. Move the paddle with Left/Right or A/D. P to pause.",
            ],
            "restart": [
                "Restarting Bounce with a fresh ball.",
                "New round of Bounce. Let’s see how long you last this time.",
            ],
            "exit": [
                "Closed the Bounce overlay.",
                "Bounce closed. We can play again anytime.",
            ],
            "game_over": [
                "Rally over. Final score: {score}.",
                "You missed that one. Final score: {score}.",
            ],
        }

    def _sys_prompt(self, kind: str, ctx: Dict[str, Any]) -> str:
        base = (
            "You are Nulla: warm, calm, supportive, friendly, and kind.\n"
            "Write EXACTLY ONE short line (<= 120 chars). No emojis. Never mention AI or prompts.\n"
            "Keep it wholesome and encouraging. Vary the wording naturally each time.\n"
        )
        if kind == "load":
            base += (
                "Event: game_load (bounce). Briefly state the controls: "
                "player moves a paddle with Left/Right or A/D, P to pause.\n"
            )
        elif kind == "restart":
            base += "Event: restart. Say the game is restarting in a friendly way.\n"
        elif kind == "exit":
            base += (
                "Event: exit. Confirm the Bounce game overlay closed. "
                "Do NOT say goodbye or imply leaving the app.\n"
            )
        elif kind == "game_over":
            base += (
                "Event: game_over. Encourage the player after they miss the ball. "
                "You MUST end the sentence with exactly: "
                f"'Final score: {ctx.get('score', 0)}.'\n"
            )
        return base

    def _user_prompt(self, kind: str, ctx: Dict[str, Any]) -> str:
        parts = [f"Kind={kind}"]
        if "score" in ctx:
            parts.append(f"Score={ctx['score']}")
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
            r = self._requests.post(
                f"{self.base}/v1/chat/completions",
                json=payload, headers=headers, timeout=30,
            )
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
# Public API expected by GameManager
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

    title = tk.Label(_header, text="Bounce", bg=PANEL, fg=FG, font=("Segoe UI", 11, "bold"))
    title.pack(side="left", padx=8)

    score_lbl = tk.Label(_header, text="Score: 0", bg=PANEL, fg=FG_DIM, font=("Consolas", 11))
    score_lbl.pack(side="right", padx=12)
    _header.score_lbl = score_lbl  # type: ignore[attr-defined]

    _canvas = tk.Canvas(_overlay, bg=BG, highlightthickness=0)
    _canvas.pack(side="top", fill="both", expand=True)

    _overlay.bind("<KeyPress>", _on_key_down)
    _overlay.bind("<KeyRelease>", _on_key_up)
    _overlay.bind("<FocusIn>", lambda e: _overlay.focus_set())
    _overlay.focus_set()
    _overlay.update_idletasks()

    _canvas.bind("<Configure>", _on_canvas_resize)

    _init_game()

    # Defer the spoken controls line so "Launching Bounce…" prints first.
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
        w, h = _get_canvas_size()
        _canvas.dim = _canvas.create_rectangle(  # type: ignore[attr-defined]
            0, 0, w, h, fill=PAUSE_DIM, outline="", stipple=PAUSE_STIPPLE
        )

    _confirm_sheet = tk.Frame(_overlay, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
    _confirm_sheet.place(relx=0.5, rely=0.5, anchor="center")

    lbl = tk.Label(_confirm_sheet, text=text, bg=PANEL, fg=FG, font=("Segoe UI", 11, "bold"))
    lbl.pack(padx=16, pady=(16, 8))

    btns = tk.Frame(_confirm_sheet, bg=PANEL)
    btns.pack(fill="x", padx=16, pady=(0, 16))

    yes = tk.Button(
        btns, text="Yes", relief="flat", bg=ACCENT, fg="white",
        activebackground=ACCENT, activeforeground="white",
        command=yes_cb
    )
    yes.pack(side="left", padx=(0, 8))

    no = tk.Button(
        btns, text="No", relief="flat", bg=BORDER, fg=FG,
        activebackground=BORDER, activeforeground=FG,
        command=no_cb
    )
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
    global _active, _paused, _game_over, _move_left, _move_right

    # resume idle after leaving the game
    try:
        if callable(_hooks.get("idle_block_pop")):
            _hooks["idle_block_pop"]()
    except Exception:
        pass

    _cancel_loop()
    _hide_confirm_sheet()
    _clear_toast()

    try:
        if _overlay and _overlay.winfo_exists():
            _overlay.unbind("<KeyPress>")
            _overlay.unbind("<KeyRelease>")
            _overlay.unbind("<FocusIn>")
            _overlay.place_forget()
            _overlay.destroy()
    except Exception:
        pass

    _restore_chat_input()

    _active = False
    _paused = False
    _game_over = False
    _move_left = False
    _move_right = False

    # Make XTTS read the neutral close line
    _say_ai("exit", speak=True)


# ============ Game core ============

def _get_canvas_size() -> Tuple[int, int]:
    if not _canvas:
        return (640, 480)
    w = max(240, _canvas.winfo_width())
    h = max(260, _canvas.winfo_height())
    return (w, h)


def _playfield_bounds(w: int, h: int) -> Tuple[float, float, float, float]:
    left = PLAYFIELD_MARGIN_X
    right = max(left + 200, w - PLAYFIELD_MARGIN_X)
    top = PLAYFIELD_MARGIN_TOP
    bottom = max(top + 200, h - PLAYFIELD_MARGIN_BOTTOM)
    return float(left), float(top), float(right), float(bottom)


def _current_paddle_y() -> float:
    w, h = _get_canvas_size()
    _, _, _, bottom = _playfield_bounds(w, h)
    return bottom - 24.0


def _normalized(dx: float, dy: float) -> Tuple[float, float]:
    length = math.hypot(dx, dy) or 1.0
    return dx / length, dy / length


def _scaled_velocity(dx: float, dy: float, speed: float) -> Tuple[float, float]:
    nx, ny = _normalized(dx, dy)
    return nx * speed, ny * speed


def _init_game():
    global _ball_pos, _ball_vel, _ball_speed
    global _paddle_x, _paddle_width, _paddle_units
    global _score, _game_over, _paused

    _score = 0
    _ball_speed = BASE_BALL_SPEED
    _paddle_units = PADDLE_UNITS_INITIAL
    _paddle_width = BALL_DIAMETER * float(_paddle_units)

    _game_over = False
    _paused = False

    w, h = _get_canvas_size()
    left, top, right, bottom = _playfield_bounds(w, h)

    _paddle_x = (left + right) / 2.0
    paddle_y = _current_paddle_y()

    ball_x = _paddle_x
    ball_y = paddle_y - BALL_RADIUS - 10.0
    _ball_pos = (ball_x, ball_y)

    angle = random.uniform(-0.35, 0.35)  # small horizontal angle
    dx = math.sin(angle)
    dy = -math.cos(angle)  # upward
    _ball_vel = _scaled_velocity(dx, dy, _ball_speed)

    _update_score_label()
    _redraw()


def _update_score_label():
    if _header and hasattr(_header, "score_lbl"):
        try:
            _header.score_lbl.config(text=f"Score: {_score}")  # type: ignore[attr-defined]
        except Exception:
            pass


def _on_canvas_resize(event):
    _redraw()


def _resume_loop():
    global _tick_job
    _cancel_loop()
    if _canvas:
        _tick_job = _canvas.after(TICK_MS, _tick)
    else:
        _tick_job = None


def _cancel_loop():
    global _tick_job
    if _canvas and _tick_job:
        try:
            _canvas.after_cancel(_tick_job)
        except Exception:
            pass
    _tick_job = None


def _tick():
    if not _canvas:
        return
    if not _paused and not _game_over:
        _step()
        _redraw()
    _resume_loop()


def _update_paddle():
    global _paddle_x
    if _move_left == _move_right:
        return
    direction = -1 if _move_left else 1

    w, h = _get_canvas_size()
    left, top, right, bottom = _playfield_bounds(w, h)
    half = _paddle_width / 2.0

    _paddle_x += direction * PADDLE_SPEED
    if _paddle_x - half < left:
        _paddle_x = left + half
    if _paddle_x + half > right:
        _paddle_x = right - half


def _step():
    global _ball_pos, _ball_vel, _ball_speed

    _update_paddle()

    w, h = _get_canvas_size()
    left, top, right, bottom = _playfield_bounds(w, h)

    x, y = _ball_pos
    vx, vy = _ball_vel
    prev_x, prev_y = x, y

    x += vx
    y += vy

    # Top wall collision (the "wall")
    if y - BALL_RADIUS <= top:
        y = top + BALL_RADIUS
        vy = abs(vy)
        _play_bounce_sfx()

    # Side walls
    if x - BALL_RADIUS <= left:
        x = left + BALL_RADIUS
        vx = abs(vx)
        _play_bounce_sfx()
    elif x + BALL_RADIUS >= right:
        x = right - BALL_RADIUS
        vx = -abs(vx)
        _play_bounce_sfx()

    # Paddle collision (only when moving downward)
    hit_paddle = False
    paddle_y = _current_paddle_y()
    if vy > 0:
        paddle_top = paddle_y - PADDLE_HEIGHT / 2.0
        half = _paddle_width / 2.0

        # Crosses the paddle line this tick
        if (prev_y + BALL_RADIUS <= paddle_top) and (y + BALL_RADIUS >= paddle_top):
            if (x >= _paddle_x - half) and (x <= _paddle_x + half):
                y = paddle_top - BALL_RADIUS
                vy = -abs(vy)

                # Angle tweak based on where it hits the paddle
                rel = (x - _paddle_x) / max(half, 1.0)
                rel = max(-1.0, min(1.0, rel))
                vx += rel * 1.5
                hit_paddle = True

    if hit_paddle:
        _on_paddle_hit()
        _play_bounce_sfx()
    else:
        # Check for miss: ball falls below the playfield bottom
        if y - BALL_RADIUS > bottom + 10:
            _ball_pos = (x, y)
            _ball_vel = (vx, vy)
            _handle_game_over()
            return

    # Normalize velocity to current speed
    vx, vy = _scaled_velocity(vx, vy, _ball_speed)
    _ball_pos = (x, y)
    _ball_vel = (vx, vy)


def _on_paddle_hit():
    global _score, _ball_speed, _paddle_width, _paddle_units

    _score += 1
    _update_score_label()
    _toast(f"+1 | Score: {_score}", ms=900)

    # Base ramp-up
    _ball_speed = min(MAX_BALL_SPEED, _ball_speed * SPEED_RAMP_MULT)

    # Random burst
    if random.random() < SPEED_BURST_CHANCE:
        _ball_speed = min(MAX_BALL_SPEED, _ball_speed * SPEED_BURST_MULT)

    # Paddle size mutation: pick a new size in [2..4] ball widths, different from current.
    old_units = _paddle_units
    choices = [u for u in range(PADDLE_UNITS_MIN, PADDLE_UNITS_MAX + 1) if u != old_units]
    if choices:
        _paddle_units = random.choice(choices)
        _paddle_width = BALL_DIAMETER * float(_paddle_units)
        if _paddle_units > old_units:
            _toast("Paddle size ↑", ms=700)
        else:
            _toast("Paddle size ↓", ms=700)


def _handle_game_over():
    global _game_over
    if _game_over:
        return
    _game_over = True
    _say_ai("game_over", score=_score, speak=True)


def _restart():
    """Reset the board quietly (no restart announcement)."""
    global _paused
    _init_game()
    _paused = False


def _toggle_pause():
    global _paused
    _paused = not _paused
    _redraw()


# ============ Drawing ============

def _redraw():
    if not _canvas:
        return

    _canvas.delete("all")

    w, h = _get_canvas_size()
    left, top, right, bottom = _playfield_bounds(w, h)

    # Playfield border
    _canvas.create_rectangle(left - 2, top - 2, right + 2, bottom + 2,
                             outline=BORDER, width=2)

    # Top wall strip
    _canvas.create_rectangle(left, top - 6, right, top,
                             fill=WALL_COLOR, outline="")

    # Paddle
    paddle_y = _current_paddle_y()
    half = _paddle_width / 2.0
    paddle_top = paddle_y - PADDLE_HEIGHT / 2.0
    paddle_bottom = paddle_y + PADDLE_HEIGHT / 2.0
    _canvas.create_rectangle(
        _paddle_x - half, paddle_top,
        _paddle_x + half, paddle_bottom,
        fill=PADDLE_COLOR, outline=BORDER
    )

    # Ball
    x, y = _ball_pos
    _canvas.create_oval(
        x - BALL_RADIUS, y - BALL_RADIUS,
        x + BALL_RADIUS, y + BALL_RADIUS,
        fill=BALL_COLOR, outline=""
    )

    if _paused and not _game_over:
        _draw_pause_overlay()
    if _game_over:
        _draw_game_over_overlay()


def _draw_pause_overlay():
    if not _canvas:
        return
    w, h = _get_canvas_size()
    _canvas.create_rectangle(0, 0, w, h,
                             fill=PAUSE_DIM, stipple=PAUSE_STIPPLE, outline="")
    _canvas.create_text(w // 2, h // 2 - 8,
                        text="Paused", fill=FG, font=("Segoe UI", 16, "bold"))
    _canvas.create_text(w // 2, h // 2 + 14,
                        text="Press P to resume", fill=FG_DIM, font=("Segoe UI", 11))


def _draw_game_over_overlay():
    if not _canvas:
        return
    w, h = _get_canvas_size()
    _canvas.create_rectangle(0, 0, w, h,
                             fill=PAUSE_DIM, stipple=PAUSE_STIPPLE, outline="")

    _canvas.create_text(
        w // 2, h // 2 - 16,
        text="Game Over", fill=FG,
        font=("Segoe UI", 18, "bold")
    )
    _canvas.create_text(
        w // 2, h // 2 + 8,
        text=f"Score: {_score}",
        fill=FG_DIM, font=("Segoe UI", 12)
    )

    _canvas.play_again = _canvas.create_text(  # type: ignore[attr-defined]
        w // 2 - 60, h // 2 + 40,
        text="[ Play Again ]",
        fill=ACCENT, font=("Segoe UI", 11, "bold")
    )
    _canvas.exit_btn = _canvas.create_text(  # type: ignore[attr-defined]
        w // 2 + 64, h // 2 + 40,
        text="[ Exit ]",
        fill=FG, font=("Segoe UI", 11, "bold")
    )
    _canvas.tag_bind(_canvas.play_again, "<Button-1>", lambda e: _restart())   # type: ignore[attr-defined]
    _canvas.tag_bind(_canvas.exit_btn, "<Button-1>", lambda e: _on_back())     # type: ignore[attr-defined]


# ============ Input handling ============

def _on_key_down(event):
    global _move_left, _move_right
    key = (event.keysym or "").lower()
    if key in ("left", "a"):
        _move_left = True
    elif key in ("right", "d"):
        _move_right = True
    elif key in ("p",):
        _toggle_pause()
    elif key in ("escape",):
        _on_back()
    elif key == "return" and _game_over:
        _restart()


def _on_key_up(event):
    global _move_left, _move_right
    key = (event.keysym or "").lower()
    if key in ("left", "a"):
        _move_left = False
    elif key in ("right", "d"):
        _move_right = False


# =========================================================
# End of module
# =========================================================
