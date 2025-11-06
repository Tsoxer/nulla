# nulla_source\scripts\nulla_game_runner.py
# Endless Runner overlay for Nulla's chat window — import-safe (no GUI at import time)

from __future__ import annotations

import os
import random
import time
import threading
import tkinter as tk
from queue import Queue
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
_spawn_job: Optional[str] = None
_toast_job: Optional[str] = None

# Theme (match your dark UI)
BG = "#0f1115"
PANEL = "#161a20"
BORDER = "#2a2f3a"
FG = "#e6e6e6"
FG_DIM = "#9aa4b2"
ACCENT = "#3b82f6"
PAUSE_DIM = "#000000"
PAUSE_STIPPLE = "gray50"

# Runner visuals
GROUND_COLOR = "#222730"
PLAYER_COLOR = "#86efac"
OBST_COLOR = "#ef4444"

# Canvas + world units
TICK_MS = 16                     # ~60 FPS
GROUND_H = 56                    # px height reserved for ground area
PLAYER_W, PLAYER_H = 28, 36      # player size
OBST_MIN_W, OBST_MAX_W = 20, 42
OBST_MIN_H, OBST_MAX_H = 26, 60
OBST_GAP_MIN_MS, OBST_GAP_MAX_MS = 800, 1400

# Physics
GRAVITY = 0.9
JUMP_VELOCITY = -14.0

# Input forgiveness (for zero-delay feel)
JUMP_BUFFER_MS = 90              # press slightly early; will fire on landing
COYOTE_MS = 110                  # tiny grace after stepping off a ledge

# Speed curve
BASE_SPEED = 6.0                 # px per tick
SPEED_PER_POINT = 0.45
MAX_SPEED = 16.0

# Horizontal control (works in air AND on ground)
def _move_step() -> float:
    return min(12.0, 0.65 * _speed() + 2.5)

# Score
_score: int = 0

# Player state (in px)
_player_x: float = 0.0
_player_y: float = 0.0
_player_vy: float = 0.0

# Horizontal input state
_left_down: bool = False
_right_down: bool = False

# Obstacles list
# each: {"x": float, "y": float, "w": int, "h": int, "passed": bool}
_obstacles: List[Dict[str, Any]] = []

# Jump buffer / coyote timers (ms since start)
_jump_buffer_until_ms: float = 0.0
_coyote_until_ms: float = 0.0

# =========================================================
# Async AI QUIPS via local OpenAI-compatible endpoint
# (runs off the Tk thread so the game never stutters)
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
                "Runner ready! Press Space to jump. Avoid red blocks.",
                "All set. Auto-run forward. Space to jump over obstacles.",
            ],
            "restart": [
                "Restarting Runner…",
                "Fresh track—let’s go again.",
            ],
            "exit": [
                "Runner closed.",
                "Closed the game overlay.",
            ],
            "game_over": [
                "Tough hit. Final score: {score}.",
                "Good effort out there. Final score: {score}.",
            ],
        }

    def _sys_prompt(self, kind: str, ctx: Dict[str, Any]) -> str:
        base = (
            "You are Nulla: warm, calm, supportive, friendly, and kind.\n"
            "Write EXACTLY ONE short line (<= 120 chars). No emojis. Never mention AI or prompts.\n"
            "Keep it wholesome and encouraging. Vary the wording naturally each time.\n"
        )
        if kind == "load":
            base += "Event: game_load. Briefly remind: 'Space to jump.' Keep it short.\n"
        elif kind == "restart":
            base += "Event: restart. Short and friendly.\n"
        elif kind == "exit":
            base += "Event: exit. Confirm the game overlay closed. Do NOT say goodbye or imply leaving the app.\n"
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

# Back-compat wrapper
def _say_ai(kind: str, speak: bool = False, **kw):
    _queue_ai(kind, speak=speak, **kw)

# =========================================================
# Public API expected by GameManager
# =========================================================

def mount_overlay(story_win: tk.Toplevel, hooks: Dict[str, Any]) -> None:
    """Required entrypoint. Creates the overlay and starts the game loop."""
    global _overlay, _header, _canvas, _story_win, _hooks, _active, _paused, _game_over

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

    # Header
    _header = tk.Frame(_overlay, bg=PANEL, height=44, highlightthickness=0)
    _header.pack(side="top", fill="x")

    back_btn = tk.Button(
        _header, text="← Back", relief="flat",
        bg=BORDER, fg=FG, activebackground=BORDER, activeforeground=FG,
        command=_on_back
    )
    back_btn.pack(side="left", padx=8, pady=8)

    title = tk.Label(_header, text="Runner", bg=PANEL, fg=FG, font=("Segoe UI", 11, "bold"))
    title.pack(side="left", padx=8)

    score_lbl = tk.Label(_header, text="Score: 0", bg=PANEL, fg=FG_DIM, font=("Consolas", 11))
    score_lbl.pack(side="right", padx=12)
    _header.score_lbl = score_lbl  # type: ignore[attr-defined]

    # Canvas
    global _canvas
    _canvas = tk.Canvas(_overlay, bg=BG, highlightthickness=0)
    _canvas.pack(side="top", fill="both", expand=True)

    # Key handling — bind ONLY on the overlay (no global bind_all)
    _overlay.bind("<KeyPress>", _on_key_down)
    _overlay.bind("<KeyRelease>", _on_key_up)
    _overlay.bind("<FocusIn>", lambda e: (_overlay.focus_set(), _reset_lr()))
    _overlay.focus_set()
    _overlay.update_idletasks()

    _canvas.bind("<Configure>", _on_canvas_resize)

    _init_game()
    # Defer spoken controls so it prints after "Launching Runner…"
    if _overlay:
        _overlay.after(200, lambda: _say_ai("load", speak=True))

    _active = True
    _paused = False
    _game_over = False
    _resume_loop()
    _schedule_spawn()

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

def _now_ms() -> float:
    return time.perf_counter() * 1000.0

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

def _unbind_keys():
    # Belt-and-suspenders: ensure no global bindings remain
    try:
        if _overlay and _overlay.winfo_exists():
            _overlay.unbind("<KeyPress>")
            _overlay.unbind("<KeyRelease>")
            _overlay.unbind("<FocusIn>")
    except Exception:
        pass

def _teardown():
    # Say exit BEFORE teardown so XTTS reads it reliably
    _say_ai("exit", speak=True)

    # resume idle after leaving the game
    try:
        if callable(_hooks.get("idle_block_pop")):
            _hooks["idle_block_pop"]()
    except Exception:
        pass

    global _active, _paused, _game_over
    _cancel_loop()
    _cancel_spawn()
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

    _active = False
    _paused = False
    _game_over = False

# ============ Game core ============

def _init_game():
    global _score, _player_x, _player_y, _player_vy
    global _obstacles, _game_over, _jump_buffer_until_ms, _coyote_until_ms
    global _left_down, _right_down
    _score = 0
    _game_over = False
    _obstacles = []
    _jump_buffer_until_ms = 0.0
    _coyote_until_ms = 0.0
    _left_down = False
    _right_down = False

    w, _ = _size()
    _player_x = int(w * 0.18)
    _player_y = _ground_y() - PLAYER_H
    _player_vy = 0.0

    _update_score_label()
    _redraw()

def _size() -> Tuple[int, int]:
    if not _canvas:
        return (800, 600)
    return max(200, _canvas.winfo_width()), max(200, _canvas.winfo_height())

def _ground_y() -> int:
    _, h = _size()
    return h - GROUND_H

def _x_bounds() -> Tuple[int, int]:
    w, _ = _size()
    min_x = int(w * 0.08)
    max_x = int(w * 0.86) - PLAYER_W
    if max_x < min_x:
        max_x = min_x
    return (min_x, max_x)

def _speed() -> float:
    return min(MAX_SPEED, BASE_SPEED + _score * SPEED_PER_POINT)

def _schedule_spawn():
    global _spawn_job
    if not _canvas or _game_over or _paused:
        return
    gap_min = max(450, OBST_GAP_MIN_MS - int(_score * 20))
    gap_max = max(gap_min + 200, OBST_GAP_MAX_MS - int(_score * 25))
    delay = random.randint(gap_min, gap_max)
    _spawn_job = _canvas.after(delay, _spawn_obstacle)

def _cancel_spawn():
    global _spawn_job
    if _canvas and _spawn_job:
        try:
            _canvas.after_cancel(_spawn_job)
        except Exception:
            pass
    _spawn_job = None

def _spawn_obstacle():
    if not _canvas or _game_over or _paused:
        _schedule_spawn()
        return
    w, _ = _size()
    gw = random.randint(OBST_MIN_W, OBST_MAX_W)
    gh = random.randint(OBST_MIN_H, OBST_MAX_H)
    ob = {
        "x": float(w + gw),
        "y": float(_ground_y() - gh),
        "w": gw,
        "h": gh,
        "passed": False,
    }
    _obstacles.append(ob)
    _schedule_spawn()

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

def _is_grounded() -> bool:
    return abs((_player_y + PLAYER_H) - _ground_y()) < 0.001

def _consume_buffer_if_possible(now_ms: float):
    global _player_vy, _player_y, _jump_buffer_until_ms
    if _jump_buffer_until_ms <= 0:
        return
    if now_ms <= _jump_buffer_until_ms and (_is_grounded() or now_ms <= _coyote_until_ms):
        _player_vy = JUMP_VELOCITY
        if _player_y + PLAYER_H >= _ground_y():
            _player_y = _ground_y() - PLAYER_H - 0.01
        _jump_buffer_until_ms = 0.0

def _apply_horizontal():
    global _player_x
    if _left_down ^ _right_down:
        step = _move_step()
        if _left_down:
            _player_x -= step
        else:
            _player_x += step
        min_x, max_x = _x_bounds()
        if _player_x < min_x:
            _player_x = float(min_x)
        elif _player_x > max_x:
            _player_x = float(max_x)

def _tick():
    if not _canvas:
        return

    if _paused:
        _redraw()
        _resume_loop()
        return

    if _game_over:
        _resume_loop()
        return

    _apply_physics()
    _advance_obstacles()
    _score_and_collide()
    _redraw()
    _resume_loop()

def _apply_physics():
    global _player_y, _player_vy, _coyote_until_ms
    now = _now_ms()
    was_grounded = _is_grounded()

    _player_vy += GRAVITY
    _player_y += _player_vy

    gy = _ground_y()
    if _player_y + PLAYER_H >= gy:
        _player_y = gy - PLAYER_H
        _player_vy = 0.0
        if not was_grounded:
            _consume_buffer_if_possible(now)
    else:
        if was_grounded:
            _coyote_until_ms = now + COYOTE_MS
        _consume_buffer_if_possible(now)

    _apply_horizontal()

def _advance_obstacles():
    spd = _speed()
    for ob in _obstacles:
        ob["x"] -= spd
    _obstacles[:] = [ob for ob in _obstacles if ob["x"] + ob["w"] > -40]

def _score_and_collide():
    global _score, _game_over
    px0, py0, px1, py1 = _player_rect()

    for ob in _obstacles:
        ox0, oy0, ox1, oy1 = ob["x"], ob["y"], ob["x"] + ob["w"], ob["y"] + ob["h"]
        if not _game_over and (px0 < ox1 and px1 > ox0 and py0 < oy1 and py1 > oy0):
            _handle_game_over()
            return

        if not ob["passed"] and (ox1 < px0):
            ob["passed"] = True
            _score += 1
            _update_score_label()
            _toast(f"+1  |  Score: {_score}", ms=900)

def _player_rect() -> Tuple[float, float, float, float]:
    return _player_x, _player_y, _player_x + PLAYER_W, _player_y + PLAYER_H

def _handle_game_over():
    global _game_over
    if _game_over:
        return
    _game_over = True
    _say_ai("game_over", score=_score, speak=True)

def _restart():
    _cancel_spawn()
    _init_game()
    _paused = False
    _resume_loop()
    _schedule_spawn()
    _say_ai("restart", speak=True)

# ============ Drawing ============

def _on_canvas_resize(event):
    _redraw()

def _redraw():
    if not _canvas:
        return
    _canvas.delete("all")
    w, h = _size()
    gy = _ground_y()

    # ground line
    _canvas.create_rectangle(0, gy, w, h, fill=GROUND_COLOR, outline="")

    # obstacles
    for ob in _obstacles:
        _canvas.create_rectangle(int(ob["x"]), int(ob["y"]),
                                 int(ob["x"] + ob["w"]), int(ob["y"] + ob["h"]),
                                 fill=OBST_COLOR, outline=BG)

    # player
    px0, py0, px1, py1 = _player_rect()
    _canvas.create_rectangle(int(px0), int(py0), int(px1), int(py1), fill=PLAYER_COLOR, outline=BG)

    # overlays
    if _paused and not _game_over:
        _draw_pause_overlay()
    if _game_over:
        _draw_game_over_overlay()

def _draw_pause_overlay():
    if not _canvas:
        return
    w, h = _size()
    _canvas.create_rectangle(0, 0, w, h, fill=PAUSE_DIM, stipple=PAUSE_STIPPLE, outline="")
    _canvas.create_text(w // 2, h // 2 - 8, text="Paused", fill=FG, font=("Segoe UI", 16, "bold"))
    _canvas.create_text(w // 2, h // 2 + 14, text="Press P to resume", fill=FG_DIM, font=("Segoe UI", 11))

def _draw_game_over_overlay():
    if not _canvas:
        return
    w, h = _size()
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

def _update_score_label():
    if _header and hasattr(_header, "score_lbl"):
        try:
            _header.score_lbl.config(text=f"Score: {_score}")  # type: ignore[attr-defined]
        except Exception:
            pass

# ============ Input handling ============

def _on_key_down(event):
    key = (event.keysym or "").lower()
    if key == "escape":
        _on_back()
    elif key in ("p",):
        _toggle_pause()
    elif key in ("space",):
        _buffer_jump_and_try_consume()
    elif key in ("a", "left"):
        _set_lr(left=True, down=True)
    elif key in ("d", "right"):
        _set_lr(left=False, down=True)
    elif key in ("return", "enter") and _game_over:
        _restart()

def _on_key_up(event):
    key = (event.keysym or "").lower()
    if key in ("a", "left"):
        _set_lr(left=True, down=False)
    elif key in ("d", "right"):
        _set_lr(left=False, down=False)

def _toggle_pause():
    global _paused
    _paused = not _paused
    if _paused:
        _cancel_spawn()
    else:
        _schedule_spawn()
    _redraw()

def _set_lr(*, left: bool, down: bool):
    global _left_down, _right_down
    if left:
        _left_down = down
    else:
        _right_down = down

def _reset_lr():
    global _left_down, _right_down
    _left_down = False
    _right_down = False

def _buffer_jump_and_try_consume():
    global _jump_buffer_until_ms
    now = _now_ms()
    _jump_buffer_until_ms = now + JUMP_BUFFER_MS
    _consume_buffer_if_possible(now)

# =========================================================
# End of module
# =========================================================
