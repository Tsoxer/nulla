# nulla_source\scripts\nulla_window.py
import os, threading, time, queue, re, tkinter as tk
from tkinter import ttk

# --------- LAZY ENGINE LOADER (avoid circular import) ----------
_engine_mod = None
def NC():
    """
    Lazy import of the engine module (nulla_chat).
    Avoids circular import crash when nulla_chat imports nulla_window.
    """
    global _engine_mod
    if _engine_mod is None:
        import importlib
        _engine_mod = importlib.import_module("nulla_chat")
    return _engine_mod

# --------- GAMES (optional) ----------
try:
    from nulla_list_game import GameManager
except Exception:
    GameManager = None  # games are optional

APP_TITLE = "Nulla - Alpha v0.0.9"

# ---- portrait placement awareness ----
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR    = os.path.abspath(os.path.join(_SCRIPTS_DIR, ".."))

PORTRAIT_TITLE       = "Nulla • Portrait"
PORTRAIT_IMAGE_PATH  = os.path.join(_BASE_DIR, "assets", "Nulla.png")
PORTRAIT_SCALE       = 0.35
CHAT_W, CHAT_H       = 700, 520
GAP_PX               = 16

def _geo_near_portrait(root: tk.Tk, w: int, h: int) -> str:
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()

    # 1) Try to read live portrait window rect
    try:
        import ctypes as ct
        from ctypes import wintypes as wt
        user32 = ct.windll.user32
        FindWindowW   = user32.FindWindowW
        GetWindowRect = user32.GetWindowRect

        hwnd = FindWindowW(None, PORTRAIT_TITLE)
        if hwnd:
            rect = (wt.RECT)()
            if GetWindowRect(hwnd, ct.byref(rect)):
                px, py, pr, pb = rect.left, rect.top, rect.right, rect.bottom
                pw, ph = pr - px, pb - py
                x = max(0, min(sw - w, pr + GAP_PX))
                y = max(0, min(sh - h, py + (ph - h) // 2))
                return f"{w}x{h}+{x}+{y}"
    except Exception:
        pass

    # 2) Estimate portrait location using its image + scale (center-left scheme)
    try:
        if os.path.isfile(PORTRAIT_IMAGE_PATH):
            tmp = tk.PhotoImage(file=PORTRAIT_IMAGE_PATH)
            pw = max(1, int(round(tmp.width()  * PORTRAIT_SCALE)))
            ph = max(1, int(round(tmp.height() * PORTRAIT_SCALE)))
            px = int(max(0, (sw * 0.25) - (pw / 2)))
            py = int(max(0, (sh * 0.5)  - (ph / 2)))
            x  = max(0, min(sw - w, px + pw + GAP_PX))
            y  = max(0, min(sh - h, py + (ph - h) // 2))
            return f"{w}x{h}+{x}+{y}"
    except Exception:
        pass

    # 3) Fallback
    x = int(sw * 0.35)
    y = int(max(0, (sh * 0.5) - (h / 2)))
    x = max(0, min(sw - w, x))
    y = max(0, min(sh - h, y))
    return f"{w}x{h}+{x}+{y}"

# ---- theme ----
BG       = "#13191E"
FG       = "#D0D0D0"
FG_META  = "#9A9A9A"
ENTRY_BG = "#101010"
BTN_BG   = "#1A1A1A"
BORDER   = "#202020"
MONO     = ("Consolas", 10)

YOU_COLOR   = "#55FF55"
NULLA_COLOR = "#FF55FF"
GAME_COLOR  = "#87CEFA"

UI_FLUSH_MS = 16  # ~60 FPS

# --- Instant TTS kick (same as v0.0.7 timing) ---
KICK_MS = 80
KICK_MIN_CHARS = 12
KICK_MAX_CHARS = 44

# --- UI sanitizer ---
UI_ASCII_ONLY = True
SMART_MAP = {
    "\u2018": "'", "\u2019": "'",
    "\u201C": '"', "\u201D": '"',
    "\u2013": "-", "\u2014": "-",
    "\u2026": "...",
    "\u00A0": " ",
}

def sanitize_for_ui(s: str) -> str:
    try:
        s = NC()._unmojibake(s)
    except Exception:
        pass
    try:
        s = NC().EMOJI_RE.sub("", s)
    except Exception:
        pass
    s = (s.replace("\u200d", "")
         .replace("\u200b", "")
         .replace("\u200c", "")
         .replace("\ufe0e", "")
         .replace("\ufe0f", ""))
    for k, v in SMART_MAP.items():
        s = s.replace(k, v)
    if UI_ASCII_ONLY:
        s = s.encode("ascii", errors="ignore").decode("ascii")
    return s

# --- STRICT TTS CLEANER (v0.0.7 style) ---
_PUNCT_BUNCH = re.compile(r"([.!?]){2,}")
_ONLY_PUNCT  = re.compile(r"^[\s\.\,\!\?\-\(\)\[\]\{\}:;\"']+$")
_BAD_CHARS   = re.compile(r"[`~^*_\\|<>{}\[\]$#@+=]+")

def clean_for_tts_strict(text: str) -> str | None:
    """
    v0.0.7 behavior: sanitize + strip noise + ensure terminal punctuation.
    """
    s = NC().sanitize_for_tts(text)
    s = _BAD_CHARS.sub("", s)
    s = " ".join(s.split())
    s = _PUNCT_BUNCH.sub(r"\1", s)
    s = s.strip(" \t\r\n.,!?") if _ONLY_PUNCT.match(s or "") else s.strip()
    if not s or not re.search(r"[A-Za-z0-9]", s):
        return None
    if s[-1] not in ".!?":
        s += "."
    return s

def _now_str():
    try:
        return time.strftime("%#I:%M:%S %p")
    except Exception:
        s = time.strftime("%I:%M:%S %p")
        return s.lstrip("0") if s.startswith("0") else s

def _stop_audio_immediately():
    try:
        import winsound
        winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception:
        pass


class NullaChatWindow(tk.Tk):
    def __init__(self):
        super().__init__()

        # ----- window chrome -----
        self.title(APP_TITLE)
        self.configure(bg=BG)
        self.geometry(_geo_near_portrait(self, CHAT_W, CHAT_H))
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close_clicked)

        # ----- ttk style -----
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Vertical.TScrollbar", troughcolor=BG, background=BORDER,
                        bordercolor=BORDER, arrowcolor=FG, gripcount=0)
        style.configure("Nulla.TButton", background=BTN_BG, foreground=FG, padding=4)
        style.map("Nulla.TButton", background=[("active", "#222222")])

        self.option_add("*Font", f"{MONO[0]} {MONO[1]}")
        self.option_add("*foreground", FG)
        self.option_add("*background", BG)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.text = tk.Text(self, wrap="word", bg=BG, fg=FG, insertbackground=FG,
                            padx=8, pady=8, relief="flat", highlightthickness=0,
                            state="disabled")
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=self.scroll.set)
        self.text.grid(row=0, column=0, sticky="nsew", padx=(8,0), pady=(8,6))
        self.scroll.grid(row=0, column=1, sticky="ns", padx=(0,8), pady=(8,6))

        bottom = tk.Frame(self, bg=BG, highlightthickness=1, highlightbackground=BORDER)
        self.bottom = bottom
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0,8))
        bottom.grid_columnconfigure(0, weight=1)

        self.entry = tk.Entry(bottom, bg=ENTRY_BG, fg=FG, insertbackground=FG, relief="flat")
        self.entry.grid(row=0, column=0, sticky="ew", padx=(6,8), pady=6)
        self.entry.bind("<Return>", self._on_enter)
        self.entry.bind("<Shift-Return>", lambda e: self._insert_newline())

        self.send_btn = ttk.Button(bottom, text="Send", command=self._on_send, style="Nulla.TButton")
        self.send_btn.grid(row=0, column=1, sticky="e", padx=6, pady=6)

        self.text.tag_configure("meta",   foreground=FG_META)
        self.text.tag_configure("you",    foreground=YOU_COLOR)
        self.text.tag_configure("nulla",  foreground=NULLA_COLOR)
        self.text.tag_configure("game",   foreground=GAME_COLOR)
        self.text.tag_configure("time",   foreground=FG_META)

        self.history = [{"role": "system", "content": NC()._persona()}]
        self.entry.focus_set()

        # ----- buffered, tag-aware UI writer -----
        self._ui_lock = threading.Lock()
        self._ui_buf = []
        self._ui_scheduled = False

        # ===== CUT-OFF + TWO-STAGE TTS PIPELINE =====
        self._tts_turn_lock = threading.Lock()
        self._tts_turn = 0
        self._last_tts = None

        # text -> wav generation (GPU thread)
        self._tts_text_q = queue.Queue()
        # wav -> playback (CPU thread), small buffer so we don't spam temp files
        self._tts_play_q = queue.Queue(maxsize=3)

        threading.Thread(target=self._tts_gen_worker, daemon=True).start()
        threading.Thread(target=self._tts_play_worker, daemon=True).start()

        # ----- games manager (optional) -----
        self.gm = None
        if GameManager is not None:
            self.gm = GameManager(
                get_story_window=lambda: self,
                story_push=lambda s: self._append_line(f"🕹️  Nulla (game): {s}", "game"),
                enqueue_sentence_if_ready=lambda s: self._tts_enqueue(s),
                mark_nulla_spoke=lambda: None,
                idle_touch=lambda: None,
                idle_block_push=lambda: None,
                idle_block_pop=lambda: None,
                tts_cutoff=lambda: self.tts_cutoff_now(),
            )

        threading.Thread(target=self._prewarm_then_intro, daemon=True).start()

        try:
            import nulla_hear
            nulla_hear.attach(self)
        except Exception:
            pass

    # ===== cutoff =====
    def _clear_queue_quick(self, q: queue.Queue, delete_wavs: bool = False):
        try:
            while True:
                item = q.get_nowait()
                if delete_wavs:
                    # item may be (turn_id, path)
                    try:
                        if isinstance(item, tuple) and len(item) >= 2:
                            p = item[1]
                            if p and os.path.isfile(p):
                                os.remove(p)
                    except Exception:
                        pass
        except Exception:
            pass

    def tts_cutoff_now(self):
        with self._tts_turn_lock:
            self._tts_turn += 1
            self._last_tts = None

        # Stops current playback AND unblocks SND_SYNC if it's in-flight
        _stop_audio_immediately()

        # Drop queued text + queued wavs
        self._clear_queue_quick(self._tts_text_q, delete_wavs=False)
        self._clear_queue_quick(self._tts_play_q, delete_wavs=True)

        _stop_audio_immediately()

    # ===== TTS enqueue =====
    def _tts_enqueue(self, snip: str):
        cleaned = clean_for_tts_strict(snip)
        if not cleaned:
            return
        if cleaned == self._last_tts:
            return
        self._last_tts = cleaned

        with self._tts_turn_lock:
            turn_id = self._tts_turn

        self._tts_text_q.put((turn_id, cleaned))

    # ===== Stage A: generate wav (GPU, single-thread) =====
    def _tts_gen_worker(self):
        while True:
            item = self._tts_text_q.get()
            if item is None:
                break

            turn_id, txt = item
            try:
                with self._tts_turn_lock:
                    cur = self._tts_turn
                if turn_id != cur:
                    continue

                # generate wav on GPU
                p = NC().tts_chunk_to_file(txt, "en")

                with self._tts_turn_lock:
                    cur = self._tts_turn
                if turn_id != cur:
                    try:
                        if p and os.path.isfile(p):
                            os.remove(p)
                    except Exception:
                        pass
                    continue

                # hand off to playback thread (blocks if buffer full)
                try:
                    self._tts_play_q.put((turn_id, p), timeout=10)
                except Exception:
                    try:
                        if p and os.path.isfile(p):
                            os.remove(p)
                    except Exception:
                        pass

            except Exception:
                pass

    # ===== Stage B: playback (CPU) =====
    def _tts_play_worker(self):
        """
        IMPORTANT: use SND_SYNC so we don't "time-stop" audio.
        That eliminates the classic 'last word clipped' issue caused by duration guesses.
        Cutoff still works via SND_PURGE from another thread.
        """
        while True:
            item = self._tts_play_q.get()
            if item is None:
                break

            turn_id, p = item
            try:
                with self._tts_turn_lock:
                    cur = self._tts_turn
                if turn_id != cur:
                    try:
                        if p and os.path.isfile(p):
                            os.remove(p)
                    except Exception:
                        pass
                    continue

                try:
                    import winsound
                    winsound.PlaySound(p, winsound.SND_FILENAME | winsound.SND_SYNC)
                except Exception:
                    # fallback to engine if winsound fails (best-effort)
                    try:
                        NC().play_once_then_delete(p)
                        p = None
                    except Exception:
                        pass

                # If we were cut off mid-play, SND_SYNC returns early; just clean up.
                try:
                    if p and os.path.isfile(p):
                        os.remove(p)
                except Exception:
                    pass

            except Exception:
                pass

    # ===== UI buffered writing =====
    def _ui_queue(self, text, tag=None):
        with self._ui_lock:
            self._ui_buf.append((text, tag))
            if not self._ui_scheduled:
                self._ui_scheduled = True
                self.text.after(UI_FLUSH_MS, self._ui_do_flush)

    def _ui_do_flush(self):
        with self._ui_lock:
            items = self._ui_buf[:]
            self._ui_buf.clear()
            self._ui_scheduled = False
        if not items:
            return
        self.text.configure(state="normal")
        for txt, tag in items:
            if tag:
                self.text.insert("end", txt, (tag,))
            else:
                self.text.insert("end", txt)
        self.text.see("end")
        self.text.configure(state="disabled")

    def _ui_force_flush(self, timeout=0.5):
        done = threading.Event()
        def do():
            self._ui_do_flush()
            done.set()
        self.text.after(0, do)
        done.wait(timeout=timeout)

    # ===== small helpers =====
    def _on_close_clicked(self):
        self.iconify()

    def _insert_newline(self):
        self.entry.insert("insert", "\n")

    def _append(self, text, tag=None):
        def do():
            self.text.configure(state="normal")
            if tag:
                self.text.insert("end", text, (tag,))
            else:
                self.text.insert("end", text)
            self.text.see("end")
            self.text.configure(state="disabled")
        self.text.after(0, do)

    def _append_line(self, text, tag=None):
        self._append(text + "\n", tag)

    def _on_enter(self, _):
        self._on_send()

    # ===== command helpers (games) =====
    def _maybe_handle_game_command(self, user_text: str) -> bool:
        if not self.gm:
            return False

        t = user_text.strip().lower()

        if self.gm.matches_help(t):
            self._append_line(self.gm.help_text(), "meta")
            return True

        ok, msg = self.gm.try_play(user_text)
        if ok:
            self._append_line(f"💬  Nulla: {msg}", "nulla")
            self._tts_enqueue(msg)
            return True

        return False

    # ===== chat flow =====
    def _on_send(self):
        user_text = self.entry.get().strip()
        if not user_text:
            return

        # New user message => instant cutoff
        self.tts_cutoff_now()

        self.entry.delete(0, "end")
        ts = _now_str()

        self._append(f"🗨  You: {user_text}  ", "you")
        self._append("·  ", "time")
        self._append_line(ts, "time")

        if self._maybe_handle_game_command(user_text):
            return

        self.send_btn.config(state="disabled")
        threading.Thread(target=self._worker_reply, args=(user_text,), daemon=True).start()

    def _worker_reply(self, user_text):
        reply = ""
        try:
            self._append(f"\n💬  Nulla: ", "nulla")
            try:
                stream = NC().chat_streaming(user_text, self.history)
                reply = self._stream_to_ui_and_tts(stream)
            except Exception as e:
                self._append_line(f"[streaming error] {e}", "meta")
                reply = NC().chat_once(user_text, self.history).strip()
                self._ui_queue(sanitize_for_ui(reply), "nulla")
                for c in NC().split_chunks(reply, max_chars=140):
                    self._tts_enqueue(c)

            self._ui_force_flush()
            self._append_line(f"  ·  { _now_str() }", "time")
        finally:
            if reply:
                self.history += [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": reply},
                ]
            self.send_btn.config(state="normal")

    def _stream_to_ui_and_tts(self, stream_iter):
        """
        v0.0.7 streaming logic, but FIXED so the kick does NOT discard extra buffered text.
        """
        buf = ""
        full = ""
        first_sent_spoken = False
        kick_started_at = None
        kick_done = False

        FIRST = min(getattr(NC(), "FIRST_CHUNK_MAX", 80), 64)
        NEXT  = min(getattr(NC(), "NEXT_CHUNK_MAX", 180), 140)
        BOUND = getattr(NC(), "BOUNDARY", None)

        def boundary_hit_fast(s):
            if not s:
                return False
            if '.' in s or '!' in s or '?' in s:
                return bool(BOUND.search(s)) if BOUND else True
            return False

        for frag in stream_iter:
            if kick_started_at is None:
                kick_started_at = time.monotonic()

            full += frag
            buf  += frag
            self._ui_queue(sanitize_for_ui(frag), "nulla")

            limit = FIRST if not first_sent_spoken else NEXT
            hit_boundary = boundary_hit_fast(buf)

            # ---- INSTANT KICK (FIXED: keep leftover instead of nuking buf) ----
            if (not first_sent_spoken) and (not hit_boundary) and (not kick_done):
                if (time.monotonic() - kick_started_at) >= (KICK_MS / 1000.0) and len(buf) >= KICK_MIN_CHARS:
                    raw = buf
                    snip_raw = raw[:KICK_MAX_CHARS]

                    cut = snip_raw.rfind(" ")
                    if cut >= max(8, KICK_MIN_CHARS // 2):
                        snip_raw = snip_raw[:cut]

                    # We enqueue the stripped version, but consume snip_raw length from buffer
                    snip = snip_raw.strip()
                    if snip:
                        self._tts_enqueue(snip)
                        first_sent_spoken = True
                        kick_done = True

                        # KEEP the remainder (this prevents "random missing word" after kick)
                        buf = raw[len(snip_raw):]
                        continue

            # ---- Normal boundary/length gate ----
            if hit_boundary or len(buf) >= limit:
                snip = buf.strip()
                self._tts_enqueue(snip)
                first_sent_spoken = True
                buf = ""

        tail = buf.strip()
        if tail:
            self._ui_queue(sanitize_for_ui(tail), "nulla")
            self._tts_enqueue(tail)
        return full

    def _prewarm_then_intro(self):
        try:
            NC().prewarm()
        except Exception:
            pass
        try:
            if hasattr(NC(), "play_intro_if_available"):
                self.after(0, NC().play_intro_if_available)
        except Exception:
            pass

def run_window():
    app = NullaChatWindow()
    app.mainloop()

if __name__ == "__main__":
    run_window()
