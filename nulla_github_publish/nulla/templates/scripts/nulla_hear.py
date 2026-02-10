# nulla_source\scripts\nulla_hear.py
# Voice toggle + Whisper worker (CPU). Parent UI = Tk window from nulla_window.py
# Worker runs inside the Whisper venv and communicates via JSONL over stdio.

from __future__ import annotations
import os, sys, json, threading, subprocess, queue, time, traceback

# ---------- Paths (resolve from scripts folder) ----------
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR    = os.path.abspath(os.path.join(_SCRIPTS_DIR, ".."))
_WISP_DIR    = os.path.join(_BASE_DIR, "Whisper")
_WISP_PY     = os.path.join(_WISP_DIR, ".venv", "Scripts", "python.exe")
_ICONS_DIR   = os.path.join(_BASE_DIR, "assets")
_ICON_OFF    = os.path.join(_ICONS_DIR, "mic_off.png")
_ICON_ON     = os.path.join(_ICONS_DIR, "mic_on.png")

# ---- verbosity controls ------------------------------------------------
# Default: quiet UI. Turn on debug with:  set WHISPER_VERBOSE=1
VERBOSE = os.environ.get("WHISPER_VERBOSE", "0") == "1"
SHOW_WHISPER_STDERR = VERBOSE  # only surface worker's stderr when verbose

def _worker_cmd():
    return [_WISP_PY, "-u", os.path.join(_SCRIPTS_DIR, "nulla_hear.py"), "--worker"]

def _worker_env() -> dict:
    env = os.environ.copy()

    # ---- keep caches local to project (portable) ----
    env.setdefault("XDG_CACHE_HOME", os.path.join(_WISP_DIR, ".cache"))
    env.setdefault("HF_HOME",        os.path.join(_WISP_DIR, ".cache", "huggingface"))

    # ---- force our portable ffmpeg ----
    _FFMPEG_BIN = os.path.join(_BASE_DIR, "bin", "ffmpeg")
    env["PATH"] = _FFMPEG_BIN + os.pathsep + env.get("PATH", "")
    env["IMAGEIO_FFMPEG_EXE"] = os.path.join(_FFMPEG_BIN, "ffmpeg.exe")
    env["FFMPEG_BINARY"]      = os.path.join(_FFMPEG_BIN, "ffmpeg.exe")
    env["FFPROBE_BINARY"]     = os.path.join(_FFMPEG_BIN, "ffprobe.exe")

    # Optional: quiet model download progress bars etc.
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env

# ======================================================================
#                     P A R E N T   (Tkinter process)
# ======================================================================

class HearController:
    """Owns the mic toggle button, the Whisper worker, and the auto-send timer."""
    def __init__(self, app):
        self.app = app  # NullaChatWindow
        self.proc: subprocess.Popen | None = None
        self.reader_thread: threading.Thread | None = None
        self.err_thread: threading.Thread | None = None
        self.read_q: "queue.Queue[dict]" = queue.Queue()
        self.alive = False
        self.listening = False
        self.ready = False
        self.timer_id = None
        self.lock = threading.Lock()
        self.session = 0  # increments every START

        # Tk images must be held to avoid GC
        import tkinter as tk
        self.img_off = tk.PhotoImage(file=_ICON_OFF) if os.path.isfile(_ICON_OFF) else None
        self.img_on  = tk.PhotoImage(file=_ICON_ON)  if os.path.isfile(_ICON_ON) else None

        self._build_button()
        threading.Thread(target=self._ensure_worker, daemon=True).start()

    def _build_button(self):
        from tkinter import ttk
        try:
            self.app.bottom.grid_columnconfigure(2, minsize=36)
        except Exception:
            pass

        style = ttk.Style(self.app)
        try:
            style.configure("Mic.TButton", background="#1A1A1A", padding=2)
            style.map("Mic.TButton", background=[("active", "#222222")])
        except Exception:
            pass

        self.btn = ttk.Button(self.app.bottom, style="Mic.TButton", command=self.toggle)
        if self.img_off: self.btn.configure(image=self.img_off)
        self.btn.grid(row=0, column=2, sticky="e", padx=(6,6), pady=6)

        # Right-click = show input devices (on-demand only)
        try:
            self.btn.bind("<Button-3>", lambda e: self._send_cmd({"cmd": "LIST_DEVICES"}))
        except Exception:
            pass

    # ---- process wiring ----
    def _ensure_worker(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return
            try:
                stderr_target = subprocess.PIPE if SHOW_WHISPER_STDERR else subprocess.DEVNULL
                self.proc = subprocess.Popen(
                    _worker_cmd(),
                    cwd=_SCRIPTS_DIR,
                    env=_worker_env(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr_target,
                    text=True, bufsize=1  # line-buffered
                )
                self.alive = True
                self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
                self.reader_thread.start()

                if SHOW_WHISPER_STDERR:
                    self.err_thread = threading.Thread(target=self._stderr_loop, daemon=True)
                    self.err_thread.start()
            except Exception as e:
                self._notify_user(f"[Voice] Failed to start worker: {e}")

    def _reader_loop(self):
        try:
            assert self.proc and self.proc.stdout
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                self.read_q.put(evt)
                self.app.after(0, self._handle_events)
        except Exception:
            pass

    def _stderr_loop(self):
        try:
            assert self.proc and self.proc.stderr
            for line in self.proc.stderr:
                line = line.strip()
                if not line:
                    continue
                # Only show when verbose
                self._notify_user(f"[Voice:stderr] {line}")
        except Exception:
            pass

    def _handle_events(self):
        while True:
            try:
                evt = self.read_q.get_nowait()
            except queue.Empty:
                break

            typ = evt.get("type")

            if typ == "ready":
                self.ready = True

            elif typ == "status":
                if evt.get("session") == self.session:
                    state = evt.get("state")
                    self._set_button_state(state == "listening")
                # stream open info -> only when verbose
                if VERBOSE and ("rate" in evt or "dtype" in evt):
                    r = evt.get("rate")
                    d = evt.get("dtype")
                    if r or d:
                        self._notify_user(f"[Voice] stream opened (rate={r}, dtype={d})")

            elif typ == "devices":
                if not VERBOSE:
                    # Stay silent in clean mode
                    continue
                items = evt.get("items", [])
                if not items:
                    self._notify_user("[Voice] (no input devices found)")
                else:
                    self._notify_user("[Voice] Input devices:")
                    for it in items:
                        if "error" in it:
                            self._notify_user(f"  ! {it['error']}")
                            continue
                        self._notify_user(
                            f"  #{it['index']}: {it['name']} | hostapi {it['hostapi']} "
                            f"| max_in {it['max_in']} | sr {it.get('default_sr')}"
                        )

            elif typ == "final":
                if evt.get("session") != self.session or not self.listening:
                    continue
                text = evt.get("text", "").strip()
                if text:
                    self._append_to_entry(text)
                    self._reset_timer()

            elif typ == "error":
                self._notify_user(f"[Voice] {evt.get('message','error')}")

    # ---- entry + timer ----
    def _append_to_entry(self, text):
        e = self.app.entry
        cur = e.get()
        if cur and not cur.endswith((" ", "\n")):
            e.insert("end", " ")
        e.insert("end", text)
        e.icursor("end")
        e.focus_set()

    def _reset_timer(self):
        if self.timer_id:
            try: self.app.after_cancel(self.timer_id)
            except Exception: pass
        self.timer_id = self.app.after(20, self._auto_send)

    def _auto_send(self):
        self.timer_id = None
        if self.app.entry.get().strip():
            try:
                self.app._on_send()
            except Exception:
                pass

    # ---- public actions ----
    def toggle(self):
        if not self.alive:
            self._ensure_worker()
        if not self.listening:
            self.start()
        else:
            self.stop()

    def start(self):
        self._ensure_worker()
        if not self.ready:
            self._spin_until_ready(5.0)
        self.session += 1
        self._send_cmd({"cmd":"START", "session": self.session})
        self.listening = True
        self._set_button_state(True)

    def stop(self):
        self.listening = False
        self._set_button_state(False)
        if self.timer_id:
            try: self.app.after_cancel(self.timer_id)
            except Exception: pass
            self.timer_id = None
        self._send_cmd({"cmd":"STOP", "session": self.session})

    def shutdown(self):
        try:
            self._send_cmd({"cmd":"EXIT"})
        except Exception:
            pass
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    # ---- helpers ----
    def _send_cmd(self, obj):
        try:
            if not self.proc or not self.proc.stdin:
                return
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()
        except Exception:
            pass

    def _spin_until_ready(self, seconds: float):
        t0 = time.monotonic()
        while not self.ready and (time.monotonic() - t0) < seconds:
            time.sleep(0.05)

    def _set_button_state(self, on: bool):
        if self.img_on and self.img_off:
            self.btn.configure(image=self.img_on if on else self.img_off)

    def _notify_user(self, msg: str):
        try:
            self.app._append_line(msg, tag="meta")
        except Exception:
            print(msg, file=sys.stderr)


# Public API for nulla_window to call
_controller: HearController | None = None

def attach(app):
    """Call once from NullaChatWindow.__init__."""
    global _controller
    if _controller is None:
        _controller = HearController(app)
    return _controller

# ======================================================================
#                       W O R K E R   P R O C E S S
# ======================================================================

def _worker_main():
    """
    Runs inside the Whisper venv. Loads model once (CPU), then responds to:
      {"cmd":"START","session":N}, {"cmd":"STOP","session":N}, {"cmd":"EXIT"}, {"cmd":"LIST_DEVICES"}
    Emits:
      {"type":"ready"}
      {"type":"devices","items":[...]}              (only on LIST_DEVICES)
      {"type":"status","state":"listening"|"idle"|"opened","session":N,"rate"?:sr,"dtype"?:str}
      {"type":"final","text":"...","session":N}
      {"type":"error","message":"..."}
    """
    import sys, json, threading, queue, time, ctypes, ctypes.wintypes as wt

    # ======== PERFORMANCE KNOBS ========
    _auto_threads = max(1, int((os.cpu_count() or 8) * 0.8))  # ~80% logical cores
    THREADS    = int(os.environ.get("WHISPER_CPU_THREADS", _auto_threads))
    BLOCK_MS   = 20
    VAD_MODE   = 2
    EOS_SIL_MS = 500
    MIN_UTT_MS = 300
    RATE       = 16000

    STREAM_DTYPE = 'int16'
    SAMPLE_BYTES = 2

    try:
        _HIGH = 0x00000080  # HIGH_PRIORITY_CLASS
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), _HIGH)
    except Exception:
        pass

    os.environ["OMP_NUM_THREADS"]       = str(THREADS)
    os.environ["MKL_NUM_THREADS"]       = str(THREADS)
    os.environ["OPENBLAS_NUM_THREADS"]  = str(THREADS)
    os.environ["NUMEXPR_NUM_THREADS"]   = str(THREADS)
    os.environ.setdefault("KMP_BLOCKTIME", "0")

    try:
        import numpy as np
        import sounddevice as sd
        import webrtcvad
        import torch
        import whisper
    except Exception as e:
        _wout({"type":"error", "message": f"Import failure: {e}"})
        sys.exit(1)

    try:
        torch.set_num_threads(THREADS)
        torch.set_num_interop_threads(max(1, THREADS // 2))
    except Exception:
        pass

    try:
        sd.default.latency = "low"
    except Exception:
        pass

    # ---- device utils ----
    def _devices_summary():
        items = []
        try:
            devs = sd.query_devices()
            hapis = sd.query_hostapis()
            for i, d in enumerate(devs):
                if d.get("max_input_channels",0) > 0:
                    ha = d.get("hostapi", 0)
                    han = (hapis[ha]["name"] if ha < len(hapis) else "?")
                    items.append({"index": i, "name": d["name"], "hostapi": han,
                                  "max_in": d["max_input_channels"],
                                  "default_sr": d.get("default_samplerate", None)})
        except Exception as e:
            items.append({"error": str(e)})
        return items

    def _pick_input_device(prefer: str|None):
        try:
            devs = sd.query_devices()
            if prefer and prefer.isdigit():
                idx = int(prefer)
                if 0 <= idx < len(devs) and devs[idx].get("max_input_channels",0) > 0:
                    return idx
            if prefer:
                p = prefer.lower()
                for i, d in enumerate(devs):
                    if d.get("max_input_channels",0) > 0 and p in (d["name"] or "").lower():
                        return i
            di = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
            if isinstance(di, int) and di >= 0 and devs[di].get("max_input_channels",0) > 0:
                return di
            for i, d in enumerate(devs):
                if d.get("max_input_channels",0) > 0:
                    return i
        except Exception:
            pass
        return None

    # ---- load model (cache paths forced by parent env) ----
    try:
        model = whisper.load_model("small.en")
    except Exception as e:
        _wout({"type":"error", "message": f"Whisper load failed: {e}"})
        sys.exit(1)

    _wout({"type":"ready"})  # no auto device dump in quiet mode

    cmd_q: "queue.Queue[tuple[str,int|None]]" = queue.Queue()
    current_session: int | None = None
    listening = False
    stream = None
    audio_q: "queue.Queue[bytes]" = queue.Queue()
    abort_event = threading.Event()
    vad = webrtcvad.Vad(VAD_MODE)

    def _stdin_reader():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cmd = obj.get("cmd","").upper()
                sess = obj.get("session")
                if cmd == "LIST_DEVICES":
                    _wout({"type":"devices","items": _devices_summary()})
                elif cmd:
                    cmd_q.put((cmd, sess))
            except Exception:
                continue
    threading.Thread(target=_stdin_reader, daemon=True).start()

    def _on_audio(indata, frames, time_info, status):
        try:
            audio_q.put(bytes(indata))
        except Exception:
            pass

    def _open_stream():
        nonlocal stream, STREAM_DTYPE, SAMPLE_BYTES
        prefer = os.environ.get("WHISPER_INPUT_DEVICE")
        idx = _pick_input_device(prefer)
        if idx is not None:
            try: sd.default.device = (idx, None)
            except Exception: pass

        def _try_open(samplerate, dtype):
            return sd.RawInputStream(samplerate=samplerate,
                                     blocksize=int(samplerate * 20 / 1000),
                                     dtype=dtype, channels=1, callback=_on_audio)

        try:
            stream = _try_open(16000, 'int16'); stream.start()
            STREAM_DTYPE = 'int16'; SAMPLE_BYTES = 2
            _wout({"type":"status","state":"opened","session": current_session, "rate": 16000, "dtype": "int16"})
            return 16000
        except Exception as e1:
            e_1 = e1
        try:
            stream = _try_open(16000, 'float32'); stream.start()
            STREAM_DTYPE = 'float32'; SAMPLE_BYTES = 4
            _wout({"type":"status","state":"opened","session": current_session, "rate": 16000, "dtype": "float32"})
            return 16000
        except Exception as e2:
            e_2 = e2

        try:
            dev = sd.query_devices(kind='input')
            sr = int(dev.get('default_samplerate', 44100) or 44100)
        except Exception:
            sr = 44100

        for dtype in ('int16', 'float32'):
            try:
                stream = _try_open(sr, dtype); stream.start()
                STREAM_DTYPE = dtype; SAMPLE_BYTES = (2 if dtype == 'int16' else 4)
                _wout({"type":"status","state":"opened","session": current_session, "rate": sr, "dtype": dtype})
                return sr
            except Exception:
                continue
        raise RuntimeError(f"could not open mic: {e_1} | {e_2}")

    def _clear_audio_q():
        try:
            while True:
                audio_q.get_nowait()
        except queue.Empty:
            pass

    def _to_int16_16k(pcm_bytes: bytes, in_rate: int):
        if STREAM_DTYPE == 'int16':
            a = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            a = np.frombuffer(pcm_bytes, dtype=np.float32)
        if in_rate != RATE:
            duration = len(a) / in_rate
            new_len = max(1, int(duration * RATE))
            if new_len <= 1: return b""
            x_old = np.linspace(0, 1, num=len(a), endpoint=False, dtype=np.float32)
            x_new = np.linspace(0, 1, num=new_len, endpoint=False, dtype=np.float32)
            a = np.interp(x_new, x_old, a)
        b = np.clip(a * 32768.0, -32768.0, 32767.0).astype(np.int16)
        return b.tobytes()

    def _collect_utterance(in_rate: int) -> bytes | None:
        samples_per_block_in  = int(in_rate * BLOCK_MS / 1000)
        samples_per_block_16k = int(RATE    * BLOCK_MS / 1000)
        block_bytes_in  = samples_per_block_in  * (2 if STREAM_DTYPE == 'int16' else 4)
        block_bytes_16k = samples_per_block_16k * 2

        voiced = bytearray()
        started = False
        silence_ms = 0
        t0 = time.monotonic()

        while not abort_event.is_set():
            try:
                chunk = audio_q.get(timeout=0.2)
            except queue.Empty:
                if started:
                    break
                continue

            for i in range(0, len(chunk), block_bytes_in):
                if abort_event.is_set():
                    return None
                frame = chunk[i:i+block_bytes_in]
                if len(frame) < block_bytes_in:
                    continue

                frame16 = _to_int16_16k(frame, in_rate)
                try:
                    speaking = webrtcvad.Vad(VAD_MODE).is_speech(frame16[:block_bytes_16k], RATE)
                except Exception:
                    speaking = False

                if speaking:
                    started = True
                    voiced.extend(frame16[:block_bytes_16k])
                    silence_ms = 0
                else:
                    if started:
                        silence_ms += BLOCK_MS
                        if silence_ms >= EOS_SIL_MS:
                            if (time.monotonic() - t0)*1000 < MIN_UTT_MS:
                                return None
                            return bytes(voiced)
        return None

    def _pcm16_to_float32(pcm_bytes: bytes):
        a = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        return a / 32768.0

    def _transcribe(pcm16: bytes) -> str:
        audio = _pcm16_to_float32(pcm16)
        try:
            with torch.inference_mode():
                result = model.transcribe(audio, language="en", fp16=False, verbose=False)
            return (result.get("text") or "").strip()
        except Exception as e:
            _wout({"type":"error", "message": f"transcribe failed: {e}"})
            return ""

    in_rate = RATE
    while True:
        try:
            cmd, sess = cmd_q.get(timeout=0.01)
        except queue.Empty:
            cmd = None
            sess = None

        if cmd == "EXIT":
            try:
                if stream: stream.stop(); stream.close()
            except Exception: pass
            _wout({"type":"status","state":"idle","session": current_session})
            break

        if cmd == "START":
            abort_event.clear()
            _clear_audio_q()
            current_session = int(sess) if isinstance(sess, int) else (current_session or 0) + 1
            try:
                in_rate = _open_stream()
                listening = True
                _wout({"type":"status","state":"listening","session": current_session})
            except Exception as e:
                _wout({"type":"error", "message": f"mic open failed: {e}"})
                listening = False
                current_session = None

        if cmd == "STOP":
            abort_event.set()
            _clear_audio_q()
            try:
                if stream: stream.stop(); stream.close()
            except Exception: pass
            stream = None
            listening = False
            _wout({"type":"status","state":"idle","session": current_session})

        if listening and not abort_event.is_set() and stream:
            utt = _collect_utterance(in_rate)
            if abort_event.is_set() or not listening or not stream:
                continue
            if utt:
                text = _transcribe(utt)
                if text:
                    _wout({"type":"final", "text": text, "session": current_session})

    sys.exit(0)

def _wout(obj: dict):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()

if __name__ == "__main__":
    if "--worker" in sys.argv:
        try:
            _worker_main()
        except Exception:
            _wout({"type":"error", "message": traceback.format_exc()})
            sys.exit(1)
    # Parent mode is imported by nulla_window; nothing to do on direct run.
