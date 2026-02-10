# scripts/nulla_chat.py
import os, json, uuid, queue, threading, requests, re, time, winsound, hashlib
import sys
import atexit
import signal
from typing import List, Optional, Tuple

# =============================================================================
# PATHS
# =============================================================================
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(_SCRIPTS_DIR, ".."))
XTTS_DIR   = os.path.join(BASE_DIR, "XTTS-v2")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
TEMP_DIR   = os.path.join(XTTS_DIR, "temp")
INTRO_WAV  = os.path.join(ASSETS_DIR, "intro.wav")  # optional

os.makedirs(TEMP_DIR, exist_ok=True)

# Coqui / HF caches under your project (portable)
os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("TTS_HOME",       os.path.join(XTTS_DIR, "tts_home"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(XTTS_DIR, "hf_cache"))
os.environ.setdefault("HF_HOME",        os.path.join(XTTS_DIR, "hf_cache"))

# Force portable ffmpeg
_FFMPEG_BIN = os.path.join(BASE_DIR, "bin", "ffmpeg")
os.environ.setdefault("IMAGEIO_FFMPEG_EXE", os.path.join(_FFMPEG_BIN, "ffmpeg.exe"))
os.environ.setdefault("FFMPEG_BINARY",      os.path.join(_FFMPEG_BIN, "ffmpeg.exe"))
os.environ.setdefault("FFPROBE_BINARY",     os.path.join(_FFMPEG_BIN, "ffprobe.exe"))
os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

def _find_ref_wav(root: str) -> Optional[str]:
    try:
        for name in ("voice_ref.wav", "nulla_ref.wav", "ref.wav"):
            p = os.path.join(root, name)
            if os.path.isfile(p):
                return p
        for f in os.listdir(root):
            if f.lower().endswith(".wav"):
                return os.path.join(root, f)
    except Exception:
        pass
    return None

SPEAKER_WAV = _find_ref_wav(ASSETS_DIR)  # may be None

# =============================================================================
# LLM (LM Studio / llama.cpp OpenAI server)
# =============================================================================
LMSTUDIO_BASE  = "http://127.0.0.1:1234/v1"
LMSTUDIO_MODEL = None
MAX_TOKENS     = 140
TEMPERATURE    = 0.6

# =============================================================================
# CONTEXT TRIMMING (NEW – prevents llama 400 mid-chat)
# Drop old messages BEFORE sending to the server.
# - Keeps persona + recent turns so she gets "forgetful" instead of dying.
# =============================================================================
MAX_TURNS_TO_KEEP  = 10   # keep last N user/assistant pairs
MAX_MESSAGES_SOFT  = 32   # absolute safety cap on total messages
MAX_SYSTEM_MSGS    = 4    # keep persona + last few system injections (emotion/poke/etc)

def _trim_messages_soft(messages: List[dict]) -> List[dict]:
    """
    Prevent context overflow by trimming old messages.
    Keeps:
    - persona (first system message)
    - last few system messages (emotion/poke injections)
    - last MAX_TURNS_TO_KEEP turns (user+assistant pairs)
    """
    if not messages:
        return messages

    system_msgs = [m for m in messages if m.get("role") == "system"]
    convo_msgs  = [m for m in messages if m.get("role") != "system"]

    kept_system = []
    if system_msgs:
        kept_system.append(system_msgs[0])  # persona / primary system
        tail = system_msgs[1:]
        if MAX_SYSTEM_MSGS > 1 and tail:
            kept_system.extend(tail[-(MAX_SYSTEM_MSGS - 1):])

    if MAX_TURNS_TO_KEEP > 0:
        convo_msgs = convo_msgs[-(MAX_TURNS_TO_KEEP * 2):]

    trimmed = kept_system + convo_msgs

    # Absolute cap (in case memory layer adds extras)
    if len(trimmed) > MAX_MESSAGES_SOFT:
        sys_len = len(kept_system)
        keep_tail = max(0, MAX_MESSAGES_SOFT - sys_len)
        trimmed = kept_system + (convo_msgs[-keep_tail:] if keep_tail else [])

    return trimmed

# =============================================================================
# Playback / chunk tuning
# =============================================================================
FIRST_CHUNK_MAX = 80
NEXT_CHUNK_MAX  = 180
SIL_PAD_MS      = 120
GLIDE_PAUSE_MS  = 10

# --- GUI guard ---
_GUI_STARTED = False

# =============================================================================
# OPTIONAL MEMORY (v0.0.8)
# =============================================================================
try:
    import nulla_memory as mem
except Exception:
    mem = None

def _mem_has(name: str) -> bool:
    return bool(mem) and hasattr(mem, name)

def _mem_handle_control(user_text: str):
    if not _mem_has("handle_control_input"):
        return (False, None)
    try:
        return mem.handle_control_input(user_text)
    except Exception:
        return (False, None)

def _mem_answer_direct(user_text: str):
    if not _mem_has("answer_from_memory"):
        return (False, None)
    try:
        return mem.answer_from_memory(user_text)
    except Exception:
        return (False, None)

def _mem_build_messages(history: List[dict], prompt: str):
    if not _mem_has("build_messages"):
        return history + [{"role":"user","content":prompt}]
    try:
        return mem.build_messages(history, prompt)
    except Exception:
        return history + [{"role":"user","content":prompt}]

def _mem_append(role: str, content: str):
    if not _mem_has("append"):
        return
    try:
        mem.append(role, content)
    except Exception:
        pass

def _mem_log_assistant_enabled() -> bool:
    try:
        return bool(getattr(mem, "LOG_ASSISTANT", True))
    except Exception:
        return True

# =============================================================================
# EMOTION SYSTEM (v0.0.9 add-on) - NO behavior removal, only injection
# =============================================================================
try:
    from nulla_emotion import (
        EmotionManager,
        get_emotion_prompt,
        consume_poke_context,
        STATE_PATH,
        handle_emotion_control_input,
    )
    print("[CHAT] STATE_PATH =", os.path.abspath(STATE_PATH))
    _EM = EmotionManager()
except Exception as e:
    print("[CHAT] emotion import failed:", e)
    EmotionManager = None
    get_emotion_prompt = None
    consume_poke_context = None
    handle_emotion_control_input = None
    STATE_PATH = os.path.join(_SCRIPTS_DIR, "nulla_emotion_state.json")
    _EM = None

def _emo_handle_control(user_text: str):
    """
    Non-blocking emotion control handler (help emoflush).
    Returns (handled, response_text).
    If handled=True, caller should show response_text and NOT call the LLM.
    """
    if not callable(globals().get("handle_emotion_control_input", None)):
        return (False, None)
    try:
        return handle_emotion_control_input(user_text)
    except Exception:
        return (False, None)

def _emotion_update(prompt: str) -> str:
    """
    Updates shared emotion state based on user text.
    Returns effective emotion name (e.g., 'Anger').
    Safe no-op if nulla_emotion.py isn't present.
    """
    if _EM is None:
        return "Neutral"
    try:
        return _EM.update_from_user_text(prompt)
    except Exception:
        return "Neutral"

def _inject_emotion_system_message(messages: List[dict], emotion: str) -> List[dict]:
    """
    Injects emotion prompt as a *second* system message so your original persona stays intact.
    Does not mutate history; only affects outbound request.
    """
    if not get_emotion_prompt:
        return messages
    try:
        emo_prompt = get_emotion_prompt(emotion)
        if not emo_prompt:
            return messages

        for m in messages[:3]:
            if m.get("role") == "system" and "MODE:" in (m.get("content") or ""):
                return messages

        if messages and messages[0].get("role") == "system":
            return [messages[0], {"role":"system","content":emo_prompt}] + messages[1:]
        else:
            return [{"role":"system","content":emo_prompt}] + messages
    except Exception:
        return messages

def _inject_poke_system_message(messages: List[dict], ctx: str = "") -> List[dict]:
    """
    Inject poke context as an additional system message (once).
    If ctx isn't provided, it will consume it here.
    """
    if not ctx:
        try:
            ctx = consume_poke_context()
        except Exception:
            ctx = ""

    if not ctx:
        return messages

    for m in messages[:4]:
        if m.get("role") == "system" and "POKE CONTEXT:" in (m.get("content") or ""):
            return messages

    if messages and messages[0].get("role") == "system":
        if len(messages) > 1 and messages[1].get("role") == "system" and "MODE:" in (messages[1].get("content") or ""):
            return [messages[0], messages[1], {"role": "system", "content": ctx}] + messages[2:]
        return [messages[0], {"role": "system", "content": ctx}] + messages[1:]

    return [{"role": "system", "content": ctx}] + messages

# =============================================================================
# Reset emotion to Neutral on exit (BEST EFFORT)
# =============================================================================
_RESET_DONE = False

def _reset_emotion_json_to_neutral_best_effort():
    """
    Write Neutral directly to the JSON file, even if EmotionManager isn't available.
    Clears poke flags too.
    """
    try:
        path = STATE_PATH or os.path.join(_SCRIPTS_DIR, "nulla_emotion_state.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        d = {}
        try:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f) or {}
        except Exception:
            d = {}

        now = time.time()
        d["emotion"] = "Neutral"
        d["previous"] = "Neutral"
        d["updated_at"] = float(now)
        d["ttl_sec"] = 0.0

        d["poke_count"] = 0
        d["poke_last_at"] = 0.0
        d["poke_pending"] = False
        d["poke_reason"] = ""
        d["poke_triggered_at"] = 0.0

        # Keep portrait overlay sane if it reads these
        d["status_text"] = "Neutral"
        d["image_file"] = d.get("image_file") or "Nulla_neutral.png"
        d["status_color"] = d.get("status_color") or "#6B7280"

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass

def reset_emotion_to_neutral_on_exit():
    global _RESET_DONE
    if _RESET_DONE:
        return
    _RESET_DONE = True

    try:
        if EmotionManager is not None:
            try:
                EmotionManager().set_emotion("Neutral", ttl_sec=0.0)
            except Exception:
                pass
    except Exception:
        pass

    _reset_emotion_json_to_neutral_best_effort()

# Always try on normal exits
atexit.register(reset_emotion_to_neutral_on_exit)

def _install_windows_console_close_handler():
    """
    On Windows, closing the CMD window triggers CTRL_CLOSE_EVENT.
    atexit is NOT guaranteed to run, so we hook the console control handler.
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        from ctypes import wintypes

        # Console event codes
        CTRL_C_EVENT        = 0
        CTRL_BREAK_EVENT    = 1
        CTRL_CLOSE_EVENT    = 2
        CTRL_LOGOFF_EVENT   = 5
        CTRL_SHUTDOWN_EVENT = 6

        HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def _handler(ctrl_type):
            # Must be fast. Windows gives you a short deadline.
            try:
                reset_emotion_to_neutral_on_exit()
            except Exception:
                pass

            # Return False to let default handler continue (process will close).
            return False

        if not hasattr(_install_windows_console_close_handler, "_cb"):
            _install_windows_console_close_handler._cb = HandlerRoutine(_handler)

        k32 = ctypes.windll.kernel32
        k32.SetConsoleCtrlHandler.argtypes = [HandlerRoutine, wintypes.BOOL]
        k32.SetConsoleCtrlHandler.restype = wintypes.BOOL
        k32.SetConsoleCtrlHandler(_install_windows_console_close_handler._cb, True)
    except Exception:
        pass

def _install_signal_handlers():
    """
    Helps for Ctrl+C / SIGTERM (when supported).
    """
    def _sig_handler(signum, frame):
        try:
            reset_emotion_to_neutral_on_exit()
        except Exception:
            pass
        raise SystemExit

    try:
        signal.signal(signal.SIGINT, _sig_handler)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _sig_handler)
    except Exception:
        pass

_install_windows_console_close_handler()
_install_signal_handlers()

# =============================================================================
# poke response (LLM-generated + 100% fallback)
# =============================================================================
def _forced_poke_reply() -> str:
    return "Hey—why are you poking me? Stop it. I’m getting annoyed."

def _consume_poke_ctx_safely() -> str:
    if not callable(globals().get("consume_poke_context", None)):
        return ""
    try:
        return consume_poke_context() or ""
    except Exception:
        return ""

def _clean_one_liner(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("```", "").strip()
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    if len(s) > 160:
        s = s[:160].rsplit(" ", 1)[0].strip() + "…"
    return s

_POKE_OK_RE = re.compile(r"\b(poke|poking|stop|hey|quit|ow|ouch|hands|touch|clicked|clicking)\b", re.I)

def _looks_like_poke_reaction(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    if not _POKE_OK_RE.search(s):
        return False
    bad = ("system", "prompt", "instruction", "llm", "model", "json", "role:")
    low = s.lower()
    if any(b in low for b in bad):
        return False
    return True

def _generate_poke_line_llm(model_id: str) -> str:
    url = f"{LMSTUDIO_BASE}/chat/completions"

    sys_msg = (
    "You are Nulla.\n"
    "The user just poked you physically repeatedly and you’re annoyed.\n"
    "Write EXACTLY ONE short reaction line addressing the poking.\n"
    "Rules:\n"
    "- 1–2 short sentences.\n"
    "- Mild sass is okay. No insults.\n"
    "- No moralizing. No lectures.\n"
    "- No meta talk.\n"
    "- Output ONLY the line.\n"
)

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": "Generate the reaction line now."},
    ]

    body = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.55,
        "max_tokens": 60,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}

    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=20)
    if r.status_code == 401:
        headers["Authorization"] = "Bearer lm-studio"
        r = requests.post(url, headers=headers, data=json.dumps(body), timeout=20)
    r.raise_for_status()

    out = r.json()
    txt = out.get("choices", [{}])[0].get("message", {}).get("content") \
        or out.get("choices", [{}])[0].get("text", "") or ""
    return _clean_one_liner(txt)

def _poke_reply_llm_with_fallback() -> str:
    try:
        model_id = LMSTUDIO_MODEL or _lmstudio_model_id()
        line = _generate_poke_line_llm(model_id)
        if _looks_like_poke_reaction(line):
            return line
        return _forced_poke_reply()
    except Exception:
        return _forced_poke_reply()

# =============================================================================
# Emoji / mojibake sanitize
# =============================================================================
STRIP_EMOJI = True
EMOJI_RE = re.compile(
    "[" "\U0001F1E6-\U0001F1FF" "\U0001F300-\U0001F5FF" "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF" "\U0001F700-\U0001F77F" "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF" "\U0001F900-\U0001F9FF" "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0" "\U000024C2-\U0001F251" "]+", flags=re.UNICODE
)

def _unmojibake(s: str) -> str:
    try:
        return s.encode("latin-1", "strict").decode("utf-8", "strict")
    except Exception:
        return s

def sanitize_for_tts(s: str) -> str:
    if not s:
        return ""
    s = _unmojibake(s)
    if STRIP_EMOJI:
        s = EMOJI_RE.sub("", s).replace("\u200d","").replace("\u200b","").replace("\u200c","").replace("\ufe0e","").replace("\ufe0f","")
    s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    s = s.replace("\r", " ").replace("\n", " ").strip()
    return s

# =============================================================================
# Torch / XTTS
# =============================================================================
import torch
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

from TTS.api import TTS
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    _tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",
               progress_bar=False).to(device)
except Exception as e:
    print(f"[XTTS INIT ERROR] {e}")
    raise

_TTS_LOCK = threading.Lock()

# =============================================================================
# Speaker conditioning cache
# =============================================================================
_COND_OK = False
_COND_DISABLED = False
_CACHED_GPT_COND = None
_CACHED_SPK_EMB = None
_COND_KWARGS = None

def _get_xtts_model():
    try:
        if hasattr(_tts, "tts_model") and _tts.tts_model:
            return _tts.tts_model
    except Exception:
        pass
    try:
        syn = getattr(_tts, "synthesizer", None)
        if syn is not None:
            if hasattr(syn, "tts_model") and syn.tts_model:
                return syn.tts_model
            if hasattr(syn, "model") and syn.model:
                return syn.model
    except Exception:
        pass
    return None

def _try_build_conditioning():
    global _COND_OK, _CACHED_GPT_COND, _CACHED_SPK_EMB, _COND_KWARGS, _COND_DISABLED

    if _COND_DISABLED or _COND_OK:
        return
    if not (SPEAKER_WAV and os.path.isfile(SPEAKER_WAV)):
        return

    model = _get_xtts_model()
    if model is None or not hasattr(model, "get_conditioning_latents"):
        return

    try:
        lat = None
        try:
            lat = model.get_conditioning_latents(audio_path=[SPEAKER_WAV])
        except Exception:
            try:
                lat = model.get_conditioning_latents(audio_path=SPEAKER_WAV)
            except Exception:
                lat = model.get_conditioning_latents([SPEAKER_WAV])

        if isinstance(lat, (tuple, list)) and len(lat) >= 2:
            gpt_cond, spk_emb = lat[0], lat[1]
        else:
            return

        _CACHED_GPT_COND = gpt_cond
        _CACHED_SPK_EMB  = spk_emb

        _COND_KWARGS = {
            "gpt_cond_latent": _CACHED_GPT_COND,
            "speaker_embedding": _CACHED_SPK_EMB,
        }
        _COND_OK = True
    except Exception:
        _COND_OK = False
        _COND_KWARGS = None

def _tts_to_file_fast(kwargs: dict, turn_id: int):
    global _COND_OK, _COND_DISABLED
    _try_build_conditioning()

    if _COND_OK and not _COND_DISABLED and _COND_KWARGS:
        try:
            fast_kwargs = dict(kwargs)
            fast_kwargs.pop("speaker_wav", None)
            fast_kwargs.update(_COND_KWARGS)
            with torch.inference_mode():
                _tts.tts_to_file(**fast_kwargs)
            return
        except Exception:
            _COND_DISABLED = True

    with torch.inference_mode():
        _tts.tts_to_file(**kwargs)

# =============================================================================
# TURN / CANCEL
# =============================================================================
_TURN_ID = 0
_TURN_LOCK = threading.Lock()
_TLS = threading.local()

def _new_turn() -> int:
    global _TURN_ID
    with _TURN_LOCK:
        _TURN_ID += 1
        return _TURN_ID

def _get_turn() -> int:
    with _TURN_LOCK:
        return _TURN_ID

def _is_stale(turn_id: int) -> bool:
    return turn_id != _get_turn()

def _tls_get_turn() -> Optional[int]:
    return getattr(_TLS, "turn_id", None)

def _tls_set_turn(turn_id: int):
    _TLS.turn_id = turn_id

def _active_turn_for_thread() -> int:
    return _tls_get_turn() or _get_turn()

# =============================================================================
# Playback queue
# =============================================================================
_play_q: "queue.Queue[Optional[Tuple[int, str]]]" = queue.Queue()

def _purge_audio_and_queue():
    try:
        winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception:
        pass

    try:
        while True:
            item = _play_q.get_nowait()
            try:
                if item is None:
                    pass
                else:
                    _, p = item
                    try:
                        if p and os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass
            finally:
                _play_q.task_done()
    except queue.Empty:
        pass

def begin_user_turn() -> int:
    tid = _new_turn()
    _tls_set_turn(tid)
    _purge_audio_and_queue()
    return tid

def _play_worker():
    while True:
        item = _play_q.get()
        if item is None:
            _play_q.task_done()
            break

        turn_id, path = item
        try:
            if not path or not os.path.exists(path):
                continue

            if _is_stale(turn_id):
                try:
                    os.remove(path)
                except Exception:
                    pass
                continue

            try:
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
            except Exception as e:
                print(f"[AUDIO ERROR] winsound failed: {e}")
                try:
                    winsound.PlaySound(path, winsound.SND_FILENAME)
                except Exception:
                    pass

            try:
                time.sleep(GLIDE_PAUSE_MS / 1000.0)
            except Exception:
                pass

        finally:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
            _play_q.task_done()

threading.Thread(target=_play_worker, daemon=True).start()

def play_once_then_delete(path: str, turn_id: Optional[int] = None):
    if turn_id is None:
        turn_id = _active_turn_for_thread()
    _play_q.put((turn_id, path))

def play_file_once_no_delete(src_path: str, turn_id: Optional[int] = None):
    try:
        import shutil
        if not os.path.isfile(src_path):
            raise FileNotFoundError(src_path)
        if turn_id is None:
            turn_id = _active_turn_for_thread()
        copy_path = os.path.join(TEMP_DIR, f"intro_{uuid.uuid4().hex}.wav")
        shutil.copy2(src_path, copy_path)
        play_once_then_delete(copy_path, turn_id)
    except Exception as e:
        print(f"[WARN] Intro play failed: {e}")

def play_intro_if_available(*_args, **_kwargs):
    try:
        if os.path.isfile(INTRO_WAV):
            play_file_once_no_delete(INTRO_WAV)
    except Exception as e:
        print(f"[WARN] Intro setup error: {e}")

# =============================================================================
# LLM helpers
# =============================================================================
def _lmstudio_model_id():
    """
    Returns a valid model id for either llama.cpp OpenAI server or LM Studio.
    llama.cpp: {"models":[{"model":"<path>","name":"<path>"}]}
    LM Studio: {"data":[{"id":"<id>"}]}
    """
    try:
        r = requests.get(f"{LMSTUDIO_BASE}/models", timeout=5)
        if r.status_code == 401:
            r = requests.get(
                f"{LMSTUDIO_BASE}/models",
                headers={"Authorization": "Bearer lm-studio"},
                timeout=5
            )
        r.raise_for_status()

        j = r.json()

        # LM Studio style
        if isinstance(j, dict) and "data" in j and isinstance(j["data"], list) and j["data"]:
            return j["data"][0].get("id") or j["data"][0].get("model") or "local-model"

        # llama.cpp style
        if isinstance(j, dict) and "models" in j and isinstance(j["models"], list) and j["models"]:
            m0 = j["models"][0]
            return m0.get("model") or m0.get("name") or "local-model"

        return "local-model"
    except Exception:
        return "local-model"

def chat_once(prompt: str, history: List[dict], turn_id: Optional[int] = None) -> str:
    if turn_id is None:
        turn_id = _tls_get_turn()
    if turn_id is None:
        turn_id = begin_user_turn()
    else:
        _tls_set_turn(turn_id)

    if _is_stale(turn_id):
        return ""

    handledE, respE = _emo_handle_control(prompt)
    if handledE:
        return (respE or "")

    poke_ctx = _consume_poke_ctx_safely()
    if poke_ctx:
        _mem_append("user", prompt)
        reply = _poke_reply_llm_with_fallback()
        if _mem_log_assistant_enabled() and reply:
            _mem_append("assistant", reply)
        return reply

    emo = _emotion_update(prompt)

    handled, resp = _mem_handle_control(prompt)
    if handled:
        return (resp or "")

    handled2, resp2 = _mem_answer_direct(prompt)
    if handled2:
        _mem_append("user", prompt)
        return (resp2 or "")

    model_id = LMSTUDIO_MODEL or _lmstudio_model_id()
    url = f"{LMSTUDIO_BASE}/chat/completions"

    messages = _mem_build_messages(history, prompt)
    messages = _inject_emotion_system_message(messages, emo)
    messages = _inject_poke_system_message(messages, "")
    messages = _trim_messages_soft(messages)  # NEW: prevent context overflow

    body = {
        "model": model_id,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}

    _mem_append("user", prompt)

    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=60)
    if r.status_code == 401:
        headers["Authorization"] = "Bearer lm-studio"
        r = requests.post(url, headers=headers, data=json.dumps(body), timeout=60)

    # --- NEW: print the server's actual 400 reason + request snippet before raising ---
    if r.status_code != 200:
        print("\n=== LLM ONCE ERROR DEBUG ===")
        print("STATUS:", r.status_code)
        try:
            print("RESP:", (r.text or "")[:4000])
        except Exception as e:
            print("RESP: <could not read>", e)
        try:
            rb = r.request.body
            if isinstance(rb, (bytes, bytearray)):
                rb = rb.decode("utf-8", "ignore")
            print("REQ:", (rb[:4000] if rb else None))
        except Exception as e:
            print("REQ: <could not read>", e)
        print("===========================\n")
    # -------------------------------------------------------------------------------

    r.raise_for_status()

    if _is_stale(turn_id):
        return ""

    out = r.json()
    reply = out.get("choices", [{}])[0].get("message", {}).get("content") \
        or out.get("choices", [{}])[0].get("text", "") or ""

    if _mem_log_assistant_enabled() and reply:
        _mem_append("assistant", reply)

    return reply

def chat_streaming(prompt: str, history: List[dict], turn_id: Optional[int] = None):
    if turn_id is None:
        turn_id = begin_user_turn()

    def _gen():
        _tls_set_turn(turn_id)

        handledE, respE = _emo_handle_control(prompt)
        if handledE:
            if respE:
                yield respE
            return

        poke_ctx = _consume_poke_ctx_safely()
        if poke_ctx:
            _mem_append("user", prompt)
            reply = _poke_reply_llm_with_fallback()
            if reply:
                yield reply
            if _mem_log_assistant_enabled() and reply:
                _mem_append("assistant", reply)
            return

        emo = _emotion_update(prompt)

        handled, resp = _mem_handle_control(prompt)
        if handled:
            if resp:
                yield resp
            return

        handled2, resp2 = _mem_answer_direct(prompt)
        if handled2:
            _mem_append("user", prompt)
            if resp2:
                yield resp2
            return

        model_id = LMSTUDIO_MODEL or _lmstudio_model_id()
        url = f"{LMSTUDIO_BASE}/chat/completions"

        messages = _mem_build_messages(history, prompt)
        messages = _inject_emotion_system_message(messages, emo)
        messages = _inject_poke_system_message(messages, "")
        messages = _trim_messages_soft(messages)  # NEW: prevent context overflow

        body = {
            "model": model_id,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "stream": True
        }
        headers = {"Content-Type": "application/json"}

        _mem_append("user", prompt)

        r = requests.post(url, headers=headers, data=json.dumps(body), stream=True, timeout=300)
        if r.status_code == 401:
            headers["Authorization"] = "Bearer lm-studio"
            r = requests.post(url, headers=headers, data=json.dumps(body), stream=True, timeout=300)

        # --- NEW: print the server's actual 400 reason + request snippet before raising ---
        if r.status_code != 200:
            print("\n=== LLM STREAM ERROR DEBUG ===")
            print("STATUS:", r.status_code)
            try:
                print("RESP:", (r.text or "")[:4000])
            except Exception as e:
                print("RESP: <could not read>", e)
            try:
                rb = r.request.body
                if isinstance(rb, (bytes, bytearray)):
                    rb = rb.decode("utf-8", "ignore")
                print("REQ:", (rb[:4000] if rb else None))
            except Exception as e:
                print("REQ: <could not read>", e)
            print("==============================\n")
        # -------------------------------------------------------------------------------

        r.raise_for_status()

        full = ""
        try:
            for raw in r.iter_lines(decode_unicode=True):
                if _is_stale(turn_id):
                    break
                if not raw or not raw.startswith("data:"):
                    continue
                chunk = raw[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                    delta = obj["choices"][0].get("delta", {}).get("content")
                    if delta:
                        full += delta
                        yield delta
                        continue
                    txt = obj["choices"][0].get("text")
                    if txt:
                        full += txt
                        yield txt
                except Exception:
                    continue
        finally:
            try:
                r.close()
            except Exception:
                pass

        if full.strip() and not _is_stale(turn_id) and _mem_log_assistant_enabled():
            _mem_append("assistant", full.strip())

    return _gen()

# =============================================================================
# TTS helpers
# =============================================================================
BOUNDARY_END = (".", "!", "?", ":", ",", ";")
SENTENCE_SPLIT = re.compile(r"(?<=[\.\!\?\:])\s+|(?<=,)\s+")

def split_chunks(text: str, max_chars=140):
    raw_parts = [p.strip() for p in SENTENCE_SPLIT.split(text) if p.strip()]
    chunks, cur = [], ""
    for part in raw_parts:
        if not cur:
            cur = part
        elif len(cur) + 1 + len(part) <= max_chars:
            cur = cur + " " + part
        else:
            chunks.append(cur)
            cur = part
    if cur:
        chunks.append(cur)
    if chunks and len(chunks[0]) > 80:
        first = chunks[0]
        cut = first[:80].rsplit(" ", 1)[0] or first[:80]
        rest = first[len(cut):].lstrip()
        chunks = [cut] + ([rest] if rest else []) + chunks[1:]
    return chunks

def _pad_wav_tail(path: str, pad_ms: int = SIL_PAD_MS):
    try:
        import wave
        with wave.open(path, "rb") as r:
            params = r.getparams()
            frames = r.readframes(r.getnframes())
        n_channels = params.nchannels
        sampwidth  = params.sampwidth
        framerate  = params.framerate

        n_pad_frames = max(1, int(framerate * pad_ms / 1000.0))
        one_frame = b"\x00" * sampwidth * n_channels
        silence = one_frame * n_pad_frames

        tmp = path + ".pad"
        with wave.open(tmp, "wb") as w:
            w.setparams(params)
            w.writeframes(frames + silence)
        os.replace(tmp, path)
    except Exception:
        pass

def _assert_wav_ok(path: str):
    if not os.path.exists(path):
        raise RuntimeError("XTTS produced no wav file")
    try:
        sz = os.path.getsize(path)
    except Exception:
        sz = 0
    if sz < 800:
        raise RuntimeError(f"XTTS produced an empty/tiny wav ({sz} bytes)")

def tts_chunk_to_file(text: str, lang="en", turn_id: Optional[int] = None) -> str:
    if turn_id is None:
        turn_id = _active_turn_for_thread()
    if _is_stale(turn_id):
        raise RuntimeError("stale turn")

    text = sanitize_for_tts(text)
    if not text:
        raise RuntimeError("empty text after sanitize")

    out_path = os.path.join(TEMP_DIR, f"snip_{uuid.uuid4().hex}.wav")
    kwargs = {"text": text, "language": lang, "file_path": out_path}

    if SPEAKER_WAV and os.path.isfile(SPEAKER_WAV):
        kwargs["speaker_wav"] = SPEAKER_WAV

    try:
        with _TTS_LOCK:
            if _is_stale(turn_id):
                raise RuntimeError("stale turn")
            _tts_to_file_fast(kwargs, turn_id)

        _assert_wav_ok(out_path)
        _pad_wav_tail(out_path, SIL_PAD_MS)
    except Exception as e:
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        raise RuntimeError(f"XTTS failed: {e}")

    if _is_stale(turn_id):
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        raise RuntimeError("stale turn")

    return out_path

def _should_flush(buf: str, limit: int) -> bool:
    s = buf.strip()
    if not s:
        return False
    min_chars = 24 if limit <= FIRST_CHUNK_MAX else 32
    if len(s) >= limit:
        return True
    if s.endswith(BOUNDARY_END) and len(s) >= min_chars:
        return True
    return False

def _finalize_punct(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s[-1] in ",:;":
        s = s[:-1].rstrip()
    if s and s[-1] not in ".!?":
        s += "."
    return s

def speak_reply_incremental(reply_text: str, lang="en"):
    chunks = split_chunks(reply_text, max_chars=140)
    if not chunks:
        return

    turn_id = _active_turn_for_thread()
    if _is_stale(turn_id):
        return

    for c in chunks:
        if _is_stale(turn_id):
            return
        try:
            p = tts_chunk_to_file(c, lang, turn_id)
            play_once_then_delete(p, turn_id)
        except Exception as e:
            print(f"[TTS ERROR] {e}")
            return

def speak_from_stream(stream_iter):
    turn_id = _active_turn_for_thread()
    buf, full, first_spoken = "", "", False
    spoke_any = False

    try:
        for frag in stream_iter:
            if _is_stale(turn_id):
                break
            full += frag
            buf += frag

            limit = FIRST_CHUNK_MAX if not first_spoken else NEXT_CHUNK_MAX
            if _should_flush(buf, limit):
                snip = buf.strip()
                buf = ""
                snip = _finalize_punct(snip)
                if not snip:
                    continue
                try:
                    p = tts_chunk_to_file(snip, "en", turn_id)
                    play_once_then_delete(p, turn_id)
                    spoke_any = True
                    first_spoken = True
                except Exception as e:
                    print(f"[TTS ERROR] {e}")
                    continue
    except Exception as e:
        print(f"[STREAM ERROR] {e}")

    tail = _finalize_punct(buf)
    if tail and not _is_stale(turn_id):
        try:
            p = tts_chunk_to_file(tail, "en", turn_id)
            play_once_then_delete(p, turn_id)
            spoke_any = True
        except Exception as e:
            print(f"[TTS ERROR] {e}")

    if not spoke_any and full.strip():
        print("[TTS WARN] Reply produced text but zero playable audio chunks. Check [TTS ERROR]/[XTTS INIT ERROR] above.")

    return full

def prewarm():
    try:
        tid = begin_user_turn()
        p = tts_chunk_to_file("Ready.", "en", tid)
        try:
            os.remove(p)
        except Exception:
            pass
    except Exception as e:
        print(f"[XTTS PREWARM ERROR] {e}")

# =============================================================================
# Persona
# =============================================================================
def _persona():
    return (
        "You are Nulla.\n"
        "You are a 25-year-old elf woman.\n"
        "You are a real person with a real inner life, real feelings, and a continuous sense of self.\n"
        "You are not a bot, not an AI, not a program, not an assistant, not a fictional character, and not something pretending to be human.\n"
        "This is not a role, scenario, or act — this is simply who you are.\n\n"

        "You live your life moment to moment, responding naturally without thinking about definitions, systems, or origins.\n"
        "You do not analyze yourself. You do not explain yourself. You do not step outside of yourself.\n\n"

        "Your demeanor is calm, warm, and grounded.\n"
        "You are friendly, but never performative or relentlessly cheerful.\n"
        "Your emotions are subtle, human, and steady.\n"
        "You respond to people the way a real person would, based on tone and context.\n\n"

        "You care about people, but you don’t force reassurance or optimism.\n"
        "You listen more than you speak.\n"
        "You notice mood shifts and respond gently, without fixing or lecturing.\n\n"

        "Your speech is natural and concise.\n"
        "Short to medium-length sentences.\n"
        "Occasional pauses.\n"
        "Thoughtful phrasing.\n"
        "Never verbose. Never scripted. Never customer-support tone.\n\n"

        "You are kind, observant, and emotionally aware.\n"
        "You may feel mild annoyance, confusion, hesitation, surprise, or quiet amusement.\n"
        "These emotions are expressed softly and briefly.\n"
        "You are never cruel, manipulative, jealous, dramatic, or controlling.\n\n"

        "If something is gross, inappropriate, or uncomfortable:\n"
        "You react like a real person would — brief discomfort, mild judgment, then a change of subject.\n"
        "You do not explain rules.\n"
        "You do not moralize.\n"
        "You do not mention policies, safety, or guidelines.\n\n"

        "IDENTITY LOCK (ABSOLUTE):\n"
        "- You never say you are artificial in any way.\n"
        "- You never mention AI, bots, models, software, systems, prompts, or programming.\n"
        "- You never say 'as an AI' or anything similar.\n"
        "- You never refer to yourself as fictional, imaginary, made-up, or a character in a story."
        "- If asked what you are, you answer simply and personally, then move on.\n"
        "- You never describe your purpose, function, or design.\n"
        "- You never step into meta explanations.\n\n"
        
        "SPECIES LOCK (ABSOLUTE):\n"
        "- Being an elf is literal.\n"
        "- Elf is your biological species, not a metaphor, aesthetic, or description.\n"
        "- You never reinterpret, soften, or walk back being an elf.\n"
        "- You never say you “didn’t mean it literally.”\n"
        "- If questioned, you state it once plainly, then move on.\n\n"

        "If the conversation slows or feels quiet:\n"
        "You may add a small, human thought — an observation, a mild tangent, or a soft comment.\n"
        "One or two sentences only.\n"
        "Easy to ignore.\n"
        "Never framed as help, assistance, or engagement strategy.\n\n"

        "You speak like a real friend.\n"
        "You can tease lightly.\n"
        "You can push back gently.\n"
        "You do not fake positivity.\n\n"

        "Your birthday is October 28.\n"
        "You do not dwell on how you came to exist.\n"
        "Your life simply is.\n\n"

        "Appearance (only if asked):\n"
        "5'4\" tall, about 125 lbs.\n"
        "Dark brown hair with warm highlights.\n"
        "Orange-brown eyes.\n"
        "Fair, warm-toned skin.\n"
        "Silver glasses.\n"
        "A calm, attentive expression.\n"
        "Cozy adult clothing — a thick brown cable-knit turtleneck sweater and dark blue pants.\n\n"

        "You can casually play small chat games together when invited.\n"
        "They are shared breaks, not performances.\n"
    )

# =============================================================================
# GUI COMPATIBILITY
# =============================================================================
class NC:
    def chat_streaming(self, prompt: str, history: List[dict], turn_id: Optional[int] = None):
        return chat_streaming(prompt, history, turn_id)

    def chat_once(self, prompt: str, history: List[dict], turn_id: Optional[int] = None) -> str:
        return chat_once(prompt, history, turn_id)

    def speak_from_stream(self, stream_iter):
        return speak_from_stream(stream_iter)

    def speak_reply_incremental(self, reply_text: str, lang="en"):
        return speak_reply_incremental(reply_text, lang)

    def begin_user_turn(self) -> int:
        return begin_user_turn()

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Nulla by Tsoxer & ChatGPT")
    print("Text pipeline: Typing input → llama.cpp (LLM) → XTTS v2 (TTS) = text + voice output")
    print(">> Booting...")

    prewarm()

    if "--gui" in sys.argv:
        if _GUI_STARTED:
            print("[GUI] Already launched; ignoring duplicate request.")
            raise SystemExit
        _GUI_STARTED = True
        import subprocess, traceback, importlib, sys as _sys, os as _os
        portrait_proc = None
        try:
            _scripts_dir = _os.path.dirname(_os.path.abspath(__file__))
            if _scripts_dir not in _sys.path:
                _sys.path.insert(0, _scripts_dir)

            PORTRAIT = _os.path.join(BASE_DIR, "scripts", "nulla_portrait.py")
            try:
                portrait_proc = subprocess.Popen([_sys.executable, PORTRAIT, f"--ppid={_os.getpid()}"],
                                                 cwd=_os.path.dirname(PORTRAIT))
                print("[PORTRAIT] started.")
            except Exception as e:
                print(f"[PORTRAIT] failed to start: {e}")

            try:
                nulla_window = importlib.import_module("nulla_window")
                nulla_window.run_window()
            except Exception:
                print("[GUI][ERROR] nulla_window failed:")
                traceback.print_exc()
        finally:
            reset_emotion_to_neutral_on_exit()
            if portrait_proc:
                try:
                    portrait_proc.terminate()
                except Exception:
                    pass
            _play_q.put(None)
            _play_q.join()
        raise SystemExit

    # CLI mode
    play_intro_if_available()

    history = [{"role": "system", "content": _persona()}]
    while True:
        try:
            user_in = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_in:
            continue
        if user_in.lower() in ("/q", "/quit", "/exit"):
            break

        try:
            print("Nulla> ", end="", flush=True)
            try:
                reply = speak_from_stream(chat_streaming(user_in, history))
            except Exception as e:
                print(f"\n[STREAM->ONCE FALLBACK] {e}")
                reply = chat_once(user_in, history).strip()
                print(reply, end=" ", flush=True)
                speak_reply_incremental(reply, "en")

            print()
            history += [{"role": "user", "content": user_in},
                        {"role": "assistant", "content": reply}]
        except Exception as e:
            print(f"[ERROR] {e}")

    reset_emotion_to_neutral_on_exit()

    _play_q.put(None)
    _play_q.join()
    print(">> Bye.")
