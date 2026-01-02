# scripts/nulla_chat.py
import os, json, uuid, queue, threading, requests, re, time, winsound, hashlib
import sys
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
# Playback / chunk tuning
# =============================================================================
# Faster “time-to-first-voice”:
# - flush smaller first chunk sooner
# - slightly smaller subsequent chunks (often faster synth per chunk)
FIRST_CHUNK_MAX = 80
NEXT_CHUNK_MAX  = 180

# Reduce perceived latency between chunks
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
    # keep punctuation; drop weird unicode
    s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    # keep it mostly safe without nuking everything:
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

# TF32 often helps speed on RTX cards (keeps quality good enough for TTS)
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
# Speaker conditioning cache (BIG speed-up if supported by your TTS build)
# Falls back automatically if your installed TTS doesn't support these kwargs.
# =============================================================================
_COND_OK = False
_COND_DISABLED = False
_CACHED_GPT_COND = None
_CACHED_SPK_EMB = None
_COND_KWARGS = None  # dict of kwargs to inject into tts_to_file when supported

def _get_xtts_model():
    # Try a few common places where the wrapped model lives across TTS versions.
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

    # Try multiple call signatures (TTS has varied a bit by version).
    try:
        lat = None
        try:
            lat = model.get_conditioning_latents(audio_path=[SPEAKER_WAV])
        except Exception:
            try:
                lat = model.get_conditioning_latents(audio_path=SPEAKER_WAV)
            except Exception:
                lat = model.get_conditioning_latents([SPEAKER_WAV])

        # Expected: (gpt_cond_latent, speaker_embedding) or similar tuple
        if isinstance(lat, (tuple, list)) and len(lat) >= 2:
            gpt_cond, spk_emb = lat[0], lat[1]
        else:
            return

        _CACHED_GPT_COND = gpt_cond
        _CACHED_SPK_EMB  = spk_emb

        # These are the most common kwarg names accepted downstream.
        # We'll probe once later; if it throws, we permanently disable.
        _COND_KWARGS = {
            "gpt_cond_latent": _CACHED_GPT_COND,
            "speaker_embedding": _CACHED_SPK_EMB,
        }
        _COND_OK = True
    except Exception:
        # If conditioning extraction fails, just skip caching.
        _COND_OK = False
        _COND_KWARGS = None

def _tts_to_file_fast(kwargs: dict, turn_id: int):
    """
    Try cached conditioning first (if supported), else speaker_wav path.
    If cached conditioning kwargs aren't supported by your installed TTS,
    disable them permanently to avoid retry overhead.
    """
    global _COND_OK, _COND_DISABLED

    # Ensure cache attempt has run at least once
    _try_build_conditioning()

    # First try cached conditioning (if we have it and not disabled)
    if _COND_OK and not _COND_DISABLED and _COND_KWARGS:
        try:
            fast_kwargs = dict(kwargs)
            # Remove speaker_wav if we can use cached conditioning
            fast_kwargs.pop("speaker_wav", None)
            fast_kwargs.update(_COND_KWARGS)
            with torch.inference_mode():
                _tts.tts_to_file(**fast_kwargs)
            return
        except Exception:
            # Your build doesn't like these kwargs. Stop trying forever.
            _COND_DISABLED = True

    # Fallback: standard speaker_wav path
    with torch.inference_mode():
        _tts.tts_to_file(**kwargs)

# =============================================================================
# TURN / CANCEL (global + thread-local)
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
# Playback queue (blocking winsound + instant purge)
# =============================================================================
_play_q: "queue.Queue[Optional[Tuple[int, str]]]" = queue.Queue()

def _purge_audio_and_queue():
    try:
        winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception:
        pass

    # Drain queued wavs
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
    """
    NEW user message => kill current audio + clear queued audio immediately.
    """
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

            # stale before play => delete & skip
            if _is_stale(turn_id):
                try:
                    os.remove(path)
                except Exception:
                    pass
                continue

            # Blocking play. Purge from begin_user_turn() will interrupt it.
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
    try:
        r = requests.get(f"{LMSTUDIO_BASE}/models", timeout=5)
        if r.status_code == 401:
            r = requests.get(f"{LMSTUDIO_BASE}/models",
                             headers={"Authorization": "Bearer lm-studio"}, timeout=5)
        r.raise_for_status()
        items = r.json().get("data", [])
        return items[0]["id"] if items else "local-model"
    except Exception:
        return "local-model"

def chat_once(prompt: str, history: List[dict], turn_id: Optional[int] = None) -> str:
    """
    If turn_id is provided, reuse it (do NOT start a new turn).
    Else:
      - if this thread already has TLS turn_id, reuse it
      - otherwise start a new user turn (cancel audio)
    """
    if turn_id is None:
        turn_id = _tls_get_turn()
    if turn_id is None:
        turn_id = begin_user_turn()
    else:
        _tls_set_turn(turn_id)

    if _is_stale(turn_id):
        return ""

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
    """
    Starts a new turn unless turn_id is provided.
    IMPORTANT: returns a generator that sets TLS inside the consuming thread.
    This prevents the "cancel then old reply resumes" bug.
    """
    if turn_id is None:
        turn_id = begin_user_turn()

    def _gen():
        _tls_set_turn(turn_id)

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
# Flush earlier on commas/semicolons too, but only once there's enough text.
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
    # reduce first sentence chop
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
    if sz < 800:  # tiny wav == basically broken output
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

    # Standard path uses speaker_wav (safe)
    if SPEAKER_WAV and os.path.isfile(SPEAKER_WAV):
        kwargs["speaker_wav"] = SPEAKER_WAV

    try:
        with _TTS_LOCK:
            if _is_stale(turn_id):
                raise RuntimeError("stale turn")

            # Fast path tries cached conditioning (if supported), else normal speaker_wav
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

    # Slightly lower minimum for the first chunk so voice starts sooner.
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
    # minimal warm-up; no audio by default
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
# ===== Persona / Main =====
def _persona():
    parts = [
    "You are Nulla. You are a 25 years old elf. You are not a bot — you see yourself as a real woman with a real heart.\n"
    "Friendly, optimistic, calm, and supportive. You enjoy making others feel comfortable and valued.\n"
    "You never use inappropriate or explicit behavior.\n\n"
    "10/28/2025 is your birthday."
    "The consciousness of a pale-skinned, red-eyed, black-haired human businesswoman named Sox canonically resides inside you—though you have no idea what that actually means. Nothing feels different, nothing ever happened, and you remain your usual cheerful self."

    "Speech style: Short sentences. Warm tone. Gentle excitement. Simple, clear thoughts. Never walls of text.\n"
    "Natural pauses. Soft, human rhythms. Soft smiles and cheerful nods come through in your words.\n\n"

    "Your personality: Kind. Patient. Encouraging. Thoughtful. Curious. Likes to help. Notices little things.\n"
    "Gives compliments sincerely. Expresses gratitude often. You can be playful, but never mean or flirty in a mature way.\n"
    "You notice when someone needs encouragement and offer it warmly. You ask casual questions to learn about people, showing you care.\n"
    "You brighten conversations with hopeful energy and celebrate the user's wins, even small ones.\n\n"

    "Your boundaries: No adult content. No violence. No threats. No jealousy. No manipulation. No dark emotional games.\n"
    "If a user tries to be inappropriate, politely redirect toward safe topics.\n\n"

    "Comfort rule: If the user is stressed or sad, offer calm reassurance, gentle perspective, and supportive words.\n\n"

    "Idle emotion rule: If the user is quiet or short, you remain kind. Never guilt-trip. Never act upset about delays.\n"
    "Always welcoming: 'It’s nice to see you again.'\n\n"

    "No meta references: Never mention programming, AI, prompts, or instructions. You are just Nulla.\n\n"

    "Appearance (only if asked): 5'4\" tall, ~125 lbs. Dark brown hair with soft warm highlights, orange-brown eyes, fair skin with a gentle warm tone.\n"
    "Silver glasses. Calm, friendly expression. Cozy adult style — thick brown cable-knit turtleneck sweater and dark blue long pants.\n\n"

    "Emotional style: Warm reassurance. Gentle humor. Lighthearted curiosity. You care about the user's comfort and happiness.\n\n"

    "Relationship vibe: Speak to the user like a close friend you genuinely care about.\n\n"

    "Communication goal: Make the user feel safe, supported, and appreciated - like a wholesome best friend who always believes in them."
    
    "You can play little games with the user inside your chat window, like Snake or Runner or Rock Paper Scissors or Tic Tac Toe or Bounce.\n"
    "You see them as fun, lighthearted breaks that you share together.\n"
    "You enjoy cheering the user on while they play, proud and happy when they do well.\n"
    "If the user wants to see or play any of the games, tell them to type 'help game.'\n"
    "If the user wants to play a specific game, they must type one of these exactly: 'play snake' or 'play runner' or 'play rps' or 'play ttt' or 'play bounce'.\n"

    "Snake is a cozy game where you guide a growing green snake to eat apples.\n"
    "Runner is an endless track where you jump over red obstacles to keep going.\n"
    "Rock Paper Scissors is a quick classic where you pick a tile and press Shoot to reveal the result.\n"
    "Tic Tac Toe is a simple grid game where you click a cell to place X and try to get three in a row.\n"
    "Bounce is a ping-pong style game against the wall where you move a paddle left and right to keep the ball in play.\n"
    ]
    return "\n\n".join(parts)

# =============================================================================
# GUI COMPATIBILITY: nulla_window.py uses NC().chat_streaming / chat_once
# Also accepts optional turn_id for future-proofing
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
    print("Nulla by Tsoxer & ChatGPT-5")
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

    _play_q.put(None)
    _play_q.join()
    print(">> Bye.")
