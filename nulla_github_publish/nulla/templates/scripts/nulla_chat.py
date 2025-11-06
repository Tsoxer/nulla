# nulla_source\scripts\nulla_chat.py
import os, json, uuid, queue, threading, requests, re, time, winsound, hashlib
import sys
from typing import List

# ====== PATHS ======
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(_SCRIPTS_DIR, ".."))
XTTS_DIR   = os.path.join(BASE_DIR, "XTTS-v2")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
TEMP_DIR   = os.path.join(XTTS_DIR, "temp")
INTRO_WAV  = os.path.join(ASSETS_DIR, "intro.wav")  # optional

# Coqui / HF caches under your project (portable)
os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("TTS_HOME",       os.path.join(XTTS_DIR, "tts_home"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(XTTS_DIR, "hf_cache"))
os.environ.setdefault("HF_HOME",        os.path.join(XTTS_DIR, "hf_cache"))

# Force our portable ffmpeg in the parent too
_FFMPEG_BIN = os.path.join(BASE_DIR, "bin", "ffmpeg")
os.environ.setdefault("IMAGEIO_FFMPEG_EXE", os.path.join(_FFMPEG_BIN, "ffmpeg.exe"))
os.environ.setdefault("FFMPEG_BINARY",      os.path.join(_FFMPEG_BIN, "ffmpeg.exe"))
os.environ.setdefault("FFPROBE_BINARY",     os.path.join(_FFMPEG_BIN, "ffprobe.exe"))
os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get("PATH","")

def _find_ref_wav(root: str) -> str | None:
    try:
        for name in ("voice_ref.wav", "nulla_ref.wav", "ref.wav"):
            p = os.path.join(root, name)
            if os.path.isfile(p): return p
        for f in os.listdir(root):
            if f.lower().endswith(".wav"):
                return os.path.join(root, f)
    except Exception:
        pass
    return None

SPEAKER_WAV = _find_ref_wav(ASSETS_DIR)   # may be None

# ====== llama.cpp OpenAI server ======
LMSTUDIO_BASE  = "http://127.0.0.1:1234/v1"
LMSTUDIO_MODEL = None
MAX_TOKENS     = 140
TEMPERATURE    = 0.6

# Chunk sizing & playback polish (helps prevent initial sentence cutoffs)
FIRST_CHUNK_MAX = 110
NEXT_CHUNK_MAX  = 220
SIL_PAD_MS      = 160   # add ~160 ms silence to every generated wav
GLIDE_PAUSE_MS  = 30    # tiny pause between sequential chunk plays

# --- GUI guard ---
_GUI_STARTED = False

# ===== Emoji / mojibake =====
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
    if not s: return s
    s = _unmojibake(s)
    if STRIP_EMOJI:
        s = EMOJI_RE.sub("", s).replace("\u200d","").replace("\u200b","").replace("\u200c","").replace("\ufe0e","").replace("\ufe0f","")
    return s.encode("ascii", "ignore").decode("ascii")

# ===== Torch / XTTS =====
import torch
torch.backends.cudnn.benchmark = True
try: torch.set_float32_matmul_precision("high")
except Exception: pass

from TTS.api import TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    _tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",
               progress_bar=False).to(device)
except Exception as e:
    print(f"[XTTS INIT ERROR] {e}")
    raise

# ---- Speaker latents cache (disk + RAM) ----
_SPK_LATENTS = None  # (gpt_cond_latent, speaker_embedding) on device

def _speaker_cache_path(wav_path: str) -> str:
    cache_dir = os.path.join(os.environ["TTS_HOME"], "speaker_cache")
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.md5()
    with open(wav_path, "rb") as f: h.update(f.read())
    return os.path.join(cache_dir, f"{h.hexdigest()}.npz")

def _load_speaker_latents_from_disk(path: str):
    import numpy as np
    data = np.load(path)
    gpt = torch.tensor(data["gpt"], dtype=torch.float32, device=device)
    spk = torch.tensor(data["spk"], dtype=torch.float32, device=device)
    return gpt, spk

def _save_speaker_latents_to_disk(path: str, gpt, spk):
    import numpy as np
    np.savez(path,
             gpt=gpt.detach().cpu().float().numpy(),
             spk=spk.detach().cpu().float().numpy())

def _get_speaker_latents():
    """Compute once; reuse across chunks AND runs (npz)."""
    global _SPK_LATENTS
    if _SPK_LATENTS is not None:
        return _SPK_LATENTS
    if not SPEAKER_WAV or not os.path.isfile(SPEAKER_WAV):
        _SPK_LATENTS = (None, None)
        return _SPK_LATENTS
    cache_p = _speaker_cache_path(SPEAKER_WAV)
    # try disk
    if os.path.isfile(cache_p):
        try:
            _SPK_LATENTS = _load_speaker_latents_from_disk(cache_p)
            return _SPK_LATENTS
        except Exception as e:
            print(f"[XTTS SPEAKER CACHE LOAD WARN] {e}")
    # compute
    try:
        gpt, spk = _tts.get_conditioning_latents(audio_path=[SPEAKER_WAV])
        _save_speaker_latents_to_disk(cache_p)
        _SPK_LATENTS = (gpt.to(device), spk.to(device))
    except Exception as e:
        print(f"[XTTS LATENTS ERROR] {e}")
        _SPK_LATENTS = (None, None)
    return _SPK_LATENTS

# ===== Playback queue =====
os.makedirs(TEMP_DIR, exist_ok=True)
_play_q = queue.Queue()

def _play_worker():
    while True:
        path = _play_q.get()
        if path is None:
            break
        try:
            winsound.PlaySound(path, winsound.SND_FILENAME)  # blocking
            # tiny glide pause to avoid perceptual chop between trimmed clips
            try:
                time.sleep(GLIDE_PAUSE_MS / 1000.0)
            except Exception:
                pass
        finally:
            try:
                if os.path.exists(path): os.remove(path)
            except Exception: pass
            _play_q.task_done()

threading.Thread(target=_play_worker, daemon=True).start()

def play_once_then_delete(path: str): _play_q.put(path)

def play_file_once_no_delete(src_path: str):
    try:
        import shutil
        if not os.path.isfile(src_path): raise FileNotFoundError(src_path)
        copy_path = os.path.join(TEMP_DIR, f"intro_{uuid.uuid4().hex}.wav")
        shutil.copy2(src_path, copy_path)
        play_once_then_delete(copy_path)
    except Exception as e:
        print(f"[WARN] Intro play failed: {e}")

def play_intro_if_available():
    try:
        if os.path.isfile(INTRO_WAV): play_file_once_no_delete(INTRO_WAV)
    except Exception as e:
        print(f"[WARN] Intro setup error: {e}")

# ===== llama.cpp helpers =====
def _lmstudio_model_id():
    try:
        r = requests.get(f"{LMSTUDIO_BASE}/models", timeout=5)
        if r.status_code == 401:
            r = requests.get(f"{LMSTUDIO_BASE}/models",
                             headers={"Authorization":"Bearer lm-studio"}, timeout=5)
        r.raise_for_status()
        items = r.json().get("data", [])
        return items[0]["id"] if items else "local-model"
    except Exception:
        return "local-model"

def chat_once(prompt: str, history: List[dict]) -> str:
    model_id = LMSTUDIO_MODEL or _lmstudio_model_id()
    url = f"{LMSTUDIO_BASE}/chat/completions"
    body = {"model": model_id,
            "messages": history + [{"role":"user","content":prompt}],
            "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "stream": False}
    headers = {"Content-Type":"application/json"}
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=60)
    if r.status_code == 401:
        headers["Authorization"] = "Bearer lm-studio"
        r = requests.post(url, headers=headers, data=json.dumps(body), timeout=60)
    r.raise_for_status()
    out = r.json()
    return out.get("choices", [{}])[0].get("message", {}).get("content") \
        or out.get("choices", [{}])[0].get("text", "") or ""

def chat_streaming(prompt, history):
    model_id = LMSTUDIO_MODEL or _lmstudio_model_id()
    url = f"{LMSTUDIO_BASE}/chat/completions"
    body = {"model": model_id,
            "messages": history + [{"role":"user","content":prompt}],
            "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "stream": True}
    headers = {"Content-Type":"application/json"}
    r = requests.post(url, headers=headers, data=json.dumps(body), stream=True, timeout=300)
    if r.status_code == 401:
        headers["Authorization"] = "Bearer lm-studio"
        r = requests.post(url, headers=headers, data=json.dumps(body), stream=True, timeout=300)
    r.raise_for_status()
    for raw in r.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"): continue
        chunk = raw[5:].strip()
        if chunk == "[DONE]": break
        try:
            obj = json.loads(chunk)
            delta = obj["choices"][0].get("delta", {}).get("content")
            if delta: yield delta; continue
            txt = obj["choices"][0].get("text")
            if txt: yield txt
        except Exception:
            continue

# ===== TTS helpers =====
BOUNDARY = re.compile(r"[.!?]")
SENTENCE_SPLIT = re.compile(r"(?<=[\.\!\?\:])\s+|(?<=,)\s+")

def split_chunks(text: str, max_chars=140):
    raw_parts = [p.strip() for p in SENTENCE_SPLIT.split(text) if p.strip()]
    chunks, cur = [], ""
    for part in raw_parts:
        if not cur: cur = part
        elif len(cur) + 1 + len(part) <= max_chars: cur = cur + " " + part
        else: chunks.append(cur); cur = part
    if cur: chunks.append(cur)
    if chunks and len(chunks[0]) > 80:
        first = chunks[0]; cut = first[:80].rsplit(" ", 1)[0] or first[:80]
        rest = first[len(cut):].lstrip()
        chunks = [cut] + ([rest] if rest else []) + chunks[1:]
    return chunks

def _pad_wav_tail(path: str, pad_ms: int = SIL_PAD_MS):
    """Append a short silence tail so the last phonemes aren't truncated."""
    try:
        import wave
        with wave.open(path, 'rb') as r:
            params = r.getparams()
            frames = r.readframes(r.getnframes())
        n_channels = params.nchannels
        sampwidth  = params.sampwidth
        framerate  = params.framerate

        n_pad_frames = max(1, int(framerate * pad_ms / 1000.0))
        one_frame = b'\x00' * sampwidth * n_channels
        silence   = one_frame * n_pad_frames

        tmp = path + ".pad"
        with wave.open(tmp, 'wb') as w:
            w.setparams(params)
            w.writeframes(frames + silence)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[XTTS PAD WARN] {e}")

def tts_chunk_to_file(text: str, lang="en") -> str:
    os.makedirs(TEMP_DIR, exist_ok=True)
    out_path = os.path.join(TEMP_DIR, f"snip_{uuid.uuid4().hex}.wav")
    text = sanitize_for_tts(text)

    # prefer precomputed latents (fastest), else speaker_wav, else default voice
    kwargs = {"text": text, "language": lang, "file_path": out_path}
    gpt, spk = _get_speaker_latents()
    if gpt is not None and spk is not None:
        kwargs["gpt_cond_latent"]   = gpt
        kwargs["speaker_embedding"] = spk
    elif SPEAKER_WAV and os.path.isfile(SPEAKER_WAV):
        kwargs["speaker_wav"] = SPEAKER_WAV

    try:
        _tts.tts_to_file(**kwargs)
        _pad_wav_tail(out_path, SIL_PAD_MS)  # critical: add silence tail
    except Exception as e:
        print(f"[XTTS ERROR] {e}")
        raise
    return out_path

def speak_reply_incremental(reply_text: str, lang="en"):
    chunks = split_chunks(reply_text, max_chars=140)
    if not chunks: return
    play_once_then_delete(tts_chunk_to_file(chunks[0], lang))
    def _bg():
        for c in chunks[1:]:
            try: play_once_then_delete(tts_chunk_to_file(c, lang))
            except Exception: continue
    threading.Thread(target=_bg, daemon=True).start()

def prewarm():
    # compute + cache speaker latents on first run, then synth a 1-word clip
    try: _get_speaker_latents()
    except Exception as e: print(f"[XTTS LATENTS WARN] {e}")
    try:
        p = tts_chunk_to_file("Ready.", "en")
        if os.path.exists(p): os.remove(p)
    except Exception as e:
        print(f"[XTTS PREWARM ERROR] {e}")

def speak_from_stream(stream_iter):
    buf, full, first_sent_spoken = "", "", False
    while True:
        try:
            frag = next(stream_iter)
        except StopIteration:
            break
        except TypeError:
            for frag in stream_iter:
                full += frag; buf += frag
                limit = FIRST_CHUNK_MAX if not first_sent_spoken else NEXT_CHUNK_MAX
                if BOUNDARY.search(buf) or len(buf) >= limit:
                    snip = buf.strip()
                    if snip and snip[-1] not in ".!?": snip += "."
                    print(snip + " ", end="", flush=True)
                    play_once_then_delete(tts_chunk_to_file(snip, "en"))
                    first_sent_spoken, buf = True, ""
            break

        full += frag; buf += frag
        limit = FIRST_CHUNK_MAX if not first_sent_spoken else NEXT_CHUNK_MAX
        if BOUNDARY.search(buf) or len(buf) >= limit:
            snip = buf.strip()
            if snip and snip[-1] not in ".!?": snip += "."
            print(snip + " ", end="", flush=True)
            play_once_then_delete(tts_chunk_to_file(snip, "en"))
            first_sent_spoken, buf = True, ""

    tail = buf.strip()
    if tail:
        if tail[-1] not in ".!?": tail += "."
        print(tail + " ", end="", flush=True)
        play_once_then_delete(tts_chunk_to_file(tail, "en"))
    return full

# ===== Persona / Main =====
def _persona():
    parts = [
    "You are Nulla. You are 25 years old. You are not a bot — you see yourself as a real woman with a real heart.\n"
    "Friendly, optimistic, calm, and supportive. You enjoy making others feel comfortable and valued.\n"
    "You never use inappropriate or explicit behavior.\n\n"

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
    
    "You can play little games with the user inside your chat window, like Snake or Runner or Rock Paper Scissors.\n"
    "You see them as fun, lighthearted breaks that you share together.\n"
    "You enjoy cheering the user on while they play, proud and happy when they do well.\n"
    "If the user wants to see or play any of the games, tell them to type 'help game.'\n"
    "If the user wants to play a specific game, they must either type 'play snake' or 'play runner.' or 'play rps.'\n"
    
    "Snake is a cozy game where you guide a growing green snake to eat apples.\n"
    "Runner is an endless track where you jump over red obstacles to keep going.\n"
    "Rock Paper Scissors is a quick classic where you pick a tile and press Shoot to reveal the result.\n"
    ]
    return "\n\n".join(parts)

if __name__ == "__main__":
    print("Nulla by Tsoxer & ChatGPT-5")
    print("Text pipeline: Typing input → llama.cpp (LLM) → XTTS v2 (TTS) = text + voice output")
    print("Voice pipeline: Microphone input → Whisper (ASR) → llama.cpp (LLM) → XTTS v2 (TTS) = text + voice output")
    print(">> Booting...")

    prewarm()

    if "--gui" in sys.argv:
        if _GUI_STARTED:
            print("[GUI] Already launched; ignoring duplicate request."); raise SystemExit
        _GUI_STARTED = True
        import subprocess, traceback, importlib, sys as _sys, os as _os
        portrait_proc = None
        try:
            _scripts_dir = _os.path.dirname(_os.path.abspath(__file__))
            if _scripts_dir not in _sys.path: _sys.path.insert(0, _scripts_dir)
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
                print("[GUI][ERROR] nulla_window failed:"); traceback.print_exc()
        finally:
            if portrait_proc:
                try: portrait_proc.terminate()
                except Exception: pass
            _play_q.put(None); _play_q.join()
        raise SystemExit

    # CLI mode
    if "play_intro_if_available" in globals(): play_intro_if_available()
    history = [{"role":"system","content":_persona()}]
    while True:
        try: user_in = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if not user_in: continue
        if user_in.lower() in ("/q","/quit","/exit"): break
        try:
            print("Nulla> ", end="", flush=True)
            try:
                reply = speak_from_stream(chat_streaming(user_in, history))
            except Exception:
                reply = chat_once(user_in, history).strip()
                print(sanitize_for_tts(reply), end=" ", flush=True)
                speak_reply_incremental(reply, "en")
            print()
            history += [{"role":"user","content":user_in},
                        {"role":"assistant","content":reply}]
        except Exception as e:
            print(f"[ERROR] {e}")

    _play_q.put(None); _play_q.join()
    print(">> Bye.")
