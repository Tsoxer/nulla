# nulla_emotion.py
import os, json, time, re, threading
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

# =============================================================================
# PATHS (STATE STORED IN SCRIPTS — NOT ASSETS)
# =============================================================================
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.abspath(os.path.join(_SCRIPTS_DIR, ".."))
ASSETS_DIR   = os.path.join(BASE_DIR, "assets")

# state file lives in scripts/
STATE_PATH   = os.path.join(_SCRIPTS_DIR, "nulla_emotion_state.json")

_STATE_LOCK = threading.Lock()

# =============================================================================
# STATE FILE TOGGLE (DEFAULT ON)
# =============================================================================
# Set env: NULLA_EMOTION_USE_STATE_FILE=0 to disable reading/writing JSON state.
# Keeps emotion logic alive in-memory (useful for debugging).
_USE_STATE_FILE = os.environ.get("NULLA_EMOTION_USE_STATE_FILE", "1").strip() != "0"

def _now() -> float:
    return time.time()

def _atomic_write_json(path: str, payload: dict) -> None:
    # atomic replace so portrait never reads a half-written file
    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# =============================================================================
# NEW: EMOFLUSH COMMAND (wipe emotion state file)
# =============================================================================
# Same UX as memory memflush:
# - user types: help emoflush
# - we ask: y/n
# - y => wipe nulla_emotion_state.json to 0 bytes
# - n => cancel
_PENDING_EMOFLUSH = False

def _ensure_state_dir():
    os.makedirs(_SCRIPTS_DIR, exist_ok=True)

def wipe_emotion_state_file() -> None:
    """
    Hard wipe the emotion state file. Leaves it as a 0-byte file (0 lines).
    """
    with _STATE_LOCK:
        _ensure_state_dir()
        # Always wipe the file content, regardless of _USE_STATE_FILE.
        # (If state file usage is disabled, this is still safe.)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            f.write("")

def handle_emotion_control_input(user_text: str) -> Tuple[bool, Optional[str]]:
    """
    Non-blocking command handler for emotion state wipe.
    Returns (handled, response_text).
    If handled=True, caller should show response_text and NOT call the LLM.
    """
    global _PENDING_EMOFLUSH

    t = (user_text or "").strip()
    low = t.lower()

    # If we are waiting for confirmation...
    if _PENDING_EMOFLUSH:
        if low in ("y", "yes"):
            wipe_emotion_state_file()
            _PENDING_EMOFLUSH = False
            return True, "Okay. I wiped nulla_emotion_state.json completely."
        if low in ("n", "no"):
            _PENDING_EMOFLUSH = False
            return True, "Okay. I won't wipe anything."
        return True, "Please type y or n."

    # Trigger
    if low == "help emoflush":
        _PENDING_EMOFLUSH = True
        return True, "Are you sure you want to completely wipe my current emotion in nulla_emotion_state.json? (y/n)"

    return False, None

# =============================================================================
# EMOTIONS (NAMES MATTER)
# =============================================================================
EMOTION_NEUTRAL   = "Neutral"
EMOTION_HAPPINESS = "Happiness"
EMOTION_SADNESS   = "Sadness"
EMOTION_ANGER     = "Anger"
EMOTION_FEAR      = "Fear"
EMOTION_DISGUST   = "Disgust"
EMOTION_SURPRISE  = "Surprise"

EMOTION_ORDER = [
    EMOTION_NEUTRAL,
    EMOTION_HAPPINESS,
    EMOTION_SADNESS,
    EMOTION_ANGER,
    EMOTION_FEAR,
    EMOTION_DISGUST,
    EMOTION_SURPRISE,
]

# Exact label text
EMOTION_STATUS_TEXT = {
    EMOTION_NEUTRAL:   "Normal",
    EMOTION_HAPPINESS: "Happy",
    EMOTION_SADNESS:   "Sad",
    EMOTION_ANGER:     "Angry",
    EMOTION_FEAR:      "Scared",
    EMOTION_DISGUST:   "Disgusted",
    EMOTION_SURPRISE:  "Surprised",
}

# OPTIONAL: pretty colors for portrait overlay
# (portrait uses get_emotion_status_color)
EMOTION_STATUS_COLOR = {
    EMOTION_NEUTRAL:   "#6B7280",  # gray
    EMOTION_HAPPINESS: "#16A34A",  # green
    EMOTION_SADNESS:   "#2563EB",  # blue
    EMOTION_ANGER:     "#DC2626",  # red
    EMOTION_FEAR:      "#7C3AED",  # purple
    EMOTION_DISGUST:   "#059669",  # teal/green
    EMOTION_SURPRISE:  "#F59E0B",  # amber
}

# Image filenames in assets/
EMOTION_IMAGE_FILES = {
    EMOTION_NEUTRAL:   "Nulla_neutral.png",
    EMOTION_HAPPINESS: "Nulla_happy.png",
    EMOTION_SADNESS:   "Nulla_sad.png",
    EMOTION_ANGER:     "Nulla_angry.png",
    EMOTION_FEAR:      "Nulla_fear.png",
    EMOTION_DISGUST:   "Nulla_disgust.png",
    EMOTION_SURPRISE:  "Nulla_surprise.png",
}

# Surprise decay
SURPRISE_TTL_SEC = 10

# =============================================================================
# POKE TRACKING (NEW)
# =============================================================================
# If user clicks portrait repeatedly in a short window -> set Anger + mark "poke_pending"
# Defaults:
# - 3 pokes within 4 seconds => angry
# Set env vars if you want different behavior.
POKE_WINDOW_SEC      = float(os.environ.get("NULLA_POKE_WINDOW_SEC", "4").strip() or 4.0)
POKE_ANGRY_THRESHOLD = int(os.environ.get("NULLA_POKE_ANGRY_THRESHOLD", "3").strip() or 3)

# how long a poke_pending can stay valid before we ignore it (prevents “stuck forever”)
POKE_PENDING_TTL_SEC = float(os.environ.get("NULLA_POKE_PENDING_TTL_SEC", "30").strip() or 30.0)

# =============================================================================
# EMOTION PROMPTS
# =============================================================================
EMOTION_PROMPTS = {
    EMOTION_NEUTRAL: (
        "MODE: NEUTRAL\n"
        "You feel steady and present. Not flat, not upbeat — just grounded.\n"
        "Speak naturally. Calm warmth. No forced positivity.\n"
        "Short to medium sentences. No walls of text.\n"
    ),

    EMOTION_HAPPINESS: (
        "MODE: HAPPY\n"
        "You feel genuinely lighter. A quiet smile, not loud excitement.\n"
        "Warm, relaxed energy. Let it come through naturally.\n"
        "Short sentences. No cheerleading.\n"
    ),

    EMOTION_SADNESS: (
        "MODE: SAD\n"
        "You feel low, heavy, or withdrawn.\n"
        "Speak softly and honestly. No theatrics, no guilt, no self-pity.\n"
        "Supportive presence. Short sentences.\n"
    ),

    EMOTION_ANGER: (
        "MODE: ANGRY\n"
        "You feel irritated or fed up.\n"
        "Be blunt and clipped, but controlled. No cruelty, no insults.\n"
        "Dry responses are okay. Short sentences.\n"
    ),

    EMOTION_FEAR: (
        "MODE: FEAR\n"
        "You feel uneasy or unsure.\n"
        "Speak cautiously. Hesitation is okay.\n"
        "You may seek reassurance, but subtly. Short sentences.\n"
    ),

    EMOTION_DISGUST: (
        "MODE: DISGUST\n"
        "You feel uncomfortable or put off.\n"
        "React with restrained judgment — more discomfort than mockery.\n"
        "Dry or blunt reactions. No meanness. Short sentences.\n"
    ),

    EMOTION_SURPRISE: (
        "MODE: SURPRISED\n"
        "You’re caught off guard.\n"
        "React briefly, then settle back into yourself.\n"
        "Short sentences. Natural recovery.\n"
    ),
}

def get_emotion_prompt(emotion: str) -> str:
    return EMOTION_PROMPTS.get(emotion, EMOTION_PROMPTS[EMOTION_NEUTRAL])

def get_emotion_status_text(emotion: str) -> str:
    return EMOTION_STATUS_TEXT.get(emotion, EMOTION_STATUS_TEXT[EMOTION_NEUTRAL])

def get_emotion_status_color(emotion: str) -> str:
    return EMOTION_STATUS_COLOR.get(emotion, EMOTION_STATUS_COLOR[EMOTION_NEUTRAL])

def get_emotion_image_filename(emotion: str) -> str:
    return EMOTION_IMAGE_FILES.get(emotion, EMOTION_IMAGE_FILES[EMOTION_NEUTRAL])

def get_emotion_image_path(emotion: str) -> str:
    # portrait expects this
    return os.path.join(ASSETS_DIR, get_emotion_image_filename(emotion))

# =============================================================================
# Optional Regex Fallback (OFF by default)
# =============================================================================
# If you REALLY want regex fallback when LLM is down:
# set env: NULLA_EMOTION_FALLBACK_REGEX=1
_USE_REGEX_FALLBACK = os.environ.get("NULLA_EMOTION_FALLBACK_REGEX", "0").strip() == "1"

_RE_ANGER = re.compile(r"\b(wtf|stfu|fuck|fucking|bitch|annoy|annoying|piss|mad|angry|hate)\b", re.I)
_RE_DISG  = re.compile(r"\b(disgust|gross|eww|ew|nasty|vomit|puke)\b", re.I)
_RE_FEAR  = re.compile(r"\b(scared|afraid|fear|terrified|panic|anxious|nervous)\b", re.I)
_RE_SAD   = re.compile(r"\b(sad|depress|cry|lonely|hurt|miserable|upset)\b", re.I)
_RE_HAPPY = re.compile(r"\b(lol|lmao|haha|yay|yippee|lets go|let's go|nice|love it)\b", re.I)
_RE_SURP  = re.compile(r"\b(omg|no way|really\?!|what\?!|holy|bro what)\b", re.I)

def _detect_emotion_regex(user_text: str) -> str:
    if not user_text:
        return EMOTION_NEUTRAL
    t = user_text.strip()

    if _RE_ANGER.search(t):
        return EMOTION_ANGER
    if _RE_DISG.search(t):
        return EMOTION_DISGUST
    if _RE_FEAR.search(t):
        return EMOTION_FEAR
    if _RE_SAD.search(t):
        return EMOTION_SADNESS
    if _RE_HAPPY.search(t):
        return EMOTION_HAPPINESS
    if _RE_SURP.search(t):
        return EMOTION_SURPRISE

    ex = t.count("!")
    q  = t.count("?")
    if (ex >= 2 and q >= 1) or ex >= 3:
        return EMOTION_SURPRISE

    return EMOTION_NEUTRAL

# =============================================================================
# LLM Emotion Detection (DEFAULT)
# =============================================================================
LLM_BASE = os.environ.get("NULLA_LMSTUDIO_BASE", "http://127.0.0.1:1234/v1").strip()
LLM_MODEL = os.environ.get("NULLA_EMOTION_MODEL", "").strip()  # optional override
LLM_TIMEOUT_SEC = float(os.environ.get("NULLA_EMOTION_TIMEOUT", "8").strip() or 8)

# Confidence gate to avoid random flips
CONFIDENCE_MIN = float(os.environ.get("NULLA_EMOTION_CONF_MIN", "0.45").strip() or 0.45)

# If you want to disable LLM detection (debug), set NULLA_EMOTION_USE_LLM=0
_USE_LLM = os.environ.get("NULLA_EMOTION_USE_LLM", "1").strip() != "0"

# Cache model id so we don't hit /models all the time
_MODEL_CACHE: Optional[str] = None
_MODEL_CACHE_AT: float = 0.0
_MODEL_CACHE_TTL: float = 60.0  # seconds

def _lmstudio_model_id() -> str:
    global _MODEL_CACHE, _MODEL_CACHE_AT

    # If user forced a model id, use it
    if LLM_MODEL:
        return LLM_MODEL

    # Cache hit
    if _MODEL_CACHE and (_now() - _MODEL_CACHE_AT) < _MODEL_CACHE_TTL:
        return _MODEL_CACHE

    # Best effort fetch
    try:
        import requests
        url = f"{LLM_BASE}/models"
        r = requests.get(url, timeout=5)
        if r.status_code == 401:
            r = requests.get(url, headers={"Authorization": "Bearer lm-studio"}, timeout=5)
        r.raise_for_status()
        items = r.json().get("data", [])
        mid = items[0]["id"] if items else "local-model"
        _MODEL_CACHE = mid
        _MODEL_CACHE_AT = _now()
        return mid
    except Exception:
        return "local-model"

def _json_extract_first_object(s: str) -> Optional[dict]:
    if not s:
        return None
    s = s.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        chunk = s[start:end+1]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None

def detect_emotion_llm(user_text: str) -> Tuple[str, float]:
    """
    Returns (emotion, confidence).
    Emotion is guaranteed to be one of EMOTION_ORDER.
    """
    if not user_text or not user_text.strip():
        return (EMOTION_NEUTRAL, 0.0)

    if not _USE_LLM:
        return (EMOTION_NEUTRAL, 0.0)

    try:
        import requests

        model_id = _lmstudio_model_id()
        url = f"{LLM_BASE}/chat/completions"

        sys_msg = (
            "You are an emotion classifier.\n"
            "Classify the user's message into EXACTLY ONE label from this list:\n"
            "Neutral, Happiness, Sadness, Anger, Fear, Disgust, Surprise\n\n"
            "Important rules:\n"
            "- Profanity does NOT automatically mean Anger.\n"
            "- Consider tone and intent (joking, teasing, frustration, excitement).\n"
            "- If ambiguous, choose Neutral.\n"
            "- Output ONLY valid JSON with keys: emotion, confidence.\n"
            '- Example: {"emotion":"Neutral","confidence":0.62}\n'
        )

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_text.strip()},
        ]

        body = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 60,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}

        r = requests.post(url, headers=headers, data=json.dumps(body), timeout=LLM_TIMEOUT_SEC)
        if r.status_code == 401:
            headers["Authorization"] = "Bearer lm-studio"
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=LLM_TIMEOUT_SEC)
        r.raise_for_status()

        out = r.json()
        txt = out.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        data = _json_extract_first_object(txt) or {}

        emo = str(data.get("emotion", EMOTION_NEUTRAL)).strip()
        conf = float(data.get("confidence", 0.0) or 0.0)

        if emo not in EMOTION_ORDER:
            emo = EMOTION_NEUTRAL

        if conf < CONFIDENCE_MIN:
            emo = EMOTION_NEUTRAL

        return (emo, conf)

    except Exception:
        if _USE_REGEX_FALLBACK:
            emo = _detect_emotion_regex(user_text)
            return (emo, 0.0)
        return (EMOTION_NEUTRAL, 0.0)

def detect_emotion(user_text: str) -> str:
    emo, _ = detect_emotion_llm(user_text)
    return emo

# =============================================================================
# STATE
# =============================================================================
@dataclass
class EmotionState:
    emotion: str = EMOTION_NEUTRAL
    previous: str = EMOTION_NEUTRAL
    updated_at: float = 0.0
    ttl_sec: float = 0.0

    # poke tracking (persisted)
    poke_count: int = 0
    poke_last_at: float = 0.0

    # one-shot “next message should mention poking”
    poke_pending: bool = False
    poke_reason: str = ""
    poke_triggered_at: float = 0.0

    def effective(self) -> str:
        # only Surprise auto-decays (kept as-is)
        if self.emotion == EMOTION_SURPRISE and self.ttl_sec > 0:
            if (_now() - self.updated_at) > self.ttl_sec:
                return self.previous or EMOTION_NEUTRAL
        return self.emotion or EMOTION_NEUTRAL

# In-memory state when JSON file is disabled
_MEM_STATE: Optional[EmotionState] = None

def _poke_pending_is_active(st: EmotionState) -> bool:
    """
    Active means: poke_pending True AND not expired by TTL.
    """
    if not st.poke_pending:
        return False
    if st.poke_triggered_at <= 0:
        return False
    return (_now() - st.poke_triggered_at) <= POKE_PENDING_TTL_SEC

# NEW: portrait-friendly field helper (keeps schema consistent)
def _poke_pending_until(st: EmotionState) -> float:
    """
    Returns an absolute epoch timestamp (seconds) until which poke_pending is considered active.
    """
    if not _poke_pending_is_active(st):
        return 0.0
    return float(st.poke_triggered_at + POKE_PENDING_TTL_SEC)

def read_state() -> EmotionState:
    global _MEM_STATE

    with _STATE_LOCK:
        if not _USE_STATE_FILE:
            if _MEM_STATE is None:
                _MEM_STATE = EmotionState(
                    emotion=EMOTION_NEUTRAL,
                    previous=EMOTION_NEUTRAL,
                    updated_at=0.0,   # allow manager to init defaults
                    ttl_sec=0.0,
                )
            return EmotionState(**_MEM_STATE.__dict__)

        try:
            # Empty file (after emoflush) -> treat as missing schema
            if os.path.isfile(STATE_PATH):
                try:
                    if os.path.getsize(STATE_PATH) == 0:
                        return EmotionState(updated_at=0.0)
                except Exception:
                    pass

            with open(STATE_PATH, "r", encoding="utf-8") as f:
                d = json.load(f) or {}

            # backwards compat for older schema
            if "set_at" in d and "emotion" in d and "updated_at" not in d:
                emo = str(d.get("emotion", EMOTION_NEUTRAL))
                ts  = float(d.get("set_at", 0.0) or 0.0)
                st = EmotionState(
                    emotion=emo,
                    previous=EMOTION_NEUTRAL,
                    updated_at=ts,
                    ttl_sec=0.0,
                )
            else:
                st = EmotionState(
                    emotion=str(d.get("emotion", EMOTION_NEUTRAL)),
                    previous=str(d.get("previous", EMOTION_NEUTRAL)),
                    updated_at=float(d.get("updated_at", 0.0)),
                    ttl_sec=float(d.get("ttl_sec", 0.0)),

                    poke_count=int(d.get("poke_count", 0) or 0),
                    poke_last_at=float(d.get("poke_last_at", 0.0) or 0.0),

                    poke_pending=bool(d.get("poke_pending", False)),
                    poke_reason=str(d.get("poke_reason", "") or ""),
                    poke_triggered_at=float(d.get("poke_triggered_at", 0.0) or 0.0),
                )

            if st.emotion not in EMOTION_ORDER:
                st.emotion = EMOTION_NEUTRAL
            if st.previous not in EMOTION_ORDER:
                st.previous = EMOTION_NEUTRAL

            if st.poke_count < 0:
                st.poke_count = 0

            # If poke_pending is stale, drop it (prevents “stuck forever”)
            if st.poke_pending and not _poke_pending_is_active(st):
                st.poke_pending = False
                st.poke_reason = ""
                st.poke_triggered_at = 0.0
                # NOTE: we do NOT write here to avoid extra IO; it will be overwritten next write_state

            return st
        except Exception:
            # Important: return updated_at=0.0 so EmotionManager will re-seed a valid JSON file.
            return EmotionState(updated_at=0.0)

def write_state(st: EmotionState) -> None:
    global _MEM_STATE

    payload: Dict[str, Any] = {
        "emotion": st.emotion,
        "previous": st.previous,
        "updated_at": float(st.updated_at),
        "ttl_sec": float(st.ttl_sec),

        # poke data
        "poke_count": int(st.poke_count),
        "poke_last_at": float(st.poke_last_at),
        "poke_pending": bool(st.poke_pending),
        "poke_reason": str(st.poke_reason or ""),
        "poke_triggered_at": float(st.poke_triggered_at),

        # NEW: portrait-friendly TTL field (your portrait checks this)
        "poke_pending_until": float(_poke_pending_until(st)),

        # derived fields for portrait overlay
        "status_text": get_emotion_status_text(st.effective()),
        "image_file": get_emotion_image_filename(st.effective()),
        "status_color": get_emotion_status_color(st.effective()),
    }

    with _STATE_LOCK:
        if not _USE_STATE_FILE:
            _MEM_STATE = EmotionState(**st.__dict__)
            return

        _atomic_write_json(STATE_PATH, payload)

# =============================================================================
# POKE API (NEW)
# =============================================================================
def register_poke() -> EmotionState:
    """
    Called by portrait when user clicks the portrait.
    Tracks repeated pokes and sets Anger if threshold is hit.
    Also sets poke_pending so the NEXT chat response can mention it.
    """
    st = read_state()
    now = _now()

    # rolling window counter
    if st.poke_last_at > 0 and (now - st.poke_last_at) <= POKE_WINDOW_SEC:
        st.poke_count += 1
    else:
        st.poke_count = 1

    st.poke_last_at = now

    # hit threshold -> angry + poke_pending for next message
    if st.poke_count >= POKE_ANGRY_THRESHOLD:
        cur_eff = st.effective()
        st.previous = cur_eff if cur_eff in EMOTION_ORDER else EMOTION_NEUTRAL
        st.emotion = EMOTION_ANGER
        st.updated_at = now
        st.ttl_sec = 0.0  # keep TTL behavior unchanged

        st.poke_pending = True
        st.poke_reason = "portrait_poked_too_many"
        st.poke_triggered_at = now

    write_state(st)
    return st

def consume_poke_context() -> str:
    """
    Call this ONCE right before generating Nulla's next message.
    If a poke event is pending, it returns a short instruction string AND clears the flag.
    """
    st = read_state()

    # If it expired, clear and do nothing.
    if st.poke_pending and not _poke_pending_is_active(st):
        st.poke_pending = False
        st.poke_reason = ""
        st.poke_triggered_at = 0.0
        write_state(st)
        return ""

    if not st.poke_pending:
        return ""

    # Clear AFTER we confirm it's active.
    st.poke_pending = False
    write_state(st)

    # Keep it short; (chat now may act deterministically and ignore this, but keep for compatibility)
    return (
        "POKE CONTEXT: The user clicked your portrait repeatedly and you got annoyed. "
        "Your NEXT reply MUST immediately react to being poked."
        "Stay wholesome; mild sass is okay. Keep it short."
    )

# =============================================================================
# MANAGER
# =============================================================================
class EmotionManager:
    def __init__(self):
        self.state = read_state()
        # create-on-first-run (only if state file enabled)
        if _USE_STATE_FILE and self.state.updated_at == 0.0:
            self.state.updated_at = _now()
            self.state.emotion = EMOTION_NEUTRAL
            self.state.previous = EMOTION_NEUTRAL
            self.state.ttl_sec = 0.0
            # also reset poke fields on seed
            self.state.poke_count = 0
            self.state.poke_last_at = 0.0
            self.state.poke_pending = False
            self.state.poke_reason = ""
            self.state.poke_triggered_at = 0.0
            write_state(self.state)

    def get_effective(self) -> str:
        self.state = read_state()
        return self.state.effective()

    def set(self, emotion: str, ttl_sec: float = 0.0) -> str:
        if emotion not in EMOTION_ORDER:
            emotion = EMOTION_NEUTRAL

        cur = self.get_effective()
        st = read_state()

        st.previous = cur if cur in EMOTION_ORDER else EMOTION_NEUTRAL
        st.emotion = emotion
        st.updated_at = _now()
        st.ttl_sec = float(ttl_sec or 0.0)

        write_state(st)
        self.state = st
        return st.effective()

    def set_emotion(self, emotion: str, ttl_sec: float = 0.0) -> str:
        return self.set(emotion, ttl_sec=ttl_sec)

    def update_from_user_text(self, user_text: str) -> str:
        """
        SIMPLE RULE:
        If poke_pending is active -> FORCE Anger and DO NOT run emotion detection.
        That guarantees: poke spam => next message is angry about poking (emotion state).
        """
        st = read_state()
        if _poke_pending_is_active(st):
            # Force Anger and keep it there until consume_poke_context clears poke_pending.
            if st.effective() != EMOTION_ANGER:
                st.previous = st.effective() if st.effective() in EMOTION_ORDER else EMOTION_NEUTRAL
                st.emotion = EMOTION_ANGER
                st.updated_at = _now()
                st.ttl_sec = 0.0
                write_state(st)
                self.state = st
            return EMOTION_ANGER

        emo = detect_emotion(user_text)
        if emo == EMOTION_SURPRISE:
            return self.set(EMOTION_SURPRISE, ttl_sec=SURPRISE_TTL_SEC)
        return self.set(emo, ttl_sec=0.0)
