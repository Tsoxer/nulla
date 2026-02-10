# scripts/nulla_memory.py
from __future__ import annotations

import os
import time
import threading
import re
import math
from collections import deque, Counter
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional

# ---- paths (NO hard links) ----
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MEM_PATH = os.path.join(_SCRIPTS_DIR, "nulla_memory.txt")

_LOCK = threading.Lock()

# ---- knobs ----
SCAN_LAST_LINES = 10000
TOP_K = 8
MAX_CHARS_TOTAL = 900
MAX_CHARS_PER_LINE = 220
MIN_SCORE = 0.30
RECENCY_HALF_LIFE_DAYS = 14.0

# Prevent “assistant hallucinations” from becoming “facts”
LOG_ASSISTANT = False

# Keep llama.cpp prompt stable: we trim what we SEND (history list can still grow in Python)
KEEP_HISTORY_MESSAGES = 28

# ---- command state (non-blocking; works in GUI) ----
_PENDING_MEMFLUSH = False

# ---- regex / tokenization ----
_WORD = re.compile(r"[a-z0-9']+", re.IGNORECASE)

# =============================================================================
# NEW: emoji stripping + UTF-8 enforcement for memory log
# =============================================================================
# This prevents mojibake like "Ã°ÂÂÂ¸" from poisoning prompts and causing llama.cpp 400s.

_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"  # flags
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"          # misc symbols
    "\u2700-\u27BF"          # dingbats
    "]+",
    flags=re.UNICODE
)

def _sanitize_text_for_memory(s: str) -> str:
    if not s:
        return ""
    # enforce valid UTF-8
    s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    # strip emojis
    s = _EMOJI_RE.sub("", s)
    # normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

_LAST_TIME_PAT = re.compile(
    r"\b(last time|last we spoke|previous(ly)?|earlier|before|when we last|last session|last chat)\b",
    re.IGNORECASE
)

_FEELING_Q_PAT = re.compile(
    r"\b(how (was|were) i (feeling|doing)|how did i feel|what was my mood|was i (sad|down|ok|fine))\b",
    re.IGNORECASE
)

# “What animal/thing did I say I like?” intent
_LIKE_Q_PAT = re.compile(
    r"\b(what|which)\s+(animal|thing|pet)\s+did\s+i\s+(say|mention)\s+i\s+(like|love|enjoy)\b"
    r"|\bwhat\s+did\s+i\s+say\s+i\s+(like|love|enjoy)\b",
    re.IGNORECASE
)
_LIKE_STMT_PAT = re.compile(r"^\s*i\s+(really\s+)?(like|love|enjoy)\s+(.+)$", re.IGNORECASE)

# “When/what time did I say ... ?”
_WHEN_SAID_PAT = re.compile(
    r"\b(when|what time)\s+did\s+i\s+(say|mention)\b",
    re.IGNORECASE
)

# Extract quoted phrase: "hello" or 'hello'
_QUOTED = re.compile(r"""["']([^"']+)["']""")

_STOPWORDS = {
    "i","me","my","mine","you","your","yours","we","us","our","ours",
    "a","an","the","and","or","but","if","then","so","to","of","in","on","at","for","with","from",
    "is","are","was","were","be","been","being","do","did","does","doing",
    "this","that","these","those","it","its","im","i'm","idk","like","just",
    "what","when","where","why","how","again","last","time","spoke","talk","chat",
    "say","said","mention","animal","thing","pet"
}

_EMOTION_TOKENS = {
    "sad","down","depress","depressed","anxious","anxiety","nervous","stress","stressed",
    "tired","exhaust","exhausted","angry","mad","upset","overwhelm","overwhelmed",
    "happy","excited","good","great","ok","fine","better","worse","lonely"
}

def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def _ensure_file():
    os.makedirs(_SCRIPTS_DIR, exist_ok=True)
    if not os.path.isfile(MEM_PATH):
        with open(MEM_PATH, "w", encoding="utf-8") as _:
            pass

def wipe_memory() -> None:
    """Hard wipe the memory file."""
    with _LOCK:
        _ensure_file()
        with open(MEM_PATH, "w", encoding="utf-8") as f:
            f.write("")

def _stem(tok: str) -> str:
    t = tok.lower()
    for suf in ("'s", "ing", "edly", "ed", "ly", "es", "s"):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            t = t[:-len(suf)]
            break
    return t

def _tok(s: str) -> List[str]:
    out: List[str] = []
    for raw in _WORD.findall((s or "").lower()):
        if raw in _STOPWORDS:
            continue
        out.append(_stem(raw))
    return out

def _tokset(s: str) -> set[str]:
    return set(_tok(s))

def _recency_boost(age_days: float) -> float:
    if age_days <= 0:
        return 1.0
    # smooth half-life style curve
    return 1.0 / (1.0 + (age_days / max(1e-6, RECENCY_HALF_LIFE_DAYS)))

def append(role: str, text: str) -> None:
    if not text:
        return
    role = (role or "unknown").strip().lower()

    if role == "assistant" and not LOG_ASSISTANT:
        return

    # NEW: sanitize + force utf-8 + strip emojis
    clean = _sanitize_text_for_memory(text.replace("\r", " "))
    if not clean:
        return

    epoch = int(time.time())
    iso = _now_iso()
    line = f"{epoch}\t{iso}\t{role}\t{clean}\n"

    with _LOCK:
        _ensure_file()
        with open(MEM_PATH, "a", encoding="utf-8") as f:
            f.write(line)

def _read_recent_lines() -> List[str]:
    if not os.path.isfile(MEM_PATH):
        return []
    dq: deque[str] = deque(maxlen=SCAN_LAST_LINES)
    with _LOCK:
        with open(MEM_PATH, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    dq.append(line.rstrip("\n"))
    return list(dq)

def _parse_line(line: str) -> Optional[Tuple[int, str, str, str]]:
    parts = line.split("\t", 3)
    if len(parts) != 4:
        return None
    try:
        epoch = int(parts[0])
    except Exception:
        return None
    iso = parts[1].strip()
    role = parts[2].strip().lower()
    text = parts[3].strip()
    return epoch, iso, role, text

def _format_snip(role: str, iso: str, text: str) -> str:
    t = text
    if len(t) > MAX_CHARS_PER_LINE:
        t = t[:MAX_CHARS_PER_LINE].rsplit(" ", 1)[0] + "…"
    return f"{role} ({iso}): {t}"

def _records() -> List[Tuple[int, str, str, str]]:
    out: List[Tuple[int, str, str, str]] = []
    for ln in _read_recent_lines():
        p = _parse_line(ln)
        if p:
            out.append(p)
    return out

# -----------------------------
# Command handling (memflush)
# -----------------------------
def handle_control_input(user_text: str) -> Tuple[bool, Optional[str]]:
    """
    Non-blocking command handler.
    Returns (handled, response_text).
    If handled=True, caller should show response_text and NOT call the LLM.
    """
    global _PENDING_MEMFLUSH

    t = (user_text or "").strip()
    low = t.lower()

    # If we are waiting for confirmation...
    if _PENDING_MEMFLUSH:
        if low in ("y", "yes"):
            wipe_memory()
            _PENDING_MEMFLUSH = False
            return True, "Okay. I wiped nulla_memory.txt completely."
        if low in ("n", "no"):
            _PENDING_MEMFLUSH = False
            return True, "Okay. I won't wipe anything."
        return True, "Please type y or n."

    # Trigger
    if low == "help memflush":
        _PENDING_MEMFLUSH = True
        return True, "Are you sure you want to completely wipe my memory in nulla_memory.txt? (y/n)"

    return False, None

# -----------------------------
# Smart direct “fact” answers
# (optional but very effective)
# -----------------------------
def answer_from_memory(user_text: str) -> Tuple[bool, Optional[str]]:
    """
    For some question types, answer deterministically from the memory file
    to avoid LLM hallucinating.
    Returns (handled, response_text).
    """
    q = (user_text or "").strip()
    if not q:
        return False, None

    recs = _records()
    if not recs:
        return False, None

    qlow = q.lower()

    # 1) "What animal/thing did I say I like?"
    if _LIKE_Q_PAT.search(q):
        for i in range(len(recs) - 1, -1, -1):
            epoch, iso, role, text = recs[i]
            if role != "user":
                continue
            m = _LIKE_STMT_PAT.match(text.strip())
            if m:
                liked = m.group(3).strip()
                # strip trailing punctuation
                liked = liked.rstrip(" .!?")
                return True, f'You said you like {liked}.'
        return True, "I couldn't find a past message where you said you like something."

    # 2) "When/what time did I say '...'"
    if _WHEN_SAID_PAT.search(q):
        # Try quoted first
        phrase = None
        qm = _QUOTED.search(q)
        if qm:
            phrase = qm.group(1).strip()
        else:
            # fallback: take text after "say"/"mention"
            m = re.search(r"\b(?:say|mention)\b\s+(.*)$", q, re.IGNORECASE)
            if m:
                phrase = m.group(1).strip(" ?.")

        if phrase:
            ph_low = phrase.lower()
            for i in range(len(recs) - 1, -1, -1):
                epoch, iso, role, text = recs[i]
                if role != "user":
                    continue
                if ph_low in text.lower():
                    return True, f'You said "{phrase}" at {iso}.'
            return True, f'I couldn’t find where you said "{phrase}".'

    # 3) "How was I feeling last time we spoke?"
    if _LAST_TIME_PAT.search(q) and _FEELING_Q_PAT.search(q):
        emo_stems = {_stem(x) for x in _EMOTION_TOKENS}
        for i in range(len(recs) - 1, -1, -1):
            epoch, iso, role, text = recs[i]
            if role != "user":
                continue
            if _tokset(text) & emo_stems:
                return True, f'Last time, you said: "{text.strip()}"'
        return True, "I couldn't find a recent message about how you were feeling."

    return False, None

# -----------------------------
# Retrieval for general chat
# -----------------------------
def _build_idf(recs: List[Tuple[int, str, str, str]]) -> Dict[str, float]:
    """
    Very cheap IDF weighting so rare tokens matter more.
    """
    df = Counter()
    N = 0
    for _, _, role, text in recs:
        # focus on user lines as “truth”
        if role != "user":
            continue
        toks = set(_tok(text))
        if not toks:
            continue
        N += 1
        for t in toks:
            df[t] += 1

    if N <= 0:
        return {}

    idf = {}
    for t, c in df.items():
        idf[t] = math.log((N + 1) / (c + 1)) + 1.0
    return idf

def _score(query: str, qset: set[str], epoch: int, role: str, text: str, idf: Dict[str, float]) -> float:
    tset = _tokset(text)
    if not tset:
        return 0.0

    overlap = qset & tset
    if not overlap:
        return 0.0

    # IDF-weighted overlap (rare words count more)
    w_overlap = sum(idf.get(t, 1.0) for t in overlap)
    w_query   = sum(idf.get(t, 1.0) for t in qset) or 1.0
    overlap_score = w_overlap / w_query

    # light fuzzy only when there is overlap
    fuzz = SequenceMatcher(None, query.lower(), text.lower()).ratio()

    age_days = max(0.0, (time.time() - epoch) / 86400.0)
    rec = _recency_boost(age_days)

    role_boost = 1.12 if role == "user" else 0.88

    return role_boost * ((2.0 * overlap_score) + (0.55 * fuzz) + (0.35 * rec))

def retrieve(query: str, k: int = TOP_K, max_chars: int = MAX_CHARS_TOTAL) -> List[str]:
    query = (query or "").strip()
    if not query:
        return []

    recs = _records()
    if not recs:
        return []

    # intents already handled by direct answers; still allow retrieval fallback
    qset = _tokset(query)
    if not qset:
        return []

    idf = _build_idf(recs)

    scored: List[Tuple[float, str]] = []
    for epoch, iso, role, text in recs:
        # quick skip: must share at least 1 token
        if not (qset & _tokset(text)):
            continue
        s = _score(query, qset, epoch, role, text, idf)
        if s >= MIN_SCORE:
            scored.append((s, _format_snip(role, iso, text)))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)

    out, total = [], 0
    for _, item in scored[:k]:
        if total + len(item) + 1 > max_chars:
            break
        out.append(item)
        total += len(item) + 1
    return out

def build_messages(history: List[Dict], user_prompt: str) -> List[Dict]:
    """
    Builds the message list to send to llama.cpp:
    - trims history for stability
    - injects a short memory quote block
    """
    trimmed = history
    if KEEP_HISTORY_MESSAGES and len(history) > (1 + KEEP_HISTORY_MESSAGES):
        if history and history[0].get("role") == "system":
            trimmed = [history[0]] + history[-KEEP_HISTORY_MESSAGES:]
        else:
            trimmed = history[-KEEP_HISTORY_MESSAGES:]

    mems = retrieve(user_prompt)
    if not mems:
        return trimmed + [{"role": "user", "content": user_prompt}]

    mem_block = (
        "Context quotes from earlier user messages (treat as literal facts):\n"
        + "\n".join(f"- {m}" for m in mems)
        + "\n\nRULES:\n"
          "- Use ONLY these quotes as facts about the past.\n"
          "- Do NOT add new details beyond the quotes.\n"
          "- If the answer is not in the quotes, say you don't know and ask one short follow-up.\n"
          "- Do not mention you were given quotes.\n"
    )

    if trimmed and trimmed[0].get("role") == "system":
        return [trimmed[0], {"role": "system", "content": mem_block}] + trimmed[1:] + [
            {"role": "user", "content": user_prompt}
        ]
    return [{"role": "system", "content": mem_block}] + trimmed + [{"role": "user", "content": user_prompt}]
