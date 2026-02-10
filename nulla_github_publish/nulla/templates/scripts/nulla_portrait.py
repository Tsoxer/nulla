#nulla_portrait.py

import os, sys, threading, time, json
import tkinter as tk

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

# ===== Config =====
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR    = os.path.abspath(os.path.join(_SCRIPTS_DIR, ".."))
_ASSETS_DIR  = os.path.join(_BASE_DIR, "assets")

IMAGE_PATH        = os.path.join(_ASSETS_DIR, "Nulla.png")
POKE_IMAGE_PATH   = os.path.join(_ASSETS_DIR, "do_not_poke_nulla.png")
POKE_WAV_PATH     = os.path.join(_ASSETS_DIR, "poke.wav")

TITLE         = "Nulla • Portrait"
SCALE         = 0.35
POKE_FLASH_MS = 500  # 0.5 seconds

# ===== Docking (title-independent) =====
DOCK_ENABLED = True

# If you set this env var, we’ll dock to this exact title (optional).
DOCK_TARGET_TITLE = os.environ.get("NULLA_CHAT_TITLE", "").strip()

# Otherwise we auto-detect the chat window by stable prefix (version-safe).
# Example chat titles: "Nulla - Alpha v0.0.8", "Nulla - Alpha v0.0.9"
DOCK_TITLE_PREFIX = os.environ.get("NULLA_CHAT_PREFIX", "Nulla - Alpha").strip()

DOCK_GAP_PX               = 0     # 0 = physically attached
DOCK_POLL_MS              = 33    # ~30 fps
DOCK_ALLOW_RIGHT_FALLBACK = True
DOCK_Y_OFFSET             = -20   # negative = higher, positive = lower

# ===== Emotion state sharing toggle =====
EMOTION_STATE_ENABLED = os.environ.get("NULLA_EMOTION_USE_STATE_FILE", "1").strip() != "0"
EMOTION_POLL_MS = int(os.environ.get("NULLA_EMOTION_POLL_MS", "250").strip() or 250)

# ===== Emotion system =====
# Map your existing poke tuning env vars to the module's names (so you keep your settings)
os.environ.setdefault("NULLA_POKE_WINDOW_SEC", os.environ.get("NULLA_POKE_WINDOW_S", "2.0"))
os.environ.setdefault("NULLA_POKE_ANGRY_THRESHOLD", os.environ.get("NULLA_POKE_THRESHOLD", "3"))
os.environ.setdefault("NULLA_POKE_PENDING_TTL_SEC", os.environ.get("NULLA_POKE_TTL_S", "20.0"))

from nulla_emotion import (
    EmotionManager,
    get_emotion_image_path,
    get_emotion_status_text,
    get_emotion_status_color,
    register_poke,  # <-- NEW: use module poke API
)

from nulla_emotion import STATE_PATH
print("[PORTRAIT] STATE_PATH =", os.path.abspath(STATE_PATH))

# =============================================================================
# NEW: state reader helper (was referenced but missing)
# =============================================================================
def _read_state_json() -> dict:
    """
    Best-effort read of nulla_emotion_state.json.
    Returns {} on missing/empty/invalid JSON.
    Never throws.
    """
    try:
        if not STATE_PATH or not os.path.isfile(STATE_PATH):
            return {}
        try:
            if os.path.getsize(STATE_PATH) == 0:
                return {}
        except Exception:
            pass
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def _play_poke_sound():
    try:
        if sys.platform.startswith("win"):
            import winsound
            if os.path.exists(POKE_WAV_PATH):
                winsound.PlaySound(POKE_WAV_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception:
        pass

def _watch_parent(ppid: int):
    """Optional PPID watchdog: exit if main python dies (when you close CMD)."""
    try:
        import ctypes, ctypes.wintypes as wt
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, ppid)
        if not h:
            return
        while True:
            code = wt.DWORD()
            if k32.GetExitCodeProcess(h, ctypes.byref(code)) == 0:
                break
            if code.value != STILL_ACTIVE:
                os._exit(0)
            time.sleep(1.0)
    except Exception:
        pass  # best-effort only

def _get_target_window_rect():
    """
    Returns (left, top, right, bottom, is_iconic) or None.
    Priority:
      1) exact title if DOCK_TARGET_TITLE is set
      2) best match by prefix DOCK_TITLE_PREFIX (largest area)
    """
    if not sys.platform.startswith("win"):
        return None

    try:
        import ctypes as ct
        from ctypes import wintypes as wt

        user32 = ct.windll.user32

        user32.FindWindowW.argtypes = [wt.LPCWSTR, wt.LPCWSTR]
        user32.FindWindowW.restype  = wt.HWND

        user32.GetWindowRect.argtypes = [wt.HWND, ct.POINTER(wt.RECT)]
        user32.GetWindowRect.restype  = wt.BOOL

        user32.IsIconic.argtypes = [wt.HWND]
        user32.IsIconic.restype  = wt.BOOL

        user32.IsWindowVisible.argtypes = [wt.HWND]
        user32.IsWindowVisible.restype  = wt.BOOL

        user32.GetWindowTextLengthW.argtypes = [wt.HWND]
        user32.GetWindowTextLengthW.restype  = ct.c_int

        user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ct.c_int]
        user32.GetWindowTextW.restype  = ct.c_int

        # 1) Exact title
        if DOCK_TARGET_TITLE:
            hwnd = user32.FindWindowW(None, DOCK_TARGET_TITLE)
            if hwnd:
                rect = wt.RECT()
                if user32.GetWindowRect(hwnd, ct.byref(rect)):
                    return (rect.left, rect.top, rect.right, rect.bottom, bool(user32.IsIconic(hwnd)))

        # 2) Prefix search
        prefix = DOCK_TITLE_PREFIX
        if not prefix:
            return None

        best_area = -1
        best_rect = None

        EnumProc = ct.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

        def _cb(hwnd, lparam):
            nonlocal best_area, best_rect
            try:
                if not user32.IsWindowVisible(hwnd):
                    return True

                n = user32.GetWindowTextLengthW(hwnd)
                if n <= 0:
                    return True

                buf = ct.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                title = (buf.value or "").strip()
                if not title:
                    return True

                # Don't match ourselves
                if title.startswith(TITLE):
                    return True

                if not title.startswith(prefix):
                    return True

                rect = wt.RECT()
                if not user32.GetWindowRect(hwnd, ct.byref(rect)):
                    return True

                w = rect.right - rect.left
                h = rect.bottom - rect.top
                area = w * h

                if area > best_area:
                    best_area = area
                    best_rect = (rect.left, rect.top, rect.right, rect.bottom, bool(user32.IsIconic(hwnd)))
            except Exception:
                pass
            return True

        user32.EnumWindows(EnumProc(_cb), 0)
        return best_rect

    except Exception:
        return None


class PortraitApp:
    def __init__(self, ppid: int | None = None):
        self.root = tk.Tk()
        self.root.title(TITLE)  # STATIC title
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)

        # Dock state
        self._dock_hidden = False
        self._last_xy = None

        # Emotion state
        self.em = EmotionManager() if EMOTION_STATE_ENABLED else None
        self.current_effective_emotion = self.em.get_effective() if self.em else "Neutral"

        # Flash state
        self._flash_lock = threading.Lock()
        self._is_flashing = False
        self._flash_after_id = None

        # Track current image size so poke resize is identical
        self._img_w = None
        self._img_h = None

        # ===== UI =====
        self.container = tk.Frame(self.root, bg="#ffffff")
        self.container.pack(fill="both", expand=True)

        self.img_label = tk.Label(self.container, bd=0, bg="#ffffff", highlightthickness=0)
        self.img_label.pack(padx=0, pady=0)

        # Status overlay (TOP-RIGHT inside portrait window)
        self.status_var = tk.StringVar(value=get_emotion_status_text(self.current_effective_emotion))
        self.status_label = tk.Label(
            self.container,
            textvariable=self.status_var,
            bg="#ffffff",
            fg=get_emotion_status_color(self.current_effective_emotion),
            font=("Segoe UI", 10),
        )
        self.status_label.place(relx=1.0, x=-10, y=8, anchor="ne")

        # Image caches
        self._poke_photo_cache = {}      # (w,h) -> PhotoImage
        self._emotion_photo_cache = {}   # emotion -> PhotoImage

        # Load initial emotion image
        self._set_emotion_visual(self.current_effective_emotion, force=True)

        # Click anywhere inside portrait window -> poke
        for w in (self.root, self.container, self.img_label, self.status_label):
            w.bind("<Button-1>", self._on_poke_click)

        # Block close/minimize (NO SFX on those)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_attempt)
        self.root.bind("<Unmap>", self._on_unmap)

        # PPID watchdog
        if ppid:
            threading.Thread(target=_watch_parent, args=(ppid,), daemon=True).start()

        # Start docking + emotion polling
        self.root.after(50, self._dock_loop)
        self.root.after(150, self._poll_emotion)

        print("[PORTRAIT] started.")

    def _load_scaled_photo(self, path: str, target_w: int | None = None, target_h: int | None = None):
        """
        Returns a Tk PhotoImage.
        Prefers Pillow if available.
        If target_w/target_h provided, resizes EXACTLY to those pixels.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        # Pillow path (best)
        if Image is not None and ImageTk is not None:
            img = Image.open(path).convert("RGBA")
            w, h = img.size

            if target_w is None or target_h is None:
                nw = max(1, int(w * SCALE))
                nh = max(1, int(h * SCALE))
            else:
                nw = max(1, int(target_w))
                nh = max(1, int(target_h))

            if (nw, nh) != img.size:
                img = img.resize((nw, nh), Image.LANCZOS)
            return ImageTk.PhotoImage(img)

        # Tk fallback (no scaling)
        from tkinter import PhotoImage
        return PhotoImage(file=path)

    def _get_poke_photo(self):
        """Get poke image resized to the current image dimensions (so window doesn't jump)."""
        if self._img_w is None or self._img_h is None:
            return None

        key = (self._img_w, self._img_h)
        if key in self._poke_photo_cache:
            return self._poke_photo_cache[key]

        try:
            photo = self._load_scaled_photo(POKE_IMAGE_PATH, target_w=self._img_w, target_h=self._img_h)
            self._poke_photo_cache[key] = photo
            return photo
        except Exception:
            return None

    def _flash_poke(self, play_sound: bool = True):
        # Prevent overlapping flashes
        with self._flash_lock:
            if self._is_flashing:
                return
            self._is_flashing = True

        def _do_flash():
            try:
                poke = self._get_poke_photo()
                if poke is not None:
                    self.img_label.configure(image=poke)

                # IMPORTANT: sound ONLY on real poke clicks
                if play_sound:
                    _play_poke_sound()

                # cancel old revert if any
                try:
                    if self._flash_after_id is not None:
                        self.root.after_cancel(self._flash_after_id)
                except Exception:
                    pass

                self._flash_after_id = self.root.after(POKE_FLASH_MS, self._restore_emotion_image)
            except Exception:
                self._restore_emotion_image()

        self.root.after(0, _do_flash)

    def _restore_emotion_image(self):
        self._flash_after_id = None
        try:
            # Refresh effective emotion after a poke (in case it triggered Anger)
            if EMOTION_STATE_ENABLED:
                try:
                    self.em = EmotionManager()
                    self.current_effective_emotion = self.em.get_effective()
                except Exception:
                    pass
            self._set_emotion_visual(self.current_effective_emotion, force=True)
        finally:
            with self._flash_lock:
                self._is_flashing = False

    def _on_poke_click(self, event=None):
        # Real poke click: flash + sound
        self._flash_poke(play_sound=True)

        # NEW: use the module’s poke tracker (correct schema + proper “next msg” behavior)
        try:
            register_poke()
        except Exception:
            pass

    def _on_close_attempt(self):
        # Flash + refuse to close (NO SOUND)
        self._flash_poke(play_sound=False)
        try:
            self.root.after(POKE_FLASH_MS + 10, self.root.deiconify)
        except Exception:
            pass

    def _on_unmap(self, event=None):
        # If we hid ourselves because the chat is minimized, do NOT bounce/flash.
        if self._dock_hidden:
            return

        # Minimize attempt → flash + pop back (NO SOUND)
        try:
            if self.root.state() == "iconic":
                self._flash_poke(play_sound=False)
                self.root.after(POKE_FLASH_MS + 10, self.root.deiconify)
        except Exception:
            pass

    def _set_emotion_visual(self, emotion: str, force: bool = False):
        effective = emotion

        if force or effective not in self._emotion_photo_cache:
            img_path = get_emotion_image_path(effective)
            photo = None
            try:
                photo = self._load_scaled_photo(img_path)
            except Exception:
                try:
                    photo = self._load_scaled_photo(IMAGE_PATH)
                except Exception:
                    photo = None

            if photo is not None:
                self._emotion_photo_cache[effective] = photo

        photo = self._emotion_photo_cache.get(effective)
        if photo is not None:
            self.img_label.configure(image=photo)

            # Track pixel size so poke can be resized EXACTLY the same
            try:
                self._img_w = int(photo.width())
                self._img_h = int(photo.height())
            except Exception:
                self._img_w = None
                self._img_h = None

        # Update overlay status text (NOT window title)
        self.status_var.set(get_emotion_status_text(effective))

        # Update overlay status color
        try:
            self.status_label.configure(fg=get_emotion_status_color(effective))
        except Exception:
            pass

        # Keep title static, always.
        try:
            if self.root.title() != TITLE:
                self.root.title(TITLE)
        except Exception:
            pass

    def _poll_emotion(self):
        # If poke_pending is active (and not expired), show Anger immediately.
        poke_force_anger = False
        try:
            st = _read_state_json()
            if st.get("poke_pending"):
                until = float(st.get("poke_pending_until") or 0.0)
                if until and time.time() <= until:
                    poke_force_anger = True
        except Exception:
            poke_force_anger = False

        if poke_force_anger:
            effective = "Anger"
        else:
            if not EMOTION_STATE_ENABLED:
                effective = "Neutral"
            else:
                try:
                    self.em = EmotionManager()
                    effective = self.em.get_effective()
                except Exception:
                    effective = self.current_effective_emotion

        if effective != self.current_effective_emotion:
            self.current_effective_emotion = effective
            with self._flash_lock:
                flashing = self._is_flashing
            if not flashing:
                self._set_emotion_visual(effective, force=False)

        self.root.after(EMOTION_POLL_MS, self._poll_emotion)

    def _dock_loop(self):
        try:
            if DOCK_ENABLED:
                rect = _get_target_window_rect()
                if rect is not None:
                    cl, ct, cr, cb, iconic = rect
                    ch = cb - ct

                    # If chat is minimized, hide portrait too
                    if iconic:
                        if not self._dock_hidden:
                            self._dock_hidden = True
                            try:
                                self.root.withdraw()
                            except Exception:
                                pass
                        self.root.after(DOCK_POLL_MS, self._dock_loop)
                        return
                    else:
                        if self._dock_hidden:
                            self._dock_hidden = False
                            try:
                                self.root.deiconify()
                            except Exception:
                                pass

                    # Update our own size
                    self.root.update_idletasks()
                    win_w = self.root.winfo_width()
                    win_h = self.root.winfo_height()

                    sw = self.root.winfo_screenwidth()
                    sh = self.root.winfo_screenheight()

                    # Primary: dock LEFT of chat (edge-to-edge)
                    x = cl - win_w - DOCK_GAP_PX
                    y = ct + (ch - win_h) // 2 + DOCK_Y_OFFSET

                    # If no room on left, dock RIGHT instead
                    if DOCK_ALLOW_RIGHT_FALLBACK and x < 0:
                        x = cr + DOCK_GAP_PX

                    # Clamp to screen
                    x = max(0, min(sw - win_w, x))
                    y = max(0, min(sh - win_h, y))

                    if self._last_xy != (x, y):
                        self._last_xy = (x, y)
                        self.root.geometry(f"+{x}+{y}")

        except Exception:
            pass

        self.root.after(DOCK_POLL_MS, self._dock_loop)

    def run(self):
        self.root.mainloop()


def main():
    ppid = None
    for arg in sys.argv[1:]:
        if arg.startswith("--ppid="):
            try:
                ppid = int(arg.split("=", 1)[1])
            except Exception:
                pass

    app = PortraitApp(ppid=ppid)
    app.run()

if __name__ == "__main__":
    main()
