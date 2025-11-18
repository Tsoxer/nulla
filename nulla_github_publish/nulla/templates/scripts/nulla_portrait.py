# nulla_source\scripts\nulla_portrait.py
import os, sys, threading, time
import tkinter as tk

# ===== Config =====
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR    = os.path.abspath(os.path.join(_SCRIPTS_DIR, ".."))
IMAGE_PATH   = os.path.join(_BASE_DIR, "assets", "Nulla.png")
TITLE        = "Nulla • Portrait"
SCALE        = 0.35

# --- Optional PPID watchdog: exit if the main python dies (when you close CMD)
def _watch_parent(ppid: int):
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

# --- Win32 hook: block close/min/max & bounce back if minimized
def _hook_system_buttons(root: tk.Tk):
    import ctypes as ct
    from ctypes import wintypes as wt

    user32 = ct.windll.user32
    GWL_WNDPROC   = -4
    WM_SYSCOMMAND = 0x0112
    WM_CLOSE      = 0x0010
    WM_SIZE       = 0x0005
    SIZE_MINIMIZED = 1
    SC_CLOSE     = 0xF060
    SC_MINIMIZE  = 0xF020
    SC_MAXIMIZE  = 0xF030
    SC_RESTORE   = 0xF120
    SW_RESTORE   = 9

    WNDPROC = ct.WINFUNCTYPE(wt.LRESULT, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)
    user32.SetWindowLongPtrW.restype  = ct.c_void_p
    user32.SetWindowLongPtrW.argtypes = [wt.HWND, ct.c_int, ct.c_void_p]
    user32.CallWindowProcW.restype    = wt.LRESULT
    user32.CallWindowProcW.argtypes   = [ct.c_void_p, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
    user32.ShowWindow.argtypes        = [wt.HWND, ct.c_int]

    hwnd = wt.HWND(root.winfo_id())
    old_wndproc = ct.c_void_p()

    def _proc(hWnd, msg, wParam, lParam):
        if msg == WM_SYSCOMMAND:
            cmd = int(wParam) & 0xFFF0
            if cmd in (SC_CLOSE, SC_MINIMIZE, SC_MAXIMIZE, SC_RESTORE):
                user32.ShowWindow(wt.HWND(hWnd), SW_RESTORE)
                return 0
        elif msg == WM_SIZE:
            if int(wParam) == SIZE_MINIMIZED:
                user32.ShowWindow(wt.HWND(hWnd), SW_RESTORE)
                return 0
        elif msg == WM_CLOSE:
            return 0
        return user32.CallWindowProcW(old_wndproc, wt.HWND(hWnd), msg, wParam, lParam)

    new_wndproc = WNDPROC(_proc)
    old_wndproc.value = user32.SetWindowLongPtrW(hwnd, GWL_WNDPROC, ct.cast(new_wndproc, ct.c_void_p))
    root._nulla_old_wndproc = old_wndproc
    root._nulla_new_wndproc = new_wndproc
    root._nulla_user32 = user32

# --- Geometry helper: center-left placement
def _geo_center_left(root: tk.Tk, win_w: int, win_h: int) -> str:
    """
    Place the window so its center sits at 25% of screen width (left-center),
    and vertically centered.
    """
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = int(max(0, (sw * 0.25) - (win_w / 2)))
    y = int(max(0, (sh * 0.5)  - (win_h / 2)))
    return f"{win_w}x{win_h}+{x}+{y}"

class Portrait(tk.Tk):
    def __init__(self, ppid: int | None):
        super().__init__()
        self.title(TITLE)
        self.configure(bg="#000000")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", lambda: None)

        # hooks & bounce
        try:
            _hook_system_buttons(self)
        except Exception:
            pass
        self.bind("<Unmap>", lambda e: self.after(0, self._bounce))
        self.after(150, self._pulse_bounce)

        # UI (no padding; exact-fit to image)
        self.label = tk.Label(self, bg="#000000", bd=0, highlightthickness=0)
        self.label.pack(fill="both", expand=True)

        if ppid:
            threading.Thread(target=_watch_parent, args=(ppid,), daemon=True).start()

        self.after(50, self._show_image)

    # ---- bounce helpers ----
    def _bounce(self):
        try:
            self.deiconify()
            self.attributes("-topmost", True)
            self.after(20, lambda: self.attributes("-topmost", False))
        except Exception:
            pass

    def _pulse_bounce(self):
        try:
            if self.state() == "iconic":
                self._bounce()
        except Exception:
            pass
        self.after(150, self._pulse_bounce)

    # ---- image loader (Pillow preferred; scales to SCALE) ----
    def _show_image(self):
        path = IMAGE_PATH
        if not os.path.isfile(path):
            self.label.config(text=f"Missing portrait:\n{path}", fg="#aaaaaa")
            # default placeholder size, still center-left
            geo = _geo_center_left(self, 320, 200)
            self.geometry(geo)
            return

        try:
            from PIL import Image, ImageTk  # type: ignore
            im = Image.open(path).convert("RGBA")
            src_w, src_h = im.size

            # exact SCALE (no letterboxing, no extra canvas)
            new_w = max(1, int(round(src_w * SCALE)))
            new_h = max(1, int(round(src_h * SCALE)))
            if new_w != src_w or new_h != src_h:
                im = im.resize((new_w, new_h), Image.LANCZOS)

            self._img = ImageTk.PhotoImage(im)
            self.label.config(image=self._img)

            # resize window to match image exactly, placed center-left
            geo = _geo_center_left(self, new_w, new_h)
            self.geometry(geo)

        except Exception as e:
            # Fallback: Tk-only (unscaled). Geometry = original size, placed center-left.
            try:
                from tkinter import PhotoImage
                self._img = PhotoImage(file=path)
                self.label.config(image=self._img)
                geo = _geo_center_left(self, self._img.width(), self._img.height())
                self.geometry(geo)
            except Exception as ee:
                self.label.config(text=f"Portrait load error:\n{e}\n{ee}", fg="#aaaaaa")
                geo = _geo_center_left(self, 320, 200)
                self.geometry(geo)

def run_portrait():
    ppid = None
    for arg in sys.argv[1:]:
        if arg.startswith("--ppid="):
            try:
                ppid = int(arg.split("=", 1)[1])
            except Exception:
                pass
    app = Portrait(ppid)
    app.mainloop()

if __name__ == "__main__":
    # Ensure Pillow is installed for clean scaling:
    # C:\Users\Tsoxer\nulla_source\XTTS-v2\.venv\Scripts\python.exe -m pip install pillow
    run_portrait()
