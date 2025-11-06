# nulla_source\scripts\nulla_list_game.py
from __future__ import annotations

import importlib, importlib.util, sys, os
from typing import Callable, Dict, Tuple, Any, Optional, List


def _dbg(msg: str) -> None:
    try:
        print(f"[GameManager] {msg}", flush=True)
    except Exception:
        pass


class GameManager:
    def __init__(
        self,
        *,
        get_story_window: Callable[[], Any],
        story_push: Callable[[str], None],
        enqueue_sentence_if_ready: Callable[[str], None],
        mark_nulla_spoke: Callable[[], None],
        idle_touch: Callable[[], None],
        # Optional: let overlays pause idle/short-idle while active
        idle_block_push: Optional[Callable[[], None]] = None,
        idle_block_pop: Optional[Callable[[], None]] = None,
        **kwargs,  # future-proof: ignore unknown kwargs
    ):
        self.get_story_window = get_story_window
        self.story_push = story_push
        self.enqueue_sentence_if_ready = enqueue_sentence_if_ready
        self.mark_nulla_spoke = mark_nulla_spoke
        self.idle_touch = idle_touch

        # Hooks shared with overlays
        self.hooks: Dict[str, Any] = {
            "story_push": self.story_push,
            "enqueue_sentence_if_ready": self.enqueue_sentence_if_ready,
            "mark_nulla_spoke": self.mark_nulla_spoke,
            "idle_touch": self.idle_touch,
        }
        if callable(idle_block_push):
            self.hooks["idle_block_push"] = idle_block_push
        if callable(idle_block_pop):
            self.hooks["idle_block_pop"] = idle_block_pop

        # Games registry
        self.games: Dict[str, Dict[str, str]] = {
            "snake": {
                "slug": "snake",
                "title": "Snake",
                "module": "nulla_game_snake",
                "summary": "Classic Snake with a clean overlay and scoreboard.",
            },
            "runner": {
                "slug": "runner",
                "title": "Runner",
                "module": "nulla_game_runner",
                "summary": "Endless runner: Space to jump over obstacles.",
            },
            "rps": {
                "slug": "rps",
                "title": "Rock Paper Scissors",
                "module": "nulla_game_rps",
                "summary": "Rock Paper Scissors. First to 3/5/10. Click a tile, then Shoot!",
            },
        }
        self._order = ["snake", "runner", "rps"]
        self._active_mod_name: Optional[str] = None
        self._search_dirs = self._compute_search_dirs()

    # ---------- UX ----------
    def matches_help(self, text: str) -> bool:
        t = text.strip().lower()
        return t in ("help game", "help games", "game help")

    def help_text(self) -> str:
        # Speak a single line via XTTS once when help is requested
        try:
            if callable(self.enqueue_sentence_if_ready):
                self.enqueue_sentence_if_ready("Here are the list of games available.")
            if callable(self.mark_nulla_spoke):
                self.mark_nulla_spoke()
            if callable(self.idle_touch):
                self.idle_touch()
        except Exception:
            pass

        lines = ["Here are the list of games available. (type it exactly like this) (press P to pause):"]
        for key in self._order:
            g = self.games[key]
            lines.append(f"- play {g['slug']}  —  {g['summary']}")
        lines.append("Enjoy! :)")
        return "\n".join(lines)

    # ---------- Actions ----------
    def try_play(self, text: str) -> Tuple[bool, str]:
        t = text.strip().lower()
        if not t.startswith("play "):
            return (False, "")
        slug = t.split(" ", 1)[1].strip()

        g = self.games.get(slug)
        if not g:
            return (False, "")

        story_win = self.get_story_window()
        if story_win is None:
            return (False, "Game UI couldn’t attach (no chat window).")

        # try primary + a few common alternates
        mod_candidates = [g["module"]]
        if slug == "snake":
            mod_candidates += ["nulla_game_snake", "game_snake", "snake_overlay"]
        elif slug == "runner":
            mod_candidates += ["nulla_game_runner", "game_runner", "runner_overlay"]
        elif slug == "rps":
            mod_candidates += ["nulla_game_rps", "game_rps", "rps_overlay"]

        # dedupe while preserving order
        seen = set()
        mod_candidates = [m for m in mod_candidates if not (m in seen or seen.add(m))]

        for name in mod_candidates:
            mod, err = self._load_module_anywhere(name)
            if not mod:
                _dbg(f"Import miss for {name}: {repr(err)}")
                continue
            if not hasattr(mod, "mount_overlay"):
                _dbg(f"{name} found but has no mount_overlay()")
                continue

            try:
                mod.mount_overlay(story_win, self.hooks)
                self._active_mod_name = name
                return (True, f"Launching {g['title']}…")
            except Exception as e:
                _dbg(f"{name}.mount_overlay() failed: {e!r}")
                return (False, f"Couldn’t start {g['title']}: {type(e).__name__}: {e}")

        return (False, f"{g['title']} isn’t installed yet.")

    def exit_game(self, confirm: bool = True) -> Tuple[bool, str]:
        candidates: List[str] = []
        if self._active_mod_name:
            candidates.append(self._active_mod_name)
        for fallback in (
            "nulla_game_snake", "game_snake", "snake_overlay",
            "nulla_game_runner", "game_runner", "runner_overlay",
            "nulla_game_rps", "game_rps", "rps_overlay",
        ):
            if fallback not in candidates:
                candidates.append(fallback)

        for name in candidates:
            try:
                mod = sys.modules.get(name)
                if not mod:
                    continue
                if hasattr(mod, "exit_game"):
                    mod.exit_game(confirm=confirm)
                    return (True, "Closing the current game.")
                for fn in ("hide_overlay", "destroy_overlay"):
                    if hasattr(mod, fn):
                        getattr(mod, fn)()
                        return (True, "Closing the current game.")
            except Exception as e:
                return (False, f"Couldn’t close: {e}")
        return (False, "No game is running.")

    # ---------- Loader helpers ----------
    def _compute_search_dirs(self) -> List[str]:
        dirs: List[str] = []
        here = os.path.dirname(__file__)
        dirs.append(here)

        # Where the main script lives (often your build root or python_scripts)
        mainfile = getattr(sys.modules.get("__main__"), "__file__", None)
        if mainfile:
            dirs.append(os.path.dirname(mainfile))

        parent = os.path.dirname(here)
        for cand in (
            parent,
            os.path.join(parent, "python_scripts"),
            os.getcwd(),
        ):
            if cand and os.path.isdir(cand):
                dirs.append(cand)

        seen: List[str] = []
        for d in dirs:
            if d not in seen and os.path.isdir(d):
                seen.append(d)
        return seen

    def _load_module_anywhere(self, mod_name: str):
        last_err = None

        # already loaded
        if mod_name in sys.modules:
            return sys.modules[mod_name], None

        # 1) package-relative
        try:
            pkg = __package__ or (__name__.rpartition(".")[0] if "." in __name__ else None)
            if pkg:
                return importlib.import_module(f".{mod_name}", pkg), None
        except Exception as e:
            last_err = e

        # 2) absolute
        try:
            return importlib.import_module(mod_name), None
        except Exception as e:
            last_err = e

        # 3) search known dirs for a .py
        for d in self._search_dirs:
            path = os.path.join(d, f"{mod_name}.py")
            if os.path.isfile(path):
                try:
                    spec = importlib.util.spec_from_file_location(mod_name, path)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules[mod_name] = mod
                        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                        return mod, None
                except Exception as e:
                    last_err = e

        return None, last_err
