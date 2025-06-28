"""MouseSkill: high-speed mouse operations with optional KeyWin backend.

If the `keywin` package (https://github.com/winstxnhdw/KeyWin) is available, we
leverage its C-extension around Win32 `SendInput` for near-instantaneous mouse
movements and clicks. Otherwise we gracefully fall back to `pyautogui`.
"""

from .base_skill import BaseSkill

try:
    import keywin.input as kwin  # type: ignore

    _HAS_KEYWIN = True
except ImportError:  # Fallback to pyautogui
    _HAS_KEYWIN = False
    # Attempt to import pyautogui, but gracefully degrade in headless setups
    try:
        import pyautogui  # type: ignore
    except Exception as e:  # pragma: no cover
        # Create a lightweight stub that exposes the minimal API we use.
        class _PyAutoGUIStub:  # noqa: D101
            def moveTo(self, *args, **kwargs):
                pass

            def click(self, *args, **kwargs):
                pass

            def scroll(self, *args, **kwargs):
                pass

        pyautogui = _PyAutoGUIStub()  # type: ignore
        print(f"[MouseSkill] pyautogui unavailable, using no-op stub ({e})")


class MouseSkill(BaseSkill):
    """A collection of skills for controlling the mouse.

    Args:
        backend: "keywin" (fast, requires keywin package) or "pyautogui".
    """

    def __init__(self, backend: str | None = None):
        if backend is None:
            backend = "keywin" if _HAS_KEYWIN else "pyautogui"
        self.backend = backend

    # ---------------- public API ----------------
    def execute(self, *args, **kwargs):
        """Default action (alias for click)."""
        self.click(*args, **kwargs)

    def move(self, x: int, y: int):
        """Instantly moves cursor to (x,y)."""
        if self.backend == "keywin":
            kwin.send_mouse_move(x, y)  # type: ignore[attr-defined]
        else:
            # pyautogui: duration=0 for instant move
            pyautogui.moveTo(x, y, duration=0)

    def click(self, x: int, y: int, button: str = "left"):
        """Moves to (x,y) then clicks with specified button."""
        self.move(x, y)
        if self.backend == "keywin":
            kwin.send_mouse_down(button)  # type: ignore[attr-defined]
            kwin.send_mouse_up(button)    # type: ignore[attr-defined]
        else:
            pyautogui.click(button=button)
        print(f"[MouseSkill] Clicked {button} at ({x}, {y}) via {self.backend}")

    def scroll(self, amount: int, direction: str = "down"):
        """Scrolls the wheel by `amount` units."""
        if self.backend == "keywin":
            kwin.send_mouse_scroll(amount if direction == "up" else -amount)  # type: ignore[attr-defined]
        else:
            scroll_amount = -amount if direction == "down" else amount
            pyautogui.scroll(scroll_amount)
        print(f"[MouseSkill] Scrolled {direction} by {amount} via {self.backend}") 