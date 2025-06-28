try:
    import pyautogui  # type: ignore
except Exception as e:  # pragma: no cover
    # Create a lightweight stub that exposes the minimal API we use.
    class _PyAutoGUIStub:  # noqa: D101
        def write(self, *args, **kwargs):
            pass

        def press(self, *args, **kwargs):
            pass

    pyautogui = _PyAutoGUIStub()  # type: ignore
    print(f"[KeyboardSkill] pyautogui unavailable, using no-op stub ({e})")

from .base_skill import BaseSkill

class KeyboardSkill(BaseSkill):
    """
    A collection of skills for controlling the keyboard.
    """

    def execute(self, *args, **kwargs):
        """The default execute can point to the primary skill, like 'type'."""
        self.type(*args, **kwargs)

    def type(self, text: str, interval: float = 0.05):
        """Types a given string of text."""
        pyautogui.write(text, interval=interval)
        print(f"Typed: '{text}'")

    def press(self, key: str):
        """Presses a single key."""
        pyautogui.press(key)
        print(f"Pressed key: '{key}'") 