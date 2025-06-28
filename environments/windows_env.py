from PIL import Image
import pyscreenshot as ImageGrab

from skills.mouse_skills import MouseSkill
from skills.keyboard_skills import KeyboardSkill

class WindowsEnv:
    """
    An environment class to interact with the Windows OS.
    This class provides methods for capturing the screen and executing
    low-level commands.
    """
    def __init__(self, mouse_backend: str | None = None):
        self.mouse = MouseSkill(backend=mouse_backend)
        self.keyboard = KeyboardSkill()

    def capture_screen(self) -> Image.Image:
        """
        Captures the current screen.

        Returns:
            A PIL Image object of the screen.
        """
        print("Capturing screen...")
        screenshot = ImageGrab.grab()
        return screenshot

    def execute_commands(self, commands: list):
        """
        Executes a sequence of low-level commands.

        Args:
            commands: A list of command tuples, e.g., 
                      [('mouse_click', (100, 150)), ('kb_type', ('hello',))]
        """
        for command, args in commands:
            try:
                if command == "mouse_move":
                    x, y = args
                    self.mouse.move(x, y)
                elif command == "mouse_click":
                    x, y, *rest = args
                    button = rest[0] if rest else "left"
                    self.mouse.click(x, y, button=button)
                elif command == "mouse_scroll":
                    amount, *rest = args
                    direction = rest[0] if rest else "down"
                    self.mouse.scroll(amount, direction)
                elif command == "kb_type":
                    text, = args
                    self.keyboard.type(text)
                elif command == "kb_press":
                    key, = args
                    self.keyboard.press(key)
                else:
                    print(f"[WindowsEnv] Unknown command: {command}")
            except Exception as e:
                print(f"[WindowsEnv] Error executing {command}: {e}") 