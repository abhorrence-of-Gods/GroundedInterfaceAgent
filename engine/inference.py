from omegaconf import DictConfig
from models.gia_agent import GiaAgent
from environments.windows_env import WindowsEnv
import torch
from torchvision import transforms
import time

class InferenceEngine:
    """
    Handles the inference process for a trained GIA model.
    This class takes a high-level instruction, runs the agent in a live
    environment, and executes the generated actions.
    """
    def __init__(self, cfg: DictConfig, model: GiaAgent):
        self.cfg = cfg
        self.model = model.eval()  # evaluation mode
        self.env = WindowsEnv()

        # Simple transform to tensor for screenshot (RGB)
        self._transform = transforms.ToTensor()

    def run(self, instruction: str, execute_env: bool = True, screenshot=None):
        """
        Runs the agent to perform a task based on an instruction.

        Args:
            instruction: The high-level instruction for the agent to follow.
            execute_env: Whether to execute the environment commands.
        """
        print(f"[Inference] Executing instruction: '{instruction}'")

        # 1. Capture observation
        if screenshot is None:
            screenshot_pil = self.env.capture_screen()
        else:
            screenshot_pil = screenshot
        screenshot_tensor = self._transform(screenshot_pil).unsqueeze(0).to(next(self.model.parameters()).device)

        # 2. Encode current state
        with torch.no_grad():
            latent_state = self.model.encode_state([instruction], screenshot_tensor)

            # 3. Imagine future and pick first latent action of best trajectory (greedy)
            _, latent_actions, _, _ = self.model.plan_in_dream(latent_state)
            latent_action_first = latent_actions[0]

            # 4. Decode to concrete action (x_norm, y_norm, click_flag, scroll)
            concrete_action = self.model.action_tower.decode_action(latent_action_first)

        ca_np = concrete_action.detach().flatten().cpu().numpy()
        # Expected format: [x_norm, y_norm, click_flag, scroll] but fall back to zeros if shorter
        x_norm = ca_np[0] if len(ca_np) > 0 else 0.5
        y_norm = ca_np[1] if len(ca_np) > 1 else 0.5
        click_flag = ca_np[2] if len(ca_np) > 2 else 0.0
        # ignore scroll

        screen_w, screen_h = screenshot_pil.size
        x_px = int(x_norm * screen_w)
        y_px = int(y_norm * screen_h)

        # Decode warp (speed, precision) from the same latent action
        warp_params = (
            self.model.action_tower.decode_warp(latent_action_first)
            .detach()
            .cpu()
            .numpy()
        )
        if warp_params.size > 0:
            speed_val = float(warp_params.flatten()[0])
        else:
            speed_val = 0.5
        speed = max(speed_val, 0.05)  # seconds delay lower bound

        cmd_list = []
        if click_flag > 0.5:
            cmd_list.append(("mouse_click", (x_px, y_px, "left")))
        else:
            cmd_list.append(("mouse_move", (x_px, y_px)))

        if execute_env:
            self.env.execute_commands(cmd_list)
            time.sleep(speed)
            print(f"[Inference] Finished one-step execution (speed={speed:.2f}s).")
        return cmd_list 