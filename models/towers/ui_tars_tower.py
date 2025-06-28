import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from torchvision import transforms
from importlib import util as _import_util

class UITarsVisionTower(nn.Module):
    """Perception tower that wraps the open-source UI-TARS-1.5-7B vision encoder.

    It loads the vision branch (CLIP-like) and returns a 4096-dim embedding
    compatible with the common latent space used by Bridge.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-vision",  # lighter vision-only ckpt (~5 GB)
        is_frozen: bool = True,
        use_8bit: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.is_frozen = is_frozen
        self.use_8bit = use_8bit

        # UI-TARS 1.5 checkpoints are based on the Qwen-2.5-VL family and do **not** ship a separate
        # "vision" sub-folder on the 🤗 Hub. Trying to force the sub-folder therefore breaks with a
        # "config.json not found" / "Unrecognised model" error. We fall back to the following
        # strategy:
        #   1) Always load the generic processor so we can build pixel inputs.
        #   2) Try to grab the dedicated vision tower (`model.vision_tower`) if it exists.  This is
        #      light-weight and avoids loading the full LLM.
        #   3) Otherwise fall back to the full model.  This is heavier but unblocks training.

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        # If bitsandbytes is available and use_8bit is True, load in 8-bit to save VRAM
        _load_kwargs = {"trust_remote_code": True}
        if self.use_8bit and _import_util.find_spec("bitsandbytes") is not None:
            _load_kwargs["load_in_8bit"] = True

        full_model = AutoModel.from_pretrained(model_name, **_load_kwargs)

        # Qwen2.5-VL style: the vision branch is exposed via `vision_tower`.
        self.vision_model = getattr(full_model, "vision_tower", full_model)

        hidden_size = getattr(self.vision_model.config, "hidden_size", 3584)
        target_dim = 4096
        self.projection = nn.Linear(hidden_size, target_dim) if hidden_size != target_dim else nn.Identity()

        # Freeze if requested
        if self.is_frozen:
            for p in self.vision_model.parameters():
                p.requires_grad = False
            self.vision_model.eval()

    @property
    def output_dim(self):
        return 4096

    def forward(self, screenshot_batch: torch.Tensor) -> torch.Tensor:
        """Expects input tensor in range [0,1] with shape (B,C,H,W)."""
        device = next(self.parameters()).device
        # AutoProcessor for Qwen/UI-TARS can ingest either PIL images or numpy arrays.
        # Convert each input tensor (B,C,H,W in [0,1]) to PIL so we do not rely on
        # any optional `postprocess_image` helpers that may be absent in the
        # installed transformers version.
        to_pil = transforms.ToPILImage()
        images = [to_pil(img.cpu()) for img in screenshot_batch]
        # Qwen2-VL style processors expect a `text` input alongside `images`
        # and internally iterate over it. Passing `None` leads to a TypeError
        # (`argument of type 'NoneType' is not iterable`). We therefore supply
        # a list of empty strings – one per image – which will be ignored by
        # the model but keeps the processor happy.
        # Provide one image placeholder token per sample so that the number of
        # <image> tokens matches the extracted visual features. This avoids
        # "tokens: 0, features X" mismatches inside the Qwen vision tower.
        image_token_placeholder = getattr(self.processor, "image_token", "<img>")
        dummy_text = [image_token_placeholder] * len(images)
        inputs = self.processor(text=dummy_text, images=images, return_tensors="pt").to(device)

        # The dedicated vision tower only needs pixel_values. Passing text-related
        # fields can cause dtype mismatches (float vs. long). Extract the needed
        # tensor and feed it explicitly.
        pixel_values = inputs["pixel_values"]
        input_ids = inputs.get("input_ids", None)
        attention_mask = inputs.get("attention_mask", None)
        image_grid_thw = inputs.get("image_grid_thw", None)

        # The Qwen vision tower expects Long tensors for ids/masks.
        if isinstance(input_ids, torch.Tensor) and input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        if isinstance(attention_mask, torch.Tensor) and attention_mask.dtype != torch.long:
            attention_mask = attention_mask.long()

        with torch.no_grad() if self.is_frozen else torch.enable_grad():
            outputs = self.vision_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
            )

            # Different Qwen/VL variants expose visual embeddings differently.
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                # CLS token (first) as global representation
                pooled = outputs.last_hidden_state[:, 0]
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                pooled = outputs[0][:, 0]
            else:
                raise RuntimeError("Unable to extract pooled visual embedding from model output")

            # Align dtype with projection layer to avoid matmul dtype mismatch
            proj_dtype = self.projection.weight.dtype
            if pooled.dtype != proj_dtype:
                pooled = pooled.to(proj_dtype)

            embedding = self.projection(pooled)
        return embedding 