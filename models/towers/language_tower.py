import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional

# Optional PEFT import
try:
    from peft import LoraConfig, get_peft_model
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False

class LanguageTower(nn.Module):
    """
    The Language Tower serves as the reasoning core of the GIA.
    It uses a pre-trained Large Language Model (LLM) to understand high-level
    instructions and generate abstract action plans.
    """
    def __init__(self, model_name: str, max_new_tokens: int, is_frozen: bool,
                 use_8bit: bool, use_lora: bool = False, lora_r: int = 8,
                 lora_alpha: int = 32, lora_dropout: float = 0.05):
        super().__init__()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.is_frozen = is_frozen
        
        # ------------------------------------------------------------------
        # Lightweight stub mode: if `model_name` is None or 'stub', we avoid
        # downloading a large LLM (useful for headless tests without GPU).
        # ------------------------------------------------------------------
        if self.model_name in [None, "", "stub", "none"]:
            print("[LanguageTower] Using lightweight stub (returns zeros)")
            self.tokenizer = None

            class _StubModel(nn.Module):
                def __init__(self, out_dim: int):
                    super().__init__()
                    self.out_dim = out_dim
                    self._dummy = nn.Parameter(torch.zeros(1))
                def forward(self, *args, **kwargs):
                    batch = kwargs.get('input_ids', torch.zeros(1)).shape[0]
                    device = kwargs.get('input_ids', torch.zeros(1)).device
                    return torch.zeros(batch, self.out_dim, device=device)

            hidden_size = 4096
            self.model = _StubModel(hidden_size)
            self.projection = nn.Identity()
            if self.is_frozen:
                for p in self.model.parameters():
                    p.requires_grad = False
        else:
            model_kwargs = {"device_map": "auto"}
            if use_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                model_kwargs["quantization_config"] = quantization_config

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

            # Output dimension expected by the Bridge / common latent space
            target_dim = 4096
            hidden_size = self.model.config.hidden_size if hasattr(self.model, "config") else None
            if hidden_size is not None and hidden_size != target_dim:
                self.projection = nn.Linear(hidden_size, target_dim)
            else:
                self.projection = nn.Identity()

            # ----- LoRA adaptation -----
            if use_lora and not self.is_frozen:
                if not _HAS_PEFT:
                    raise ImportError("peft is required for LoRA; install with `pip install peft`.")

                target_modules = [
                    name for name, _ in self.model.named_modules()
                    if any(k in name for k in ["q_proj", "v_proj", "k_proj", "o_proj"])
                ]
                lora_cfg = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                )
                self.model = get_peft_model(self.model, lora_cfg)
                print(f"[LanguageTower] LoRA enabled with r={lora_r}, alpha={lora_alpha}")

            if self.is_frozen:
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.eval()

    def forward(self, text: list[str]) -> torch.Tensor:
        """
        Gets the embedding for a given batch of text.
        """
        if self.tokenizer is None:
            # Stub mode: return zeros
            batch = len(text)
            device = next(self.model.parameters()).device if len(list(self.model.parameters())) else torch.device("cpu")
            return torch.zeros(batch, 4096, device=device)

        # Ensure the tokenizer can handle batches with padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use the hidden state of the last token as the embedding
            embedding = outputs.hidden_states[-1][:, -1, :]
            embedding = self.projection(embedding)
        return embedding