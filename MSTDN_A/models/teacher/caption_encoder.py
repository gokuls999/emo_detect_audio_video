from __future__ import annotations

import torch
from torch import nn

try:
    from transformers import AutoTokenizer, CLIPTextModel
except Exception:
    CLIPTextModel = None
    AutoTokenizer = None


class FrozenCaptionEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", output_dim: int = 256) -> None:
        super().__init__()
        self.enabled = CLIPTextModel is not None and AutoTokenizer is not None
        self.output_dim = output_dim
        if self.enabled:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                self.model = CLIPTextModel.from_pretrained(model_name, local_files_only=True)
                for param in self.model.parameters():
                    param.requires_grad = False
                hidden = getattr(self.model.config, "hidden_size", 512)
                self.proj = nn.Linear(hidden, output_dim)
            except Exception:
                self.enabled = False
        if not self.enabled:
            self.register_buffer("dummy", torch.zeros(output_dim), persistent=False)
            self.proj = nn.Identity()

    def forward(self, captions: list[str]) -> torch.Tensor:
        if not self.enabled:
            return self.dummy.unsqueeze(0).repeat(len(captions), 1)
        tokens = self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
        tokens = {key: value.to(self.proj.weight.device) for key, value in tokens.items()}
        outputs = self.model(**tokens)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return self.proj(pooled)
