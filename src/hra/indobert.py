import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from hra.data import ASPECTS, CLASSES

ENCODER = "indobenchmark/indobert-base-p1"


class AspectModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(encoder.config.hidden_size, len(ASPECTS) * len(CLASSES))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(self.dropout(pooled)).view(-1, len(ASPECTS), len(CLASSES))

    def save(self, out_dir, tokenizer, max_length):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.config.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        torch.save(self.state_dict(), out_dir / "model.pt")
        (out_dir / "meta.json").write_text(json.dumps({"max_length": max_length}) + "\n")

    @classmethod
    def load(cls, model_dir, device="cpu"):
        model_dir = Path(model_dir)
        model = cls(AutoModel.from_config(AutoConfig.from_pretrained(model_dir)))
        model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device, weights_only=True))
        meta = json.loads((model_dir / "meta.json").read_text())
        return model.to(device).eval(), AutoTokenizer.from_pretrained(model_dir), meta["max_length"]


def class_weights(y, power=0.5):
    w = np.ones((y.shape[1], len(CLASSES)), dtype=np.float32)
    for j in range(y.shape[1]):
        counts = np.bincount(y[:, j], minlength=len(CLASSES)) + 1
        w[j] = (counts.sum() / counts) ** power
        w[j] /= w[j].mean()
    return torch.tensor(w)


def loss_fn(logits, y, weights):
    losses = [nn.functional.cross_entropy(logits[:, j], y[:, j], weight=weights[j]) for j in range(y.shape[1])]
    return torch.stack(losses).mean()


@torch.no_grad()
def predict(model, tokenizer, texts, max_length=256, batch_size=32, device="cpu"):
    model.eval()
    probs = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i : i + batch_size], truncation=True, max_length=max_length, padding=True, return_tensors="pt"
        )
        logits = model(**{k: v.to(device) for k, v in enc.items()})
        probs.append(torch.softmax(logits, -1).cpu())
    p = torch.cat(probs).numpy()
    return p.argmax(-1), p
