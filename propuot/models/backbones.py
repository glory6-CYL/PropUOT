from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel
from torchvision.models import ResNet34_Weights, resnet34


class EHRLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 76,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.feats_dim = hidden_dim
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, parameter in self.encoder.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def forward(self, values: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            values, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.encoder(packed)
        return self.output_dropout(hidden[-1])


class CXRResNet34(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        self.encoder = resnet34(weights=weights)
        self.feats_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def load_backbone_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Load ResNet weights from a CMCM-style derived checkpoint."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"CXR checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, dict):
            raise ValueError(f"CXR checkpoint has no state_dict: {path}")

        prefixes = (
            "cxr_model.vision_backbone.",
            "vision_backbone.",
            "auxiliary_encoder.encoder.",
            "cxr_encoder.encoder.",
            "encoder.",
        )
        own_state = self.encoder.state_dict()
        selected = {}
        for raw_name, raw_value in state.items():
            name = str(raw_name).removeprefix("module.")
            for prefix in prefixes:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    break
            value = raw_value.data if isinstance(raw_value, nn.Parameter) else raw_value
            if name in own_state and isinstance(value, torch.Tensor):
                if own_state[name].shape == value.shape:
                    selected[name] = value

        if not selected:
            raise ValueError(f"No compatible ResNet-34 weights found in {path}")
        self.encoder.load_state_dict(selected, strict=False)
        print(f"Loaded {len(selected)} CXR backbone tensors from {path}", flush=True)
        return len(selected)


class TinyBERTEncoder(nn.Module):
    def __init__(self, model_name: str, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        actual_size = int(self.encoder.config.hidden_size)
        if actual_size != hidden_size:
            raise ValueError(
                f"Configured text hidden size {hidden_size} does not match "
                f"{model_name} ({actual_size})"
            )
        dropout = float(getattr(self.encoder.config, "hidden_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, output_size)
        self.feats_dim = output_size

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, note_count, sequence_length = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * note_count, sequence_length)
        flat_mask = attention_mask.reshape(batch_size * note_count, sequence_length)
        encoded = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        cls = encoded.last_hidden_state[:, 0]
        cls = self.dropout(cls).reshape(batch_size, note_count, -1).mean(dim=1)
        return self.projection(cls)
