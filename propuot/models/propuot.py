from __future__ import annotations

import torch
from torch import nn

from .backbones import CXRResNet34, EHRLSTM, TinyBERTEncoder
from .losses import PropensityLoss, UnbalancedOTLoss, propensity_odds


class _PropUOTBase(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.propensity_loss = PropensityLoss(
            entropy_weight=args.propensity_entropy_weight,
            entropy_sign=args.propensity_entropy_sign,
        )
        self.uot_loss = UnbalancedOTLoss(
            blur=args.sinkhorn_blur,
            reach=args.sinkhorn_reach,
        )

    def _omega(self, logits: torch.Tensor) -> torch.Tensor:
        return propensity_odds(
            torch.sigmoid(logits),
            gamma=self.args.omega_gamma,
            eps=self.args.propensity_eps,
            bound=self.args.propensity_bound,
            maximum=self.args.omega_max,
        )


class PropUOTBiModal(_PropUOTBase):
    """PropUOT for EHR+Note (MIMIC-III) or EHR+CXR (MIMIC-IV)."""

    def __init__(self, args) -> None:
        super().__init__(args)
        self.ehr_encoder = EHRLSTM(
            hidden_dim=args.dim,
            num_layers=args.ehr_layers,
            dropout=args.ehr_dropout,
        )
        if args.modalities == "ehr-note":
            self.auxiliary_name = "note"
            self.auxiliary_encoder = TinyBERTEncoder(
                args.text_model, args.text_hidden_size, args.text_output_size
            )
        elif args.modalities == "ehr-cxr":
            self.auxiliary_name = "cxr"
            self.auxiliary_encoder = CXRResNet34(pretrained=args.pretrained_vision)
            if args.load_state_cxr:
                self.auxiliary_encoder.load_backbone_checkpoint(args.load_state_cxr)
        else:
            raise ValueError(f"Unsupported bi-modal configuration: {args.modalities}")

        auxiliary_dim = self.auxiliary_encoder.feats_dim
        self.auxiliary_projection = nn.Linear(auxiliary_dim, args.dim)
        self.propensity_network = nn.Sequential(
            nn.Linear(args.dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.residual_projection = nn.Sequential(
            nn.Linear(args.dim, args.dim),
            nn.GELU(),
            nn.Linear(args.dim, args.dim),
        )
        self.uot_source_projection = nn.Linear(args.dim, args.dim)
        self.missing_embedding = nn.Parameter(torch.randn(args.dim) * 0.02)
        self.predictor = nn.Sequential(
            nn.Linear(2 * args.dim, args.dim),
            nn.ReLU(),
            nn.Dropout(args.head_dropout),
            nn.Linear(args.dim, 1),
        )

    def _encode_auxiliary(self, batch: dict) -> torch.Tensor:
        if self.auxiliary_name == "note":
            return self.auxiliary_encoder(
                batch["input_ids"], batch["attention_mask"]
            )
        return self.auxiliary_encoder(batch["image"])

    def forward(
        self, batch: dict, *, compute_auxiliary_losses: bool = True
    ) -> dict[str, torch.Tensor]:
        ehr_features = self.ehr_encoder(batch["ehr"], batch["lengths"])
        auxiliary_features = self.auxiliary_projection(self._encode_auxiliary(batch))
        observed = batch["presence"][:, 0]
        observed_bool = observed.bool()

        propensity_logits = self.propensity_network(ehr_features).squeeze(-1)
        propensity_loss = (
            self.propensity_loss(propensity_logits, observed)
            if compute_auxiliary_losses and self.args.setting == "partial"
            else ehr_features.new_zeros(())
        )
        omega = self._omega(propensity_logits)
        correction = omega[:, None] * self.residual_projection(ehr_features)
        corrected_ehr = ehr_features + (~observed_bool).float()[:, None] * correction

        auxiliary_part = (
            observed[:, None] * auxiliary_features
            + (1.0 - observed[:, None]) * self.missing_embedding[None, :]
        )
        uot_loss = (
            self.uot_loss(
                self.uot_source_projection(corrected_ehr),
                auxiliary_features,
                observed_bool,
            )
            if compute_auxiliary_losses
            else ehr_features.new_zeros(())
        )
        fused = torch.cat((corrected_ehr, auxiliary_part), dim=-1)
        logits = self.predictor(fused)
        return {
            "logits": logits,
            "propensity_logits": propensity_logits,
            "propensity": torch.sigmoid(propensity_logits),
            "propensity_loss": propensity_loss,
            "uot_loss": uot_loss,
            "ehr_features": ehr_features,
            "auxiliary_features": auxiliary_features,
        }


class PropUOTTriModal(_PropUOTBase):
    """PropUOT for the fully matched EHR+CXR+radiology-report cohort.

    EHR is the anchor, CXR is the acquisition target for propensity/UOT, and
    the finalized radiology report is an additional prediction-time modality.
    """

    def __init__(self, args) -> None:
        super().__init__(args)
        self.ehr_encoder = EHRLSTM(
            hidden_dim=args.dim,
            num_layers=args.ehr_layers,
            dropout=args.ehr_dropout,
        )
        self.cxr_encoder = CXRResNet34(pretrained=args.pretrained_vision)
        if args.load_state_cxr:
            self.cxr_encoder.load_backbone_checkpoint(args.load_state_cxr)
        self.note_encoder = TinyBERTEncoder(
            args.text_model, args.text_hidden_size, args.text_output_size
        )
        self.cxr_projection = nn.Linear(self.cxr_encoder.feats_dim, args.dim)
        self.note_projection = nn.Linear(self.note_encoder.feats_dim, args.dim)

        self.propensity_network = nn.Sequential(
            nn.Linear(args.dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.residual_projection = nn.Sequential(
            nn.Linear(args.dim, args.dim),
            nn.GELU(),
            nn.Linear(args.dim, args.dim),
        )
        self.uot_source_projection = nn.Linear(args.dim, args.dim)
        self.missing_cxr_embedding = nn.Parameter(torch.randn(args.dim) * 0.02)
        self.missing_note_embedding = nn.Parameter(torch.randn(args.dim) * 0.02)
        self.predictor = nn.Sequential(
            nn.Linear(3 * args.dim, args.dim),
            nn.ReLU(),
            nn.Dropout(args.head_dropout),
            nn.Linear(args.dim, 1),
        )

    def forward(
        self, batch: dict, *, compute_auxiliary_losses: bool = True
    ) -> dict[str, torch.Tensor]:
        ehr_features = self.ehr_encoder(batch["ehr"], batch["lengths"])
        cxr_features = self.cxr_projection(self.cxr_encoder(batch["image"]))
        note_features = self.note_projection(
            self.note_encoder(batch["input_ids"], batch["attention_mask"])
        )
        cxr_observed = batch["presence"][:, 0]
        note_observed = batch["presence"][:, 1]
        propensity_logits = self.propensity_network(ehr_features).squeeze(-1)
        propensity_loss = (
            self.propensity_loss(propensity_logits, cxr_observed)
            if compute_auxiliary_losses and self.args.setting == "partial"
            else ehr_features.new_zeros(())
        )
        omega = self._omega(propensity_logits)
        correction = omega[:, None] * self.residual_projection(ehr_features)
        corrected_ehr = (
            ehr_features + (1.0 - cxr_observed[:, None]) * correction
        )

        cxr_part = (
            cxr_observed[:, None] * cxr_features
            + (1.0 - cxr_observed[:, None]) * self.missing_cxr_embedding[None, :]
        )
        note_part = (
            note_observed[:, None] * note_features
            + (1.0 - note_observed[:, None]) * self.missing_note_embedding[None, :]
        )
        uot_loss = (
            self.uot_loss(
                self.uot_source_projection(corrected_ehr),
                cxr_features,
                cxr_observed.bool(),
            )
            if compute_auxiliary_losses
            else ehr_features.new_zeros(())
        )
        logits = self.predictor(torch.cat((corrected_ehr, cxr_part, note_part), dim=-1))
        return {
            "logits": logits,
            "propensity_logits": propensity_logits,
            "propensity": torch.sigmoid(propensity_logits),
            "propensity_loss": propensity_loss,
            "uot_loss": uot_loss,
            "ehr_features": ehr_features,
            "cxr_features": cxr_features,
            "note_features": note_features,
        }


def build_propuot(args) -> nn.Module:
    if args.modalities == "ehr-cxr-note":
        return PropUOTTriModal(args)
    return PropUOTBiModal(args)
