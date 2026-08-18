from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import propuot.models.propuot as propuot_module
from propuot.config import build_test_parser, build_train_parser
from propuot.data.datasets import ObservedBatchSampler
from propuot.models.backbones import CXRResNet34
from propuot.models.losses import PropensityLoss, propensity_odds


class DummyEHR(nn.Module):
    def __init__(self, hidden_dim: int, **_: object) -> None:
        super().__init__()
        self.projection = nn.Linear(76, hidden_dim)
        self.feats_dim = hidden_dim

    def forward(self, values: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        del lengths
        return self.projection(values.mean(dim=1))


class DummyCXR(nn.Module):
    feats_dim = 7

    def __init__(self, **_: object) -> None:
        super().__init__()
        self.projection = nn.Linear(3, self.feats_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.mean(dim=(2, 3)))


class DummyText(nn.Module):
    feats_dim = 7

    def __init__(self, *_: object, **__: object) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(self.feats_dim))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        values = (input_ids * attention_mask).float().mean(dim=(1, 2), keepdim=False)
        return values[:, None] * self.scale[None, :]


def arguments(modalities: str, setting: str = "partial") -> SimpleNamespace:
    return SimpleNamespace(
        modalities=modalities,
        setting=setting,
        dim=8,
        ehr_layers=2,
        ehr_dropout=0.1,
        pretrained_vision=False,
        text_model="unused",
        text_hidden_size=7,
        text_output_size=7,
        head_dropout=0.1,
        propensity_entropy_weight=0.005,
        propensity_entropy_sign=-1.0,
        sinkhorn_blur=0.1,
        sinkhorn_reach=0.1,
        load_state_cxr=None,
        omega_gamma=0.5,
        propensity_eps=1e-6,
        propensity_bound=0.05,
        omega_max=10.0,
    )


def batch(modalities: str) -> dict:
    size = 4
    result = {
        "ehr": torch.randn(size, 6, 76),
        "lengths": [6, 5, 4, 6],
        "image": torch.randn(size, 3, 16, 16),
        "input_ids": torch.randint(0, 20, (size, 5 if modalities == "ehr-note" else 4, 12)),
        "attention_mask": torch.ones(size, 5 if modalities == "ehr-note" else 4, 12),
        "presence": torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
    }
    if modalities == "ehr-cxr-note":
        result["presence"] = torch.ones(size, 2)
    return result


class PropUOTTests(unittest.TestCase):
    def test_cmcm_cxr_backbone_checkpoint(self) -> None:
        model = CXRResNet34(pretrained=False)
        expected = torch.full_like(model.encoder.conv1.weight, 0.25)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_checkpoint.pth.tar"
            torch.save(
                {
                    "state_dict": {
                        "cxr_model.vision_backbone.conv1.weight": expected,
                    }
                },
                checkpoint,
            )
            loaded = model.load_backbone_checkpoint(checkpoint)
        self.assertEqual(loaded, 1)
        self.assertTrue(torch.equal(model.encoder.conv1.weight, expected))

    def test_release_defaults(self) -> None:
        train_parser = build_train_parser()
        self.assertEqual(train_parser.get_default("seed"), 13)
        self.assertEqual(train_parser.get_default("ehr_layers"), 2)
        self.assertEqual(train_parser.get_default("ehr_dropout"), 0.2)
        self.assertEqual(train_parser.get_default("head_dropout"), 0.1)
        self.assertEqual(train_parser.get_default("adam_beta1"), 0.9)
        self.assertEqual(train_parser.get_default("adam_beta2"), 0.9)
        self.assertEqual(train_parser.get_default("sinkhorn_blur"), 0.1)
        self.assertEqual(train_parser.get_default("sinkhorn_reach"), 0.1)
        self.assertIsNone(train_parser.get_default("load_state_cxr"))
        self.assertEqual(build_test_parser().get_default("seed"), 13)

    def test_propensity_components_are_finite(self) -> None:
        logits = torch.tensor([-20.0, 0.0, 20.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0, 1.0])
        loss = PropensityLoss()(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        odds = propensity_odds(torch.tensor([0.0, 0.5, 1.0]), gamma=0.5)
        self.assertTrue(torch.isfinite(odds).all())
        self.assertLessEqual(float(odds.max()), 10.0)

    @patch.object(propuot_module, "TinyBERTEncoder", DummyText)
    @patch.object(propuot_module, "CXRResNet34", DummyCXR)
    @patch.object(propuot_module, "EHRLSTM", DummyEHR)
    def test_bimodal_forward_and_backward(self) -> None:
        for modalities in ("ehr-note", "ehr-cxr"):
            model = propuot_module.PropUOTBiModal(arguments(modalities))
            output = model(batch(modalities))
            self.assertEqual(output["logits"].shape, (4, 1))
            loss = output["logits"].mean() + output["propensity_loss"] + output["uot_loss"]
            self.assertTrue(torch.isfinite(loss))
            loss.backward()

            inference = model(batch(modalities), compute_auxiliary_losses=False)
            self.assertEqual(float(inference["propensity_loss"]), 0.0)
            self.assertEqual(float(inference["uot_loss"]), 0.0)

    @patch.object(propuot_module, "TinyBERTEncoder", DummyText)
    @patch.object(propuot_module, "CXRResNet34", DummyCXR)
    @patch.object(propuot_module, "EHRLSTM", DummyEHR)
    def test_paired_and_trimodal_paths(self) -> None:
        paired = propuot_module.PropUOTBiModal(arguments("ehr-cxr", "paired"))
        paired_batch = batch("ehr-cxr")
        paired_batch["presence"] = torch.ones(4, 1)
        self.assertEqual(float(paired(paired_batch)["propensity_loss"]), 0.0)

        tri = propuot_module.PropUOTTriModal(arguments("ehr-cxr-note", "paired"))
        output = tri(batch("ehr-cxr-note"))
        self.assertEqual(output["logits"].shape, (4, 1))
        self.assertTrue(torch.isfinite(output["uot_loss"]))

    def test_observed_batch_sampler(self) -> None:
        # Special methods are looked up on the type, so use a tiny concrete class.
        dataset_type = type(
            "DatasetStub",
            (),
            {"observed_indices": [0, 1, 2], "__len__": lambda self: 20},
        )
        sampler = ObservedBatchSampler(dataset_type(), batch_size=4, seed=42)
        batches = list(sampler)
        self.assertEqual(len(batches), 5)
        for indices in batches:
            self.assertEqual(len(indices), 4)
            self.assertTrue(any(index in {0, 1, 2} for index in indices))


if __name__ == "__main__":
    unittest.main()
