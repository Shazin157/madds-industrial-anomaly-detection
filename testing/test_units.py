"""
test_units.py
Unit tests for all core modules.
Run: python -m pytest test_units.py -v
"""

import pytest
import torch
import numpy as np
import cv2
from pathlib import Path

from modality_gate import detect_modality, preprocess_ir, preprocess_rgb, preprocess_uploaded
from anomaly_head   import BaselineBuilder, ZScoreAnomalyHead, get_device
from localization   import upsample, otsu_mask, get_bboxes


# ─── FIXTURES ────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_rgb_image():
    """Random RGB image — channels differ significantly → should detect as RGB."""
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return img

@pytest.fixture
def synthetic_ir_image():
    """Near-identical channel image → should detect as IR."""
    base  = np.random.randint(100, 200, (256, 256), dtype=np.uint8)
    noise = np.random.normal(0, 1.5, (256, 256)).astype(np.int16)
    ch    = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.merge([ch, ch, ch])

@pytest.fixture
def dummy_features():
    """Dummy feature dict simulating branch output."""
    return {
        "fused" : torch.randn(1, 196, 880),
        "local" : torch.randn(1, 196, 112),
        "global": torch.randn(1, 196, 768),
    }

@pytest.fixture
def dummy_baseline(dummy_features):
    """Build a baseline from 5 dummy images."""
    builder = BaselineBuilder("RGB", "Test")
    for _ in range(5):
        feats = {
            "fused" : torch.randn(1, 196, 880),
            "local" : torch.randn(1, 196, 112),
            "global": torch.randn(1, 196, 768),
        }
        builder.update(feats)
    builder.save("baselines/test_baseline.pt")
    from anomaly_head import load_baseline
    return load_baseline("baselines/test_baseline.pt")


# ─── MODALITY GATE TESTS ─────────────────────────────────────────────────────

class TestModalityGate:

    def test_detects_rgb(self, synthetic_rgb_image):
        result = detect_modality(synthetic_rgb_image)
        assert result == "RGB", f"Expected RGB, got {result}"

    def test_detects_ir(self, synthetic_ir_image):
        result = detect_modality(synthetic_ir_image)
        assert result == "IR", f"Expected IR, got {result}"

    def test_wrong_shape_raises(self):
        bad_image = np.zeros((256, 256), dtype=np.uint8)  # 2D grayscale
        with pytest.raises(ValueError):
            detect_modality(bad_image)

    def test_preprocess_rgb_shape(self, synthetic_rgb_image):
        tensor = preprocess_rgb(synthetic_rgb_image)
        assert tensor.shape == (1, 3, 224, 224), f"Got shape {tensor.shape}"

    def test_preprocess_rgb_dtype(self, synthetic_rgb_image):
        tensor = preprocess_rgb(synthetic_rgb_image)
        assert tensor.dtype == torch.float32

    def test_preprocess_ir_shape(self, synthetic_ir_image):
        tensor = preprocess_ir(synthetic_ir_image)
        assert tensor.shape == (1, 3, 224, 224), f"Got shape {tensor.shape}"

    def test_preprocess_ir_range(self, synthetic_ir_image):
        tensor = preprocess_ir(synthetic_ir_image)
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0, \
            f"IR tensor out of [0,1]: min={tensor.min()}, max={tensor.max()}"

    def test_preprocess_uploaded_returns_tuple(self, synthetic_rgb_image):
        tensor, modality = preprocess_uploaded(synthetic_rgb_image)
        assert isinstance(tensor, torch.Tensor)
        assert modality in ["RGB", "IR"]


# ─── BASELINE BUILDER TESTS ──────────────────────────────────────────────────

class TestBaselineBuilder:

    def test_update_increments_count(self):
        builder = BaselineBuilder("RGB", "Test")
        feat = {"fused": torch.randn(1,196,880),
                "local": torch.randn(1,196,112),
                "global":torch.randn(1,196,768)}
        builder.update(feat)
        assert builder.n_images == 1

    def test_save_creates_file(self, tmp_path):
        builder = BaselineBuilder("RGB", "Test")
        feat = {"fused": torch.randn(1,196,880),
                "local": torch.randn(1,196,112),
                "global":torch.randn(1,196,768)}
        builder.update(feat)
        path = str(tmp_path / "test.pt")
        builder.save(path)
        assert Path(path).exists()

    def test_baseline_contains_required_keys(self, tmp_path):
        builder = BaselineBuilder("RGB", "Test")
        feat = {"fused": torch.randn(1,196,880),
                "local": torch.randn(1,196,112),
                "global":torch.randn(1,196,768)}
        builder.update(feat)
        path = str(tmp_path / "test.pt")
        builder.save(path)

        import torch as th
        b = th.load(path, map_location="cpu", weights_only=False)
        for key in ["modality","mode","n_images","fused_mu","fused_sigma",
                    "local_mu","local_sigma","global_mu","global_sigma"]:
            assert key in b, f"Missing key: {key}"

    def test_sigma_positive(self, tmp_path):
        builder = BaselineBuilder("RGB", "Test")
        for _ in range(10):
            feat = {"fused": torch.randn(1,196,880),
                    "local": torch.randn(1,196,112),
                    "global":torch.randn(1,196,768)}
            builder.update(feat)
        path = str(tmp_path / "test.pt")
        builder.save(path)

        import torch as th
        b = th.load(path, map_location="cpu", weights_only=False)
        assert (b["fused_sigma"] > 0).all(), "Sigma must be strictly positive"

    def test_save_without_update_raises(self, tmp_path):
        builder = BaselineBuilder("RGB", "Test")
        with pytest.raises(AssertionError):
            builder.save(str(tmp_path / "empty.pt"))


# ─── Z-SCORE HEAD TESTS ───────────────────────────────────────────────────────

class TestZScoreHead:

    def test_score_returns_required_keys(self, dummy_baseline, dummy_features):
        head   = ZScoreAnomalyHead(dummy_baseline, threshold=3.0)
        result = head.score(dummy_features)
        for key in ["anomaly_map","score_fused","score_local",
                    "score_global","max_score","mean_score","is_anomalous"]:
            assert key in result, f"Missing key: {key}"

    def test_anomaly_map_shape(self, dummy_baseline, dummy_features):
        head   = ZScoreAnomalyHead(dummy_baseline)
        result = head.score(dummy_features)
        assert result["anomaly_map"].shape == (14, 14), \
            f"Expected (14,14), got {result['anomaly_map'].shape}"

    def test_is_anomalous_type(self, dummy_baseline, dummy_features):
        head   = ZScoreAnomalyHead(dummy_baseline)
        result = head.score(dummy_features)
        assert isinstance(result["is_anomalous"], bool)

    def test_known_normal_scores_low(self, tmp_path):
        """Features identical to baseline should produce near-zero Z-scores."""
        fixed_feat = torch.ones(1, 196, 880) * 0.5

        builder = BaselineBuilder("RGB", "Test")
        for _ in range(20):
            builder.update({
                "fused" : fixed_feat.clone(),
                "local" : torch.ones(1, 196, 112) * 0.5,
                "global": torch.ones(1, 196, 768) * 0.5,
            })
        path = str(tmp_path / "known.pt")
        builder.save(path)

        import torch as th
        from anomaly_head import load_baseline
        baseline = load_baseline(path)
        head     = ZScoreAnomalyHead(baseline, threshold=3.0)
        result   = head.score({
            "fused" : fixed_feat,
            "local" : torch.ones(1, 196, 112) * 0.5,
            "global": torch.ones(1, 196, 768) * 0.5,
        })
        assert result["max_score"] < 1.0, \
            f"Identical features should score near 0, got {result['max_score']:.4f}"
        assert result["is_anomalous"] == False

    def test_known_anomaly_scores_high(self, tmp_path):
        """Features far from baseline should produce high Z-scores."""
        normal_feat = torch.zeros(1, 196, 880)
        defect_feat = torch.ones(1, 196, 880) * 100.0  # very different

        builder = BaselineBuilder("RGB", "Test")
        for _ in range(20):
            builder.update({
                "fused" : normal_feat.clone(),
                "local" : torch.zeros(1, 196, 112),
                "global": torch.zeros(1, 196, 768),
            })
        path = str(tmp_path / "anomaly.pt")
        builder.save(path)

        from anomaly_head import load_baseline
        baseline = load_baseline(path)
        head     = ZScoreAnomalyHead(baseline, threshold=3.0)
        result   = head.score({
            "fused" : defect_feat,
            "local" : torch.ones(1, 196, 112) * 100.0,
            "global": torch.ones(1, 196, 768) * 100.0,
        })
        assert result["max_score"] > 3.0, \
            f"Defect features should score > 3.0, got {result['max_score']:.4f}"
        assert result["is_anomalous"] == True


# ─── LOCALISATION TESTS ───────────────────────────────────────────────────────

class TestLocalisation:

    def test_upsample_shape(self):
        amap   = torch.randn(14, 14)
        result = upsample(amap)
        assert result.shape == (224, 224), f"Expected (224,224), got {result.shape}"

    def test_otsu_mask_shape(self, synthetic_rgb_image):
        mask = otsu_mask(synthetic_rgb_image)
        assert mask.shape == (224, 224)

    def test_otsu_mask_binary(self, synthetic_rgb_image):
        mask   = otsu_mask(synthetic_rgb_image)
        unique = np.unique(mask)
        assert set(unique).issubset({0, 255}), \
            f"Otsu mask should be binary {{0,255}}, got {unique}"

    def test_bboxes_returns_list(self):
        amap  = np.random.uniform(0, 10, (224, 224)).astype(np.float32)
        boxes, binary = get_bboxes(amap, threshold=3.0)
        assert isinstance(boxes, list)
        assert binary.shape == (224, 224)

    def test_no_bbox_below_threshold(self):
        """All-zero map should produce no bounding boxes."""
        amap          = np.zeros((224, 224), dtype=np.float32)
        boxes, binary = get_bboxes(amap, threshold=3.0)
        assert len(boxes) == 0