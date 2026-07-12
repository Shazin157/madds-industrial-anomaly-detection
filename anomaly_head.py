import torch
import numpy as np
from pathlib import Path

EPS               = 1e-6
DEFAULT_THRESHOLD = 3.0
W_FUSED, W_LOCAL, W_GLOBAL = 0.5, 0.3, 0.2


class BaselineBuilder:
    def __init__(self, modality: str, mode: str):
        self.modality  = modality
        self.mode      = mode
        self.n_images  = 0
        self._sums     = {}

    def update(self, features: dict):
        for k in ["fused", "local", "global"]:
            v = features[k].squeeze(0).cpu().float()
            if k not in self._sums:
                self._sums[k] = {"s": torch.zeros_like(v), "sq": torch.zeros_like(v)}
            self._sums[k]["s"]  += v
            self._sums[k]["sq"] += v ** 2
        self.n_images += 1

    def save(self, path: str):
        assert self.n_images > 0, "No images accumulated."
        N = float(self.n_images)

        def stats(k):
            mu  = self._sums[k]["s"] / N
            var = torch.clamp(self._sums[k]["sq"] / N - mu**2, min=0)
            return mu, torch.sqrt(var) + EPS

        payload = {"modality": self.modality, "mode": self.mode, "n_images": self.n_images}
        for k in ["fused", "local", "global"]:
            mu, sigma = stats(k)
            payload[f"{k}_mu"]    = mu
            payload[f"{k}_sigma"] = sigma

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        print(f"Baseline saved -> {path}  ({self.n_images} images)")


def load_baseline(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Baseline not found: {p}")
    b = torch.load(p, map_location="cpu", weights_only=False)
    print(f"Loaded baseline: {b['mode']} / {b['modality']} ({b['n_images']} images)")
    return b


class ZScoreAnomalyHead:
    def __init__(self, baseline, threshold=DEFAULT_THRESHOLD, device="cpu"):
        self.threshold = threshold
        self.device    = device
        self.stats     = {
            k: (baseline[f"{k}_mu"].to(device), baseline[f"{k}_sigma"].to(device))
            for k in ["fused", "local", "global"]
        }

    def _zscore(self, x, mu, sigma):
        D    = x.shape[-1]
        z    = (x - mu) / sigma
        return torch.norm(z, dim=-1) / (D ** 0.5)

    @torch.inference_mode()
    def score(self, features: dict) -> dict:
        maps = {}
        for k in ["fused", "local", "global"]:
            x         = features[k].squeeze(0).to(self.device)
            mu, sigma = self.stats[k]
            maps[k]   = self._zscore(x, mu, sigma).reshape(14, 14)

        anomaly_map = W_FUSED * maps["fused"] + W_LOCAL * maps["local"] + W_GLOBAL * maps["global"]
        max_score   = float(anomaly_map.max())

        return {
            "anomaly_map" : anomaly_map,
            "score_fused" : maps["fused"],
            "score_local" : maps["local"],
            "score_global": maps["global"],
            "max_score"   : max_score,
            "mean_score"  : float(anomaly_map.mean()),
            "is_anomalous": max_score > self.threshold,
        }


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"