"""
test_evaluation.py
Evaluates anomaly detection performance on MVTec metal_nut.
Computes AUROC, inference time, and memory usage.

Run: python test_evaluation.py
Requires: baselines/industrial_rgb.pt + data/rgb/test/
"""

import torch
import numpy as np
import cv2
import time
import tracemalloc
from pathlib import Path
from sklearn.metrics import roc_auc_score

from modality_gate   import preprocess_uploaded, detect_modality
from models.rgb_branch import RGBBranch
from models.ir_branch  import IRBranch
from anomaly_head    import ZScoreAnomalyHead, load_baseline, get_device
from localization    import localize


# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASELINE_PATH  = "baselines/industrial_rgb.pt"
TEST_GOOD_DIR  = "data/rgb/test/good"
TEST_DEFECT_DIRS = {
    "bent"   : "data/rgb/test/bent",
    "color"  : "data/rgb/test/color",
    "flip"   : "data/rgb/test/flip",
    "scratch": "data/rgb/test/scratch",
    "good"   : "data/rgb/test/good"
}
THRESHOLD      = 1.30
EXTS           = {".jpg", ".jpeg", ".png", ".bmp"}


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def load_images(directory):
    p = Path(directory)
    return sorted([f for f in p.iterdir() if f.suffix.lower() in EXTS])


def run_inference(image_bgr, branch, head, device):
    tensor, modality = preprocess_uploaded(image_bgr)
    tensor = tensor.to(device)
    with torch.inference_mode():
        features = branch(tensor)
    scores = head.score(features)
    return scores["max_score"], scores["is_anomalous"]


# ─── 1. AUROC EVALUATION ─────────────────────────────────────────────────────

def evaluate_auroc():
    print("\n" + "="*60)
    print("  AUROC EVALUATION")
    print("="*60)

    device   = get_device()
    baseline = load_baseline(BASELINE_PATH)
    branch   = RGBBranch().eval()
    head     = ZScoreAnomalyHead(baseline, threshold=THRESHOLD, device=device)

    scores_all = []
    labels_all = []

    # Normal images → label 0
    normal_paths = load_images(TEST_GOOD_DIR)
    print(f"\nScoring {len(normal_paths)} normal images...")
    for path in normal_paths:
        img = cv2.imread(str(path))
        if img is None: continue
        score, _ = run_inference(img, branch, head, device)
        scores_all.append(score)
        labels_all.append(0)

    # Defect images → label 1
    for defect_type, directory in TEST_DEFECT_DIRS.items():
        paths = load_images(directory)
        print(f"Scoring {len(paths)} '{defect_type}' images...")
        for path in paths:
            img = cv2.imread(str(path))
            if img is None: continue
            score, _ = run_inference(img, branch, head, device)
            scores_all.append(score)
            labels_all.append(1)

    # Compute AUROC
    auroc = roc_auc_score(labels_all, scores_all)

    # Compute accuracy at threshold
    preds    = [1 if s > THRESHOLD else 0 for s in scores_all]
    correct  = sum(p == l for p, l in zip(preds, labels_all))
    accuracy = correct / len(labels_all) * 100

    # Per-class breakdown
    n_normal  = labels_all.count(0)
    n_defect  = labels_all.count(1)
    tp = sum(p==1 and l==1 for p,l in zip(preds,labels_all))
    tn = sum(p==0 and l==0 for p,l in zip(preds,labels_all))
    fp = sum(p==1 and l==0 for p,l in zip(preds,labels_all))
    fn = sum(p==0 and l==1 for p,l in zip(preds,labels_all))

    print(f"\n  AUROC              : {auroc:.4f}")
    print(f"  Accuracy           : {accuracy:.1f}%")
    print(f"  True Positives     : {tp} / {n_defect}")
    print(f"  True Negatives     : {tn} / {n_normal}")
    print(f"  False Positives    : {fp}")
    print(f"  False Negatives    : {fn}")
    print(f"  Threshold used     : {THRESHOLD}")

    return auroc, accuracy


# ─── 2. INFERENCE TIME TEST ───────────────────────────────────────────────────

def evaluate_inference_time(n_runs=20):
    print("\n" + "="*60)
    print("  INFERENCE TIME BENCHMARK")
    print("="*60)

    device   = get_device()
    baseline = load_baseline(BASELINE_PATH)
    branch   = RGBBranch().eval()
    head     = ZScoreAnomalyHead(baseline, threshold=THRESHOLD, device=device)

    # Warm up
    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    run_inference(dummy, branch, head, device)

    # Benchmark
    times = []
    for _ in range(n_runs):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        t0  = time.perf_counter()
        run_inference(img, branch, head, device)
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # ms
    print(f"\n  Device             : {device.upper()}")
    print(f"  Runs               : {n_runs}")
    print(f"  Mean latency       : {times.mean():.1f} ms")
    print(f"  Std deviation      : {times.std():.1f} ms")
    print(f"  Min / Max          : {times.min():.1f} / {times.max():.1f} ms")
    print(f"  Throughput         : {1000/times.mean():.2f} images/sec")

    return times.mean(), times.std()


# ─── 3. MODALITY GATE ACCURACY ───────────────────────────────────────────────

def evaluate_modality_gate():
    print("\n" + "="*60)
    print("  MODALITY GATE ACCURACY")
    print("="*60)

    from rgb_to_synthetic_ir import to_ir

    rgb_paths = load_images("data/rgb/train/good")[:50]
    correct_rgb = correct_ir = total = 0

    for path in rgb_paths:
        img = cv2.imread(str(path))
        if img is None: continue

        # RGB image should classify as RGB
        if detect_modality(img) == "RGB":
            correct_rgb += 1

        # Synthetic IR version should classify as IR
        ir_img = to_ir(img)
        if detect_modality(ir_img) == "IR":
            correct_ir += 1

        total += 1

    print(f"\n  Images tested      : {total}")
    print(f"  RGB accuracy       : {correct_rgb}/{total}  ({correct_rgb/total*100:.1f}%)")
    print(f"  IR  accuracy       : {correct_ir}/{total}  ({correct_ir/total*100:.1f}%)")
    print(f"  Overall gate acc   : {(correct_rgb+correct_ir)/(total*2)*100:.1f}%")

    return correct_rgb / total, correct_ir / total


# ─── 4. MEMORY USAGE TEST ────────────────────────────────────────────────────

def evaluate_memory():
    print("\n" + "="*60)
    print("  MEMORY USAGE")
    print("="*60)

    tracemalloc.start()

    device   = get_device()
    baseline = load_baseline(BASELINE_PATH)
    branch   = RGBBranch().eval()
    head     = ZScoreAnomalyHead(baseline, threshold=THRESHOLD, device=device)

    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    run_inference(img, branch, head, device)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n  Peak memory usage  : {peak / 1024 / 1024:.1f} MB")
    print(f"  Current memory     : {current / 1024 / 1024:.1f} MB")

    return peak / 1024 / 1024


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nDefect Detection System — Evaluation Suite")
    print("=" * 60)

    auroc, acc     = evaluate_auroc()
    mean_t, std_t  = evaluate_inference_time()
    rgb_acc, ir_acc = evaluate_modality_gate()
    peak_mem       = evaluate_memory()

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  AUROC              : {auroc:.4f}")
    print(f"  Classification Acc : {acc:.1f}%")
    print(f"  Inference Time     : {mean_t:.1f} ± {std_t:.1f} ms")
    print(f"  Modality Gate Acc  : RGB={rgb_acc*100:.1f}%  IR={ir_acc*100:.1f}%")
    print(f"  Peak Memory        : {peak_mem:.1f} MB")
    print("="*60)