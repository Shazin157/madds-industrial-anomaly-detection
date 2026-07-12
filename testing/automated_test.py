import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime

# --- IMPORT YOUR ACTUAL MODULES ---
from modality_gate import detect_modality, preprocess_ir, preprocess_rgb
from models.rgb_branch import RGBBranch
from models.ir_branch import IRBranch
from anomaly_head import ZScoreAnomalyHead, load_baseline, get_device
from localization import localize

class MVTecEvaluator:
    def __init__(self, dataset_root, baseline_dir, device=None):
        self.root = dataset_root 
        self.baseline_dir = baseline_dir
        self.device = device or get_device()
        
        # Cache models
        self.models = {
            "RGB": RGBBranch().to(self.device).eval(),
            "IR": IRBranch().to(self.device).eval()
        }
        
        # Load Baselines (adjust filenames to match your saved files)
        self.baselines = {
            "RGB": load_baseline(os.path.join(baseline_dir, "industrial_rgb.pt")),
            "IR": load_baseline(os.path.join(baseline_dir, "industrial_ir.pt"))
        }

    def run_full_evaluation(self, output_dir="test_rgb_results"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_dir = f"{output_dir}_{timestamp}"
        os.makedirs(run_dir, exist_ok=True)
        
        results = []
        defect_types = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        
        for defect in defect_types:
            folder_path = os.path.join(self.root, defect)
            print(f"\n📂 Processing: {defect.upper()}")
            
            for img_name in tqdm(os.listdir(folder_path)):
                img_path = os.path.join(folder_path, img_name)
                img_bgr = cv2.imread(img_path)
                if img_bgr is None: continue
                
                # 1. Pipeline Execution
                modality = detect_modality(img_bgr)
                tensor = preprocess_ir(img_bgr) if modality == "IR" else preprocess_rgb(img_bgr)
                
                with torch.inference_mode():
                    features = self.models[modality](tensor.to(self.device))
                    head = ZScoreAnomalyHead(self.baselines[modality], device=self.device)
                    scores = head.score(features)
                    res = localize(img_bgr, scores["anomaly_map"])

                # 2. Log Data
                results.append({
                    "defect_type": defect,
                    "is_anomaly": 0 if defect == "good" else 1,
                    "max_z": res.max_score,
                    "modality": modality
                })

        self.generate_performance_report(results, run_dir)

    def generate_performance_report(self, results, run_dir):
        df = pd.DataFrame(results)
        y_true = df['is_anomaly']
        y_scores = df['max_z']
        
        # Calculate AUROC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # --- PLOT 1: AUROC CURVE ---
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (AUROC)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(f"{run_dir}/auroc_curve.png", dpi=300)
        
        # --- PLOT 2: Z-SCORE DISTRIBUTION ---
        plt.figure(figsize=(10, 5))
        for label in df['defect_type'].unique():
            subset = df[df['defect_type'] == label]
            plt.hist(subset['max_z'], bins=20, alpha=0.5, label=label)
        plt.axvline(x=1.70, color='r', linestyle='--', label='Threshold (Z=1.30)')
        plt.xlabel('Peak Z-Score')
        plt.ylabel('Frequency')
        plt.title('Z-Score Distribution by Defect Type')
        plt.legend()
        plt.savefig(f"{run_dir}/zscore_distribution.png", dpi=300)

        print("\n" + "="*40)
        print(f"📊 AUROC Score: {roc_auc:.4f}")
        print(f"📈 Plots saved to: {run_dir}/")
        print("="*40)

if __name__ == "__main__":
    evaluator = MVTecEvaluator(
        dataset_root="data/rgb/test",
        baseline_dir="baselines"
    )
    evaluator.run_full_evaluation()