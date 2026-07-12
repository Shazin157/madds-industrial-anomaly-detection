import argparse
import sys
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

from modality_gate import detect_modality, preprocess_ir, preprocess_rgb
from models.rgb_branch import RGBBranch
from models.ir_branch  import IRBranch
from anomaly_head import BaselineBuilder, get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True)
    p.add_argument("--modality",   choices=["RGB","IR"], default=None)
    p.add_argument("--mode",       default="Industrial")
    p.add_argument("--output",     required=True)
    p.add_argument("--device",     default=None)
    return p.parse_args()


def main():
    args    = parse_args()
    device  = args.device or get_device()
    src     = Path(args.images_dir)
    exts    = {".jpg",".jpeg",".png",".bmp",".tiff",".tif"}
    paths   = sorted([p for p in src.iterdir() if p.suffix.lower() in exts])

    if not paths:
        print(f"No images found in {src}"); sys.exit(1)

    # Detect modality
    modality = args.modality or detect_modality(cv2.imread(str(paths[0])))
    print(f"Modality: {modality} | Device: {device} | Images: {len(paths)}")

    # Load branch
    branch = (RGBBranch() if modality == "RGB" else IRBranch()).eval()
    builder = BaselineBuilder(modality, args.mode)

    for path in tqdm(paths, desc="Building baseline"):
        img = cv2.imread(str(path))
        if img is None:
            continue
        tensor   = (preprocess_rgb(img) if modality == "RGB" else preprocess_ir(img)).to(device)
        features = branch(tensor)
        builder.update(features)

    if builder.n_images == 0:
        print("No images processed."); sys.exit(1)

    builder.save(args.output)


if __name__ == "__main__":
    main()