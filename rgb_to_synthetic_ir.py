import cv2
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from tqdm import tqdm

RGB_SRC    = "data/rgb/test/flip"
IR_DST     = "data/ir/test/flip"
NOISE_STD  = 1.5
EXTS       = {".jpg",".jpeg",".png",".bmp",".tiff",".tif"}


def to_ir(image_bgr: np.ndarray, noise_std=NOISE_STD) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    base = gray.astype(np.float32)
    h, w = base.shape
    ch   = [np.clip(base + np.random.normal(0, noise_std, (h,w)), 0, 255).astype(np.uint8)
            for _ in range(3)]
    return cv2.merge([ch[2], ch[1], ch[0]])   # BGR order


def verify(image: np.ndarray) -> bool:
    chs  = [image[:,:,i].flatten().astype(np.float32) for i in range(3)]
    r01, _ = pearsonr(chs[0], chs[1])
    r02, _ = pearsonr(chs[0], chs[2])
    return r01 > 0.98 and r02 > 0.98


def main():
    src = Path(RGB_SRC)
    dst = Path(IR_DST)
    dst.mkdir(parents=True, exist_ok=True)

    paths = sorted([p for p in src.iterdir() if p.suffix.lower() in EXTS])
    if not paths:
        raise ValueError(f"No images found in {RGB_SRC}")

    print(f"Converting {len(paths)} images: {RGB_SRC} -> {IR_DST}")

    ok = fail = 0
    for i, p in enumerate(tqdm(paths)):
        img = cv2.imread(str(p))
        if img is None:
            fail += 1; continue
        ir = to_ir(img)
        if i == 0:
            print(f"  Verification: {'PASS ✓' if verify(ir) else 'FAIL ✗ — reduce NOISE_STD'}")
            print(f"  Output shape: {ir.shape} dtype: {ir.dtype}")
        cv2.imwrite(str(dst / f"ir_{p.stem}.png"), ir)
        ok += 1

    print(f"Done — Converted: {ok} | Failed: {fail}")
    print(f"Next: python build_baseline.py --images_dir {IR_DST} --modality IR --mode Industrial --output baselines/industrial_ir.pt")


if __name__ == "__main__":
    main()