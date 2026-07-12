import cv2
import numpy as np
import torch
from torchvision import transforms
from scipy.stats import pearsonr

IR_THRESHOLD = 0.98
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
TARGET_SIZE   = (224, 224)


def detect_modality(image: np.ndarray) -> str:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) image, got shape {image.shape}")
    ch0 = image[:, :, 0].flatten().astype(np.float32)
    ch1 = image[:, :, 1].flatten().astype(np.float32)
    ch2 = image[:, :, 2].flatten().astype(np.float32)
    r01, _ = pearsonr(ch0, ch1)
    r02, _ = pearsonr(ch0, ch2)
    return "IR" if r01 > IR_THRESHOLD and r02 > IR_THRESHOLD else "RGB"


def preprocess_ir(image: np.ndarray) -> torch.Tensor:
    img = cv2.resize(image, TARGET_SIZE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    rgb = cv2.cvtColor(cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2,0,1).float().div(255).unsqueeze(0)


def preprocess_rgb(image: np.ndarray) -> torch.Tensor:
    img = cv2.resize(image, TARGET_SIZE)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])(rgb).unsqueeze(0)


def preprocess_uploaded(image_bgr: np.ndarray) -> tuple[torch.Tensor, str]:
    modality = detect_modality(image_bgr)
    tensor   = preprocess_ir(image_bgr) if modality == "IR" else preprocess_rgb(image_bgr)
    return tensor, modality