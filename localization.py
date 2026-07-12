import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

DEFAULT_THRESHOLD = 3.0
HEATMAP_ALPHA     = 0.5
MIN_BBOX_AREA     = 100
SIZE              = 224


@dataclass
class BoundingBox:
    x: int; y: int; w: int; h: int
    max_score: float; area: int


@dataclass
class LocalizationResult:
    anomaly_map_raw:    np.ndarray
    anomaly_map_masked: np.ndarray
    foreground_mask:    np.ndarray
    heatmap_overlay:    np.ndarray
    binary_anomaly:     np.ndarray
    bounding_boxes:     list
    max_score:          float
    is_anomalous:       bool


def upsample(anomaly_map: torch.Tensor) -> np.ndarray:
    t = anomaly_map.unsqueeze(0).unsqueeze(0).float()
    return F.interpolate(t, size=(SIZE,SIZE), mode="bilinear", align_corners=False).squeeze().cpu().numpy()


def otsu_mask(image_bgr: np.ndarray) -> np.ndarray:
    img  = cv2.resize(image_bgr, (SIZE,SIZE))
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5,5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)


def make_heatmap(image_bgr, amap, alpha=HEATMAP_ALPHA):
    img  = cv2.resize(image_bgr, (SIZE,SIZE))
    a, b = amap.min(), amap.max()
    if b - a < 1e-8:
        return img
    norm    = ((amap - a) / (b - a) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return cv2.addWeighted(colored, alpha, img, 1-alpha, 0)


def get_bboxes(amap, threshold=DEFAULT_THRESHOLD):
    binary  = (amap > threshold).astype(np.uint8) * 255
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < MIN_BBOX_AREA:
            continue
        x,y,w,h = cv2.boundingRect(c)
        boxes.append(BoundingBox(x,y,w,h, float(amap[y:y+h,x:x+w].max()), int(cv2.contourArea(c))))
    return sorted(boxes, key=lambda b: b.max_score, reverse=True), binary


def draw_boxes(overlay, boxes, threshold=DEFAULT_THRESHOLD):
    out = overlay.copy()
    for box in boxes:
        color = (0,0,220) if box.max_score > threshold*2 else (0,140,255)
        cv2.rectangle(out, (box.x,box.y), (box.x+box.w,box.y+box.h), color, 2)
        cv2.putText(out, f"Z={box.max_score:.1f}", (box.x+2, max(box.y-4,12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def localize(image_bgr, anomaly_map, threshold=DEFAULT_THRESHOLD) -> LocalizationResult:
    amap_raw    = upsample(anomaly_map)
    mask        = otsu_mask(image_bgr)
    amap_masked = amap_raw * (mask > 0).astype(np.float32)
    heatmap     = make_heatmap(image_bgr, amap_masked)
    boxes, binary = get_bboxes(amap_masked, threshold)
    heatmap     = draw_boxes(heatmap, boxes, threshold)

    return LocalizationResult(
        anomaly_map_raw    = amap_raw,
        anomaly_map_masked = amap_masked,
        foreground_mask    = mask,
        heatmap_overlay    = heatmap,
        binary_anomaly     = binary,
        bounding_boxes     = boxes,
        max_score          = float(amap_masked.max()),
        is_anomalous       = len(boxes) > 0 or float(amap_masked.max()) > threshold,
    )