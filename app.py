import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image
from pathlib import Path

from modality_gate   import preprocess_uploaded, detect_modality
from models.rgb_branch import RGBBranch
from models.ir_branch  import IRBranch
from anomaly_head    import ZScoreAnomalyHead, load_baseline, get_device
from localization    import localize

st.set_page_config(page_title="Defect Detection", page_icon="🔍", layout="wide")

APP_MODES = ["Industrial", "Agriculture", "Grid", "Custom"]


@st.cache_resource(show_spinner="Loading RGB branch...")
def load_rgb_branch():
    return RGBBranch().eval()


@st.cache_resource(show_spinner="Loading IR branch...")
def load_ir_branch():
    return IRBranch().eval()


@st.cache_data(show_spinner="Loading baseline...")
def load_baseline_cached(path: str):
    try:
        return load_baseline(path)
    except FileNotFoundError:
        return None


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def to_uint8(amap: np.ndarray) -> np.ndarray:
    a, b = amap.min(), amap.max()
    if b - a < 1e-8:
        return np.zeros_like(amap, dtype=np.uint8)
    return ((amap - a) / (b - a) * 255).astype(np.uint8)


def render_sidebar():
    st.sidebar.title("⚙️ Configuration")
    mode = st.sidebar.selectbox("Application Mode", APP_MODES)
    st.sidebar.markdown("---")
    threshold = st.sidebar.slider("Z-score threshold", 1.0, 6.0, 3.0, 0.1,
                                   help="Lower = more sensitive. Higher = more strict.")
    return mode, threshold


def render_results(image_bgr, modality, result, scores, threshold, mode):
    status = "🔴 DEFECT DETECTED" if result.is_anomalous else "🟢 NO DEFECT DETECTED"
    st.markdown(f"### {status} | Mode: **{mode}** | Modality: **{modality}** | Peak Z: **{result.max_score:.2f}** | Threshold: **{threshold:.1f}**")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Original")
        st.image(cv2.cvtColor(cv2.resize(image_bgr,(224,224)), cv2.COLOR_BGR2RGB), use_container_width=True)
    with c2:
        st.subheader("Anomaly Heatmap")
        st.image(cv2.cvtColor(result.heatmap_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("Red = anomalous | Blue = normal | Boxes = defects")
    with c3:
        st.subheader("Binary Map")
        st.image(result.binary_anomaly, use_container_width=True)
        st.caption(f"White = Z > {threshold:.1f}")

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak Z — Blended",  f"{result.max_score:.3f}", "ANOMALOUS" if result.is_anomalous else "normal")
    m2.metric("Peak Z — Fused",    f"{float(scores['score_fused'].max()):.3f}")
    m3.metric("Peak Z — Local",    f"{float(scores['score_local'].max()):.3f}")
    m4.metric("Peak Z — Global",   f"{float(scores['score_global'].max()):.3f}")
    st.markdown("---")

    with st.expander("🔬 Per-stream patch maps (14×14)"):
        s1, s2, s3 = st.columns(3)
        s1.image(to_uint8(scores["score_fused"].cpu().numpy()),  caption="Fused",  use_container_width=True)
        s2.image(to_uint8(scores["score_local"].cpu().numpy()),  caption="Local",  use_container_width=True)
        s3.image(to_uint8(scores["score_global"].cpu().numpy()), caption="Global", use_container_width=True)

    with st.expander("🎭 Foreground mask (Otsu)"):
        st.image(result.foreground_mask, width=224)
        st.caption("White = foreground | Black = suppressed background")

    st.subheader("Detected Regions")
    if not result.bounding_boxes:
        st.info(f"No anomaly clusters found above Z = {threshold:.1f}.")
    else:
        st.write(f"**{len(result.bounding_boxes)}** anomalous region(s) detected:")
        st.table([{
            "Region"   : f"#{i+1}",
            "Severity" : "🔴 Severe" if b.max_score > threshold*2 else "🟠 Moderate",
            "Peak Z"   : f"{b.max_score:.2f}",
            "Position" : f"({b.x}, {b.y})",
            "Size"     : f"{b.w}×{b.h}",
            "Area px²" : b.area,
        } for i, b in enumerate(result.bounding_boxes)])


def main():
    st.title("🔍 Modality-Agnostic Defect Detection System")
    st.markdown("Upload a **Visual (RGB)** or **Thermal (IR)** image. Modality is auto-detected.")
    st.markdown("---")

    mode, threshold = render_sidebar()
    device = get_device()

    st.subheader("Upload Inspection Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png","bmp","tiff"])

    if uploaded_file is None:
        st.info("👆 Upload an image to begin.")
        st.stop()

    image_bgr = pil_to_bgr(Image.open(uploaded_file))
    modality  = detect_modality(image_bgr)
    st.success(f"✅ Modality: **{modality}** ({'Thermal IR' if modality=='IR' else 'Visual RGB'})")

    # Auto-load baseline
    baseline_path = f"baselines/{mode.lower()}_{modality.lower()}.pt"
    baseline      = load_baseline_cached(baseline_path)

    if baseline is None:
        st.warning(f"⚠️ No baseline at `{baseline_path}`. Run `build_baseline.py` first.")
        st.stop()

    with st.expander("ℹ️ Baseline info"):
        st.markdown(f"- **Mode**: {baseline.get('mode')} | **Modality**: {baseline.get('modality')} | **Images**: {baseline.get('n_images')}")

    # Run branch
    with st.spinner(f"Running {modality} branch..."):
        branch = load_rgb_branch() if modality == "RGB" else load_ir_branch()
        tensor, _ = preprocess_uploaded(image_bgr)
        with torch.inference_mode():
            features = branch(tensor.to(device))

    # Score + localize
    with st.spinner("Computing anomaly scores..."):
        scores = ZScoreAnomalyHead(baseline, threshold, device).score(features)

    with st.spinner("Localizing defects..."):
        result = localize(image_bgr, scores["anomaly_map"], threshold)

    render_results(image_bgr, modality, result, scores, threshold, mode)


if __name__ == "__main__":
    main()