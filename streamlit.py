import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter

st.set_page_config(
    page_title="SymmetryMatch",
    page_icon="🔥",
    layout="centered"
)

st.title("SymmetryMatch")
st.caption("Experimental facial attractiveness model (non-egalitarian)")

# -----------------------------
# Core feature extractors
# -----------------------------

def image_symmetry(image):
    gray = ImageOps.grayscale(image)
    img = np.array(gray).astype(np.float32)

    h, w = img.shape
    mid = w // 2

    left = img[:, :mid]
    right = img[:, w - mid:]
    right = np.fliplr(right)

    return np.mean(np.abs(left - right))


def face_ratio(image):
    """
    Proxy for facial width-to-height ratio.
    Wider / compact faces score higher.
    """
    gray = ImageOps.grayscale(image)
    img = np.array(gray)

    h, w = img.shape
    return w / h


def jawline_strength(image):
    """
    Edge density in lower third of face.
    Strong jawlines produce strong edges.
    """
    gray = ImageOps.grayscale(image)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.array(edges).astype(np.float32)

    h, _ = arr.shape
    jaw_region = arr[int(h * 0.65):h, :]

    return np.mean(jaw_region)


def contrast_score(image):
    """
    High contrast faces tend to be rated more attractive.
    """
    gray = ImageOps.grayscale(image)
    arr = np.array(gray).astype(np.float32)
    return np.std(arr)


# -----------------------------
# Scoring logic (HARSH)
# -----------------------------

def normalize(val, min_v, max_v):
    return max(0, min(1, (val - min_v) / (max_v - min_v)))


def attractiveness_score(image):
    sym = image_symmetry(image)
    ratio = face_ratio(image)
    jaw = jawline_strength(image)
    contrast = contrast_score(image)

    # Normalize aggressively (empirical ranges)
    sym_n = 1 - normalize(sym, 5, 40)
    ratio_n = normalize(ratio, 0.6, 0.85)
    jaw_n = normalize(jaw, 8, 35)
    contrast_n = normalize(contrast, 40, 80)

    # Weighted like humans actually judge
    score = (
        sym_n * 0.30 +
        jaw_n * 0.30 +
        ratio_n * 0.25 +
        contrast_n * 0.15
    )

    return round(score * 100, 1)


# -----------------------------
# Demo matching pool
# -----------------------------

USER_DB = {
    "Alex": 81.0,
    "Jamie": 76.5,
    "Sam": 88.2,
    "Morgan": 83.7,
    "Taylor": 62.4,
    "Jordan": 79.1
}

# -----------------------------
# UI
# -----------------------------

uploaded = st.file_uploader(
    "Upload a clear, front-facing face (neutral expression)",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, width=300)

    with st.spinner("Analyzing facial structure..."):
        score = attractiveness_score(image)

    st.success(f"Attractiveness score: **{score} / 100**")

    st.subheader("Matches (similar tier)")
    matches = {
        name: s for name, s in USER_DB.items()
        if abs(s - score) <= 4
    }

    if matches:
        for name, s in matches.items():
            st.write(f"{name} — {s}")
    else:
        st.write("No close matches found.")

st.markdown("---")
st.caption(
    "This model reflects common Western beauty standards "
    "and is intentionally not egalitarian."
)
