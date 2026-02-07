import streamlit as st
import numpy as np
from PIL import Image, ImageOps

# -------------------------------------------------
# App config
# -------------------------------------------------
st.set_page_config(
    page_title="SymmetryMatch",
    page_icon="🔥",
    layout="centered"
)

st.title("SymmetryMatch")
st.caption("Structural facial attractiveness model (deploy-safe)")

# -------------------------------------------------
# Heuristic face locking (NO AI LIBS)
# -------------------------------------------------
def lock_face_region(image):
    """
    Heuristically lock onto the face by:
    - converting to grayscale
    - favoring central brightness
    - cropping central-biased region
    """
    gray = ImageOps.grayscale(image)
    arr = np.array(gray).astype(np.float32)

    h, w = arr.shape
    cx, cy = w // 2, h // 2

    # Central weighting mask
    y, x = np.ogrid[:h, :w]
    mask = np.exp(-(((x - cx)**2 + (y - cy)**2) / (0.25 * w * h)))

    weighted = arr * mask

    # Crop central 60%
    crop_w = int(w * 0.6)
    crop_h = int(h * 0.6)
    left = cx - crop_w // 2
    top = cy - crop_h // 2

    face = image.crop((left, top, left + crop_w, top + crop_h))
    return face

# -------------------------------------------------
# Feature extractors (FACE ONLY)
# -------------------------------------------------
def symmetry_error(face):
    gray = ImageOps.grayscale(face)
    img = np.array(gray).astype(np.float32)

    h, w = img.shape
    mid = w // 2

    left = img[:, :mid]
    right = np.fliplr(img[:, w - mid:])

    return np.mean(np.abs(left - right))


def face_ratio(face):
    w, h = face.size
    return w / h

# -------------------------------------------------
# Normalization (realistic ranges)
# -------------------------------------------------
def normalize_symmetry(sym):
    # Face-only symmetry usually ~20–90
    return np.clip(1.2 - sym / 90.0, 0.0, 1.0)


def normalize_ratio(r):
    # Ideal facial ratio ~0.72–0.78
    return np.clip(1.0 - abs(r - 0.75) / 0.35, 0.0, 1.0)

# -------------------------------------------------
# Final attractiveness score
# -------------------------------------------------
def attractiveness_score(face):
    sym = symmetry_error(face)
    ratio = face_ratio(face)

    sym_n = normalize_symmetry(sym)
    ratio_n = normalize_ratio(ratio)

    score = sym_n * 0.65 + ratio_n * 0.35
    return round(score * 100, 1)

# -------------------------------------------------
# Demo database
# -------------------------------------------------
USER_DATABASE = {
    "Alex": 82.0,
    "Jamie": 76.0,
    "Sam": 88.0,
    "Morgan": 84.0,
    "Taylor": 62.0,
    "Jordan": 79.0
}

# -------------------------------------------------
# UI
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a clear, front-facing portrait",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original image", width=300)

    with st.spinner("Locking face region..."):
        face = lock_face_region(image)

    st.image(face, caption="Locked face region", width=250)

    with st.spinner("Analyzing facial structure..."):
        score = attractiveness_score(face)

    st.success(f"Attractiveness score: {score} / 100")

    st.subheader("Matches (similar tier)")
    matches = {
        name: s for name, s in USER_DATABASE.items()
        if abs(s - score) <= 4
    }

    if matches:
        for name, s in matches.items():
            st.write(f"{name} — {s}")
    else:
        st.write("No close matches found.")

st.markdown("---")
st.caption(
    "This model uses heuristic face locking and structural facial metrics. "
    "It reflects common Western beauty standards and is intentionally non-egalitarian."
)
