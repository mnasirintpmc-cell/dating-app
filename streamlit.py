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
st.caption("Experimental facial attractiveness model based on structure")

# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
def preprocess_face(image):
    """
    Center-crop to reduce background and hair influence.
    """
    w, h = image.size
    size = int(min(w, h) * 0.85)
    left = (w - size) // 2
    top = (h - size) // 2
    return image.crop((left, top, left + size, top + size))


# -------------------------------------------------
# Feature extractors
# -------------------------------------------------
def symmetry_score(image):
    gray = ImageOps.grayscale(image)
    img = np.array(gray).astype(np.float32)

    h, w = img.shape
    mid = w // 2

    left = img[:, :mid]
    right = np.fliplr(img[:, w - mid:])

    return np.mean(np.abs(left - right))


def face_ratio_score(image):
    """
    Proxy for facial width-to-height ratio.
    Western attractiveness peak ~0.72–0.78
    """
    w, h = image.size
    ratio = w / h
    return max(0.0, 1.0 - abs(ratio - 0.75) / 0.25)


# -------------------------------------------------
# Final attractiveness model (HARSH, STRUCTURAL)
# -------------------------------------------------
def attractiveness_score(image):
    image = preprocess_face(image)

    sym = symmetry_score(image)
    ratio = face_ratio_score(image)

    # Normalize symmetry aggressively
    sym_n = max(0.0, 1.0 - sym / 28.0)

    # Weighted like human judgment
    score = (
        sym_n * 0.65 +
        ratio * 0.35
    )

    return round(score * 100, 1)


# -------------------------------------------------
# Demo matching pool
# -------------------------------------------------
USER_DATABASE = {
    "Alex": 81.0,
    "Jamie": 77.5,
    "Sam": 88.0,
    "Morgan": 84.2,
    "Taylor": 63.1,
    "Jordan": 79.3
}

# -------------------------------------------------
# UI
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a clear, front-facing face photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    with st.spinner("Analyzing facial structure..."):
        score = attractiveness_score(image)

    st.success(f"Attractiveness score: {score} / 100")

    st.subheader("Matches (similar tier)")
    matches = {
        name: s for name, s in USER_DATABASE.items()
        if abs(s - score) <= 4
    }

    if matches:
        for name, s in matches.items():
            st.write(f"{name} - {s}")
    else:
        st.write("No close matches found.")

st.markdown("---")
st.caption(
    "This model reflects common Western facial structure standards. "
    "It is experimental and intentionally non-egalitarian."
)
