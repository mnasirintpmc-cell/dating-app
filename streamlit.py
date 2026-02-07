import streamlit as st
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(
    page_title="SymmetryMatch",
    page_icon="💘",
    layout="centered"
)

st.title("SymmetryMatch")
st.caption("Experimental dating app using facial symmetry")

def calculate_image_symmetry(image):
    """
    Calculate symmetry by comparing left half of the image
    with the mirrored right half.
    """
    gray = ImageOps.grayscale(image)
    img = np.array(gray).astype(np.float32)

    height, width = img.shape
    mid = width // 2

    left = img[:, :mid]
    right = img[:, width - mid:]
    right_flipped = np.fliplr(right)

    diff = np.abs(left - right_flipped)
    return diff.mean()

def normalize_score(raw_score):
    score = 100 - (raw_score / 2)
    return round(max(0, min(score, 100)), 1)

USER_DATABASE = {
    "Alex": 82.3,
    "Jamie": 79.9,
    "Sam": 85.1,
    "Morgan": 83.0,
    "Taylor": 60.4,
    "Jordan": 77.2
}

uploaded_file = st.file_uploader(
    "Upload a clear, front-facing photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    with st.spinner("Analyzing symmetry..."):
        raw = calculate_image_symmetry(image)
        score = normalize_score(raw)

    st.success(f"Your symmetry score: {score} / 100")

    st.subheader("Potential matches")
    matches = {
        name: s for name, s in USER_DATABASE.items()
        if abs(s - score) <= 3
    }

    if matches:
        for name, s in matches.items():
            st.write(f"{name} — score {s}")
    else:
        st.write("No close matches found.")

st.markdown("---")
st.caption(
    "This app is experimental. Symmetry is only one small factor "
    "in human attraction."
)
