import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="SymmetryMatch",
    page_icon="💘",
    layout="centered"
)

st.title("💘 SymmetryMatch")
st.caption("Experimental dating app using facial symmetry")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

def extract_landmarks(image_np):
    h, w, _ = image_np.shape
    results = face_mesh.process(image_np)
    if not results.multi_face_landmarks:
        return None, None
    landmarks = [
        (lm.x * w, lm.y * h)
        for lm in results.multi_face_landmarks[0].landmark
    ]
    return landmarks, w

def symmetry_raw_score(landmarks, width):
    mid_x = width / 2
    return np.mean([abs((2 * mid_x - x) - x) for x, _ in landmarks])

def normalize(score):
    return round(max(0, min(100 - score * 100, 100)), 1)

USER_DB = {
    "Alex": 82.3,
    "Jamie": 79.9,
    "Sam": 85.1,
    "Morgan": 83.0,
    "Taylor": 60.4,
    "Jordan": 77.2
}

uploaded = st.file_uploader(
    "Upload a clear, front-facing photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)
    st.image(image, width=300)

    with st.spinner("Analyzing facial symmetry..."):
        landmarks, width = extract_landmarks(img_np)

    if landmarks is None:
        st.error("No face detected. Try another image.")
    else:
        raw = symmetry_raw_score(landmarks, width)
        score = normalize(raw)

        st.success(f"✨ Your Facial Symmetry Score: **{score}/100**")

        st.subheader("💞 Potential Matches")
        matches = {
            name: s for name, s in USER_DB.items()
            if abs(s - score) <= 3
        }

        if matches:
            for name, s in matches.items():
                st.write(f"**{name}** — Symmetry {s}/100")
        else:
            st.write("No close matches yet.")

st.markdown("---")
st.caption(
    "⚠️ This app is experimental. Facial symmetry is only one small factor "
    "in human attraction."
)
