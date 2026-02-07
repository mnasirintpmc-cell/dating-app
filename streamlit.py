import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# -------------------------------------------------
# Streamlit Config
# -------------------------------------------------
st.set_page_config(
    page_title="SymmetryMatch",
    page_icon="💘",
    layout="centered"
)

st.title("💘 SymmetryMatch")
st.caption("Experimental dating app using facial symmetry")

# -------------------------------------------------
# MediaPipe Setup
# -------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

# -------------------------------------------------
# Utility Functions
# -------------------------------------------------
def extract_landmarks(image_np):
    """Detect facial landmarks and return pixel coordinates"""
    h, w, _ = image_np.shape
    results = face_mesh.process(image_np)

    if not results.multi_face_landmarks:
        return None, None

    landmarks = []
    for lm in results.multi_face_landmarks[0].landmark:
        landmarks.append((lm.x * w, lm.y * h))

    return landmarks, w


def calculate_symmetry_score(landmarks, width):
    """
    Calculate facial symmetry by mirroring x-coordinates
    Lower distance = higher symmetry
    """
    mid_x = width / 2
    diffs = []

    for x, y in landmarks:
        mirrored_x = 2 * mid_x - x
        diffs.append(abs(mirrored_x - x))

    return np.mean(diffs)


def normalize_score(raw_score):
    """
    Convert raw symmetry into a 0–100 attractiveness score
    """
    score = 100 - (raw_score * 100)
    return round(max(0, min(score, 100)), 1)


def find_matches(user_score, database, tolerance=3):
    """Find users with similar symmetry scores"""
    return {
        name: score
        for name, score in database.items()
        if abs(score - user_score) <= tolerance
    }

# -------------------------------------------------
# Fake User Database (replace later with DB)
# -------------------------------------------------
USER_DATABASE = {
    "Alex": 82.4,
    "Jamie": 79.8,
    "Sam": 85.1,
    "Taylor": 60.5,
    "Morgan": 83.2,
    "Jordan": 77.0
}

# -------------------------------------------------
# UI
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a clear, front-facing photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", width=300)

    with st.spinner("Analyzing facial symmetry..."):
        landmarks, width = extract_landmarks(image_np)

    if landmarks is None:
        st.error("❌ No face detected. Try another image.")
    else:
        raw_symmetry = calculate_symmetry_score(landmarks, width)
        final_score = normalize_score(raw_symmetry)

        st.success(f"✨ Your Facial Symmetry Score: **{final_score}/100**")

        # -------------------------------------------------
        # Matching Section
        # -------------------------------------------------
        st.subheader("💞 Potential Matches")

        matches = find_matches(final_score, USER_DATABASE)

        if matches:
            for name, score in matches.items():
                st.write(f"**{name}** — Symmetry Score: {score}/100")
        else:
            st.write("No close matches found yet.")

# -------------------------------------------------
# Footer / Ethics
# -------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ This app is experimental. Facial symmetry is only one small factor "
    "in human attraction and should not be used as a measure of personal worth."
)
