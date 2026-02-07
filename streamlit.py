import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import mediapipe as mp

# -------------------------------------------------
# App config
# -------------------------------------------------
st.set_page_config(
    page_title="SymmetryMatch",
    page_icon="🔥",
    layout="centered"
)

st.title("SymmetryMatch")
st.caption("AI-locked facial attractiveness model (structural)")

# -------------------------------------------------
# MediaPipe face detector (AI)
# -------------------------------------------------
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6
)

# -------------------------------------------------
# Face detection + crop
# -------------------------------------------------
def detect_and_crop_face(image):
    img_np = np.array(image)
    h, w, _ = img_np.shape

    results = face_detector.process(img_np)
    if not results.detections:
        return None

    # Take the most confident face
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box

    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)

    # Expand slightly to include jaw & forehead
    pad = int(0.15 * bw)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x1 + bw + pad)
    y2 = min(h, y1 + bh + pad)

    face = image.crop((x1, y1, x2, y2))
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
# Normalization (soft, realistic)
# -------------------------------------------------
def normalize_symmetry(sym):
    # Typical face-only range: ~15–70
    return np.clip(1.3 - sym / 60.0, 0.0, 1.0)


def normalize_ratio(r):
    # Ideal male face ~0.72–0.78
    return np.clip(1.0 - abs(r - 0.75) / 0.30, 0.0, 1.0)

# -------------------------------------------------
# Final attractiveness score
# -------------------------------------------------
def attractiveness_score(face):
    sym = symmetry_error(face)
    ratio = face_ratio(face)

    sym_n = normalize_symmetry(sym)
    ratio_n = normalize_ratio(ratio)

    score = (
        sym_n * 0.65 +
        ratio_n * 0.35
    )

    return round(score * 100, 1)

# -------------------------------------------------
# Demo matching pool
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
    "Upload a clear, front-facing face photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original", width=300)

    with st.spinner("Detecting face..."):
        face = detect_and_crop_face(image)

    if face is None:
        st.error("No face detected. Use a clear, front-facing photo.")
    else:
        st.image(face, caption="AI-detected face", width=250)

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
    "This model uses AI face detection and structural facial metrics. "
    "It reflects common Western beauty standards and is intentionally non-egalitarian."
)
