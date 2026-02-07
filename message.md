# 💘 SymmetryMatch

**SymmetryMatch** is an experimental dating app prototype that uses **facial symmetry analysis** to estimate an attractiveness score and match users with others who have similar symmetry levels.

> ⚠️ This project is for educational and experimental purposes only.  
> Facial symmetry is only one small factor in human attraction and should **not** be interpreted as a measure of personal value.

---

## 🧠 How It Works

1. User uploads a clear, front-facing facial photo
2. The app detects facial landmarks using **MediaPipe Face Mesh**
3. Facial symmetry is calculated by comparing mirrored landmark positions
4. A normalized **symmetry score (0–100)** is generated
5. Users are matched with others having similar scores

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – Web app framework
- **MediaPipe Face Mesh** – Facial landmark detection
- **OpenCV** – Image processing
- **NumPy** – Mathematical operations
- **Pillow** – Image handling

---

## 📁 Project Structure

