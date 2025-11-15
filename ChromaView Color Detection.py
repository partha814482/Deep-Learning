import cv2
import numpy as np
import streamlit as st

st.title("🎨 Real-Time Color Detection Using OpenCV + Streamlit")

st.sidebar.header("Select Color to Detect")

option = st.sidebar.selectbox(
    "Choose a color",
    ("Red", "Blue", "Green", "Yellow", "All Except White")
)

st.sidebar.write("Press **Ctrl+C** in terminal to stop camera")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

# Color ranges
color_ranges = {
    "Red": (np.array([161, 155, 84]), np.array([179, 255, 255])),
    "Blue": (np.array([94, 80, 2]), np.array([126, 255, 255])),
    "Green": (np.array([40, 100, 100]), np.array([102, 255, 255])),
    "Yellow": (np.array([20, 100, 100]), np.array([30, 255, 255])),
    "All Except White": (np.array([0, 42, 0]), np.array([179, 255, 255]))
}

low, high = color_ranges[option]

while True:
    ret, frame = cap.read()
    if not ret:
        st.write("Camera not detected!")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, low, high)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    FRAME_WINDOW.image(result, channels="BGR")

