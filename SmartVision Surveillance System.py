import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="AI Detection App", layout="wide")

st.title("🚀 Computer Vision Detection System")
st.write("Select a module from the left sidebar")

# ---------------------- 1. WALKING PEOPLE DETECTION --------------------------
def walking_people_detection(video_file):
    body_classifier = cv2.CascadeClassifier(
        r"C:\Users\HP\OneDrive\Documents\SENAPATI SIR FSDS NOTE\NOVEMBER MONTH DS NOTE\12th,  - Intro to cv2\12th,  - Intro to cv2\opencv\Haarcascades\haarcascade_fullbody.xml"
    )

    if video_file is None:
        st.warning("Please upload a video!")
        return

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        stframe.image(frame, channels="BGR")

    cap.release()

# ---------------------- 2. CAR DETECTION --------------------------
def car_detection(video_file):
    car_classifier = cv2.CascadeClassifier(
        r"C:\Users\HP\OneDrive\Documents\SENAPATI SIR FSDS NOTE\NOVEMBER MONTH DS NOTE\12th,  - Intro to cv2\12th,  - Intro to cv2\opencv\Haarcascades\haarcascade_car.xml"
    )

    if car_classifier.empty():
        st.error("Car Haarcascade NOT LOADED — check your path!")
        return

    if video_file is None:
        st.warning("Upload a car video file!")
        return

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_classifier.detectMultiScale(gray, 1.1, 3)

        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        stframe.image(frame, channels="BGR")

    cap.release()

# ---------------------- 3. FACE + EYE DETECTION (Webcam) ---------------------
class FaceEyeDetector(VideoTransformerBase):
    def __init__(self):
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_classifier.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 255), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            eyes = self.eye_classifier.detectMultiScale(roi_gray, 1.1, 10)

            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda e: e[0])
                for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):
                    label = "Left Eye" if i == 0 else "Right Eye"
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    cv2.putText(roi_color, label, (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img

# ---------------------- FRONTEND MENU ---------------------------
menu = st.sidebar.selectbox(
    "Choose Detection Type",
    ["Walking People Detection", "Car Detection", "Face + Eye Detection (Webcam)"]
)

# ---------------------- MAIN UI ---------------------------
import tempfile

if menu == "Walking People Detection":
    st.header("🏃 Walking People Detection")
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if st.button("Start Detection"):
        walking_people_detection(video_file)

elif menu == "Car Detection":
    st.header("🚗 Car Detection")
    video_file = st.file_uploader("Upload Car Video", type=["mp4", "avi", "mov"])
    if st.button("Start Detection"):
        car_detection(video_file)

elif menu == "Face + Eye Detection (Webcam)":
    st.header("😊 Face + Eye Detection")
    webrtc_streamer(key="face-eye", video_transformer_factory=FaceEyeDetector)
