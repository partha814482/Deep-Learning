import streamlit as st
import cv2
import tempfile
import numpy as np
import time
from ultralytics import YOLO

st.set_page_config(page_title="YOLOv11 Advanced App", layout="wide")

# ---------------- MODEL PATHS ----------------
MODEL_OPTIONS = {
    "YOLOv11 Detection": r"C:\Users\HP\OneDrive\Documents\data science\YOLO MODEL\yolo11n.pt",
    "YOLOv11 Segmentation": r"C:\Users\HP\OneDrive\Documents\data science\YOLO MODEL\yolo11n-seg.pt",
    "YOLOv11 Pose": r"C:\Users\HP\OneDrive\Documents\data science\YOLO MODEL\yolo11n-pose.pt"
}

# ---------------- UI ----------------
st.title("🚀 YOLOv11 Advanced Streamlit Application")

task = st.sidebar.selectbox(
    "Select a Task",
    [
        "Image Processing",
        "Webcam Detection",
        "Webcam Object Counting",
        "Webcam Tracking",
        "Webcam Zone Counting",
        "Video Processing"
    ]
)

selected_model = st.sidebar.selectbox("Select YOLO Model", list(MODEL_OPTIONS.keys()))
model_path = MODEL_OPTIONS[selected_model]
model = YOLO(model_path)

# ---------------- UTILS ----------------
def draw_zone(frame):
    pts = np.array([[100, 100], [500, 100], [500, 300], [100, 300]], np.int32)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    return pts


# ---------------- IMAGE PROCESSING ----------------
if task == "Image Processing":
    st.header("🖼 Image Processing")

    uploaded = st.file_uploader("Upload an Image", type=["jpg", "png"])

    if uploaded:
        img_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        results = model(img)
        output = results[0].plot()

        st.image(output, channels="BGR")

        # Download
        out_img = "output.jpg"
        cv2.imwrite(out_img, output)
        with open(out_img, "rb") as file:
            st.download_button("📥 Download Output Image", data=file, file_name="YOLO_output.jpg")


# ---------------- WEBCAM DETECTION ----------------
elif task == "Webcam Detection":
    st.header("📷 Webcam Object Detection")

    start = st.button("▶ Start Webcam")
    stop = st.button("⏹ Stop Webcam")

    FRAME = st.image([])

    cap = cv2.VideoCapture(0)

    while start and not stop:
        ret, frame = cap.read()
        if not ret:
            break

        t1 = time.time()
        results = model(frame)
        output = results[0].plot()

        fps = int(1 / (time.time() - t1))
        cv2.putText(output, f"FPS: {fps}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

        FRAME.image(output, channels="BGR")

    cap.release()


# ---------------- OBJECT COUNTING ----------------
elif task == "Webcam Object Counting":
    st.header("🔢 Webcam Object Counting")

    start = st.button("▶ Start Counting")
    stop = st.button("⏹ Stop")

    FRAME = st.image([])

    cap = cv2.VideoCapture(0)

    while start and not stop:
        ret, frame = cap.read()
        if not ret:
            break

        t1 = time.time()
        results = model(frame)

        count = len(results[0].boxes)

        output = results[0].plot()
        cv2.putText(output, f"Count: {count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        fps = int(1 / (time.time() - t1))
        cv2.putText(output, f"FPS: {fps}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        FRAME.image(output, channels="BGR")

    cap.release()


# ---------------- TRACKING ----------------
elif task == "Webcam Tracking":
    st.header("🎯 Object Tracking with YOLOv11 + ByteTrack")

    start = st.button("▶ Start Tracking")
    stop = st.button("⏹ Stop Tracking")

    FRAME = st.image([])

    cap = cv2.VideoCapture(0)

    while start and not stop:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)
        output = results[0].plot()

        FRAME.image(output, channels="BGR")

    cap.release()


# ---------------- ZONE COUNTING ----------------
elif task == "Webcam Zone Counting":
    st.header("🟩 Zone Counting System")

    start = st.button("▶ Start Zone Count")
    stop = st.button("⏹ Stop")

    FRAME = st.image([])

    cap = cv2.VideoCapture(0)

    zone_count = 0

    while start and not stop:
        ret, frame = cap.read()
        if not ret:
            break

        pts = draw_zone(frame)
        results = model(frame)

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = box[:4]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                zone_count += 1

        output = results[0].plot()
        cv2.polylines(output, [pts], True, (0, 255, 0), 2)
        cv2.putText(output, f"Zone Count: {zone_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)

        FRAME.image(output, channels="BGR")

    cap.release()


# ---------------- VIDEO PROCESSING ----------------
elif task == "Video Processing":
    st.header("🎥 Video Processing")

    video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

    if video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video.read())

        cap = cv2.VideoCapture(tfile.name)
        FRAME = st.image([])

        temp_out = "processed_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            output = results[0].plot()

            if out is None:
                h, w, _ = output.shape
                out = cv2.VideoWriter(temp_out, fourcc, 20, (w, h))

            out.write(output)
            FRAME.image(output, channels="BGR")

        cap.release()
        out.release()

        with open(temp_out, "rb") as file:
            st.download_button("📥 Download Processed Video", file, "YOLO_video_output.mp4")
