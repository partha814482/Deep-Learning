import cv2
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk

# Load Haar cascades
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = None  # webcam variable

def start_camera():
    global cap
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access webcam!")
        return

    update_frame()


def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None
        video_label.config(image="")  # clear the box


def update_frame():
    global cap
    if cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 255), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_classifier.detectMultiScale(roi_gray, 1.1, 10, minSize=(25, 25))

        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda ex: ex[0])  # left → right sorting
            for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                label = "Left Eye" if i == 0 else "Right Eye"
                cv2.putText(roi_color, label, (ex, ey - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Convert image for Tkinter
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    video_label.after(10, update_frame)


# GUI WINDOW
root = tk.Tk()
root.title("Face & Eye Detection System")
root.geometry("900x700")
root.configure(bg="#1b1b1b")

title = Label(root, text="Face & Eye Detection (CNN + Haar Cascade)", 
              font=("Arial", 22, "bold"), bg="#1b1b1b", fg="white")
title.pack(pady=20)

# Video display area
video_label = Label(root, bg="black")
video_label.pack(padx=20, pady=20)

# Buttons
button_frame = Frame(root, bg="#1b1b1b")
button_frame.pack(pady=20)

start_btn = Button(button_frame, text="Start Camera", command=start_camera,
                   font=("Arial", 14), bg="#2ecc71", fg="white", padx=20, pady=10)
start_btn.grid(row=0, column=0, padx=15)

stop_btn = Button(button_frame, text="Stop Camera", command=stop_camera,
                  font=("Arial", 14), bg="#e74c3c", fg="white", padx=20, pady=10)
stop_btn.grid(row=0, column=1, padx=15)

exit_btn = Button(button_frame, text="Exit", command=root.destroy,
                  font=("Arial", 14), bg="#3498db", fg="white", padx=20, pady=10)
exit_btn.grid(row=0, column=2, padx=15)

root.mainloop()
