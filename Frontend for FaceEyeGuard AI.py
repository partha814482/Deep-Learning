import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Load Haar cascades
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

current_img = None


def upload_image():
    global current_img

    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )

    if not file_path:
        return

    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Error", "Could not load image!")
        return

    img = cv2.resize(img, (550, 350))  # smaller so it fits screen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 255), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_classifier.detectMultiScale(roi_gray, 1.1, 8)
        eyes = sorted(eyes, key=lambda e: e[0])

        for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            label = "Left Eye" if i == 0 else "Right Eye"
            cv2.putText(roi_color, label, (ex, ey - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    current_img = ImageTk.PhotoImage(Image.fromarray(img_rgb))

    img_label.config(image=current_img)
    img_label.image = current_img


# ----------- GUI -----------
root = tk.Tk()
root.title("Face & Eye Detection")
root.geometry("650x550")
root.configure(bg="#1b1b1b")
root.resizable(False, False)

# Title
title = tk.Label(root, text="Upload Image for Face & Eye Detection",
                 font=("Arial", 16, "bold"), bg="#1b1b1b", fg="white")
title.pack(pady=8)

# Small button row directly below title — ALWAYS VISIBLE
button_frame = tk.Frame(root, bg="#1b1b1b")
button_frame.pack(pady=5)

upload_btn = tk.Button(button_frame, text="Upload Image", command=upload_image,
                       font=("Arial", 12), bg="#3498db", fg="white",
                       padx=15, pady=5)
upload_btn.grid(row=0, column=0, padx=10)

exit_btn = tk.Button(button_frame, text="Exit", command=root.destroy,
                     font=("Arial", 12), bg="#e74c3c", fg="white",
                     padx=15, pady=5)
exit_btn.grid(row=0, column=1, padx=10)

# Image display area (below buttons)
img_label = tk.Label(root, bg="black", width=550, height=350)
img_label.pack(pady=10)

root.mainloop()
