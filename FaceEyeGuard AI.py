import cv2
import numpy as np

# Load Haar cascades for face and eyes
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load the image
image = cv2.imread(r"C:\Users\HP\OneDrive\Pictures\gemni\gembooth (1).jpg")

# Check if the image loaded correctly
if image is None:
    print("Error: Image not found or cannot be loaded!")
    exit()

# Resize for better visualization (optional)
image = cv2.resize(image, (800, 600))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

if len(faces) == 0:
    print("No faces found!")
else:
    print(f"Detected {len(faces)} face(s).")

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)

        # Define region of interest (ROI) for the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

        # Detect eyes within the face ROI
        eyes = eye_classifier.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

        if len(eyes) >= 2:
            # Sort eyes based on x-coordinate (left to right)
            eyes = sorted(eyes, key=lambda ex: ex[0])

            for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):  # Only label two main eyes
                # Draw rectangle
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                # Label left and right eyes
                eye_label = "Left Eye" if i == 0 else "Right Eye"
                cv2.putText(roi_color, eye_label, (ex, ey - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            print("Eyes not clearly detected in this face region.")

# Show the result
cv2.imshow("Face with Left and Right Eye Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()