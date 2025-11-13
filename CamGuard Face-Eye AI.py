import cv2

# Load Haar cascades
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from webcam!")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 255), 2)

        # Region of interest for eyes
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes inside the face region
        eyes = eye_classifier.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(25, 25))

        if len(eyes) >= 2:
            # Sort eyes by x position (left to right)
            eyes = sorted(eyes, key=lambda ex: ex[0])

            for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):  # only consider two main eyes
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                label = "Left Eye" if i == 0 else "Right Eye"
                cv2.putText(roi_color, label, (ex, ey - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Webcam - Face and Eye Detection", frame)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()