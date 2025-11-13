import cv2
import matplotlib.pyplot as plt

# Use OpenCV's built-in human face detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load image
image = cv2.imread(r"C:\Users\HP\OneDrive\Pictures\gemni\download.png")

if image is None:
    print("Error: Image not found or cannot be loaded!")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Try more sensitive settings
faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

if len(faces) == 0:
    print("No faces found!")
else:
    print(f"Found {len(faces)} face(s)")
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title('Detected Face(s)')
    plt.show()
