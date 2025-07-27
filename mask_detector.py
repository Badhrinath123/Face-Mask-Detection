import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the trained model
model = load_model("face_mask_model.h5")

# Load OpenCV's Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def test_image(img_path):
    """Predict mask presence on a static image and display the result."""
    try:
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        label = "Mask" if prediction >= 0.5 else "No Mask"
        confidence = f"{prediction * 100:.1f}%"

        # Load original image to display
        orig = cv2.imread(img_path)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        cv2.putText(orig, f"{label} ({confidence})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label == "Mask" else (0, 0, 255), 2)

        plt.imshow(orig)
        plt.title(f"{label} - Confidence: {confidence}")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Error processing image: {e}")


# Test images
print("Testing image with mask:")
test_image("test_images/download.jpg")

print("Testing image without mask:")
test_image("test_images/images.jpg")


# Webcam detection
cap = cv2.VideoCapture(0)
print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        try:
            face = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
            face_array = img_to_array(face_resized) / 255.0
            face_array = np.expand_dims(face_array, axis=0)

            prediction = model.predict(face_array)[0][0]
            label = "Mask" if prediction >= 0.5 else "No Mask"
            confidence = f"{prediction * 100:.1f}%"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} ({confidence})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except Exception as e:
            print(f"Face processing error: {e}")
            continue

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
