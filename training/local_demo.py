"""
Run YOUR trained model on the webcam locally (no browser).
Run after train.py:  python local_demo.py   (press 'q' to quit)
"""
import json
import cv2
import numpy as np
from tensorflow import keras

model = keras.models.load_model("emotion_model.h5")
labels = json.load(open("labels.json"))
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y:y + h, x:x + w], (48, 48)).astype("float32") / 255.0
        probs = model.predict(face[None, :, :, None], verbose=0)[0]
        i = int(np.argmax(probs))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(frame, f"{labels[i]} {probs[i]*100:.0f}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

    cv2.imshow("Emotion detection (your model)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
