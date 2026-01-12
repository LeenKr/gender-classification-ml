import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("../models/gender_mlp.h5")

IMG_SIZE = (64, 64)

def preprocess(frame):
    resized = cv2.resize(frame, IMG_SIZE)
    normalized = resized / 255.0
    flattened = normalized.reshape(1, -1)
    return flattened

# Open webcam
cap = cv2.VideoCapture(0)

plt.ion()  # interactive mode
fig, ax = plt.subplots(figsize=(6, 6))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Predict
    processed = preprocess(frame)
    pred = model.predict(processed)[0][0]

    label = "FEMALE" if pred >= 0.5 else "MALE"
    confidence = pred if pred >= 0.5 else 1 - pred

    # Display in matplotlib window
    ax.clear()
    ax.imshow(rgb_frame)
    ax.axis("off")
    ax.set_title(f"{label} ({confidence*100:.1f}%)", fontsize=20, color="blue")

    plt.pause(0.001)

cap.release()
plt.close()
