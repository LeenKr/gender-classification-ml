import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Load the trained model
model = load_model("../models/gender_mlp.h5")


IMG_SIZE = (64, 64)

def predict_image(image_path):
    """Reads an image, preprocesses it, predicts gender, and displays result professionally."""
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error loading image.")
        return

    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize & normalize
    img_resized = cv2.resize(img, IMG_SIZE)
    img_resized = img_resized / 255.0
    img_resized = img_resized.reshape(1, -1)

    # Predict
    pred = model.predict(img_resized)[0][0]
    confidence = float(pred) if pred >= 0.5 else 1 - float(pred)
    label = "FEMALE" if pred >= 0.5 else "MALE"

    # Banner color
    color = "hotpink" if label == "FEMALE" else "dodgerblue"

    # Display
    plt.figure(figsize=(7, 7))
    plt.imshow(original)
    plt.axis("off")

    # ⭐ Centered stylish banner
    plt.text(
        0.5, 0.1,
        f"{label}\n({confidence * 100:.2f}%)",
        fontsize=30,
        color="white",
        ha="center",
        va="center",
        bbox=dict(
            facecolor=color,
            alpha=0.85,
            boxstyle="round,pad=1.0"
        )
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    Tk().withdraw()

    print("📂 Please select an image...")
    img_path = askopenfilename(
        title="Choose an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if img_path:
        print("📌 Selected:", img_path)
        predict_image(img_path)
    else:
        print("❌ No image selected.")
