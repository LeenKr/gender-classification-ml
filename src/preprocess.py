import os
import cv2
import numpy as np
from sklearn.utils import shuffle

def load_dataset(dataset_path, img_size=(64, 64)):
    """
    Loads all images from Male/Female folders, preprocesses them,
    and returns X (images) and y (labels).
    
    - Female label = 0
    - Male label = 1
    """

    X = []
    y = []

    # Loop through both classes
    for label, category in enumerate(["Female", "Male"]):
        folder = os.path.join(dataset_path, category)

        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)

            # Load image
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Resize
            img = cv2.resize(img, img_size)

            # Normalize (0–1)
            img = img / 255.0

            # Store image + label
            X.append(img)
            y.append(label)

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Flatten images (MLP needs 1D vectors)
    X = X.reshape(X.shape[0], -1)

    # Shuffle dataset (professional)
    X, y = shuffle(X, y, random_state=42)

    return X, y

def dataset_statistics(y):
    """
    Prints dataset balance (very professional for your doctor).
    """
    total = len(y)
    females = np.sum(y == 0)
    males = np.sum(y == 1)

    print("===== DATASET BALANCE =====")
    print(f"Total samples: {total}")
    print(f"Female: {females} ({(females/total)*100:.2f}%)")
    print(f"Male:   {males} ({(males/total)*100:.2f}%)")
    print("============================")
