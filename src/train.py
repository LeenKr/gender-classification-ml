import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from preprocess import load_dataset, dataset_statistics
from model import create_mlp

def main():
    dataset_path = "../dataset"
    
    # Load dataset
    X, y = load_dataset(dataset_path)

    # Show dataset balance
    dataset_statistics(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Print split statistics
    print("\n===== TRAIN/TEST SPLIT BALANCE =====")
    print("Train females:", np.sum(y_train == 0))
    print("Train males:  ", np.sum(y_train == 1))
    print("Test females:", np.sum(y_test == 0))
    print("Test males:  ", np.sum(y_test == 1))
    print("====================================\n")

    # Build model
    model = create_mlp(input_dim=X.shape[1])

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32
    )

    # Save model
    os.makedirs("../models", exist_ok=True)
    model.save("../models/gender_mlp.h5")

    # Save model summary
    with open("../models/model_summary.txt", "w", encoding="utf-8") as f:

        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # Plot accuracy graph
    os.makedirs("../results", exist_ok=True)

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("../results/accuracy.png")
    plt.show()

    # Plot loss curve
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Testing Loss')
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("../results/loss.png")
    plt.show()

if __name__ == "__main__":
    main()
