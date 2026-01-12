import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

from preprocess import load_dataset

def main():
    # Load model
    model = load_model("../models/gender_mlp.h5")

    # Load dataset
    X, y = load_dataset("../dataset")

    # Predict
    predictions = model.predict(X)
    predictions = (predictions > 0.5).astype("int32").flatten()

    # Confusion Matrix
    cm = confusion_matrix(y, predictions)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Female", "Male"],
                yticklabels=["Female", "Male"])
    plt.title("Confusion Matrix")
    plt.savefig("../results/confusion_matrix.png")
    plt.show()

    # Save classification report
    report = classification_report(y, predictions, target_names=["Female", "Male"])
    print("\n=== CLASSIFICATION REPORT ===")
    print(report)

    with open("../results/classification_report.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()
