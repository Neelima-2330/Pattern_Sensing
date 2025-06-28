import numpy as np
from tensorflow.keras.models import load_model
from src.dataloader import get_data_generators
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def evaluate():
    model = load_model('outputs/model.h5')
    _, val_gen = get_data_generators('data/raw/patterns')

    val_gen.reset()
    predictions = model.predict(val_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes
    class_names = list(val_gen.class_indices.keys())

    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    os.makedirs('outputs', exist_ok=True)
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()
