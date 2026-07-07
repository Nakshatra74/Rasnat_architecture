"""
Evaluation on the held-out test set:
- Overall accuracy, precision, recall, F1 (macro-averaged, matches resume metrics)
- Per-class precision/recall/F1
- Confusion matrix (raw + normalized), saved as a heatmap
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from config import DEVICE, OUTPUT_DIR


@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for inputs, labels in loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def evaluate_model(model, test_loader, class_names, tag="baseline"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    y_true, y_pred = get_predictions(model, test_loader)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n=== Test Set Results ({tag}) ===")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%  (macro-avg)")
    print(f"Recall:    {recall*100:.2f}%  (macro-avg)")
    print(f"F1-score:  {f1*100:.2f}%  (macro-avg)")

    report = classification_report(y_true, y_pred, target_names=class_names,
                                    zero_division=0, digits=4)
    print("\nPer-class report:\n", report)
    with open(f"{OUTPUT_DIR}/classification_report_{tag}.txt", "w") as f:
        f.write(report)

    plot_confusion_matrix(y_true, y_pred, class_names, tag)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def plot_confusion_matrix(y_true, y_pred, class_names, tag="baseline"):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(6 + len(class_names), 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"Confusion Matrix (counts) - {tag}")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"Confusion Matrix (normalized) - {tag}")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_{tag}.png", dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {OUTPUT_DIR}/confusion_matrix_{tag}.png")


if __name__ == "__main__":
    # Standalone usage: python evaluate.py loads the saved checkpoint and
    # re-runs evaluation without retraining.
    from model import build_model
    from dataset import get_dataloaders

    _, _, test_loader, class_names = get_dataloaders()
    model = build_model(len(class_names)).to(DEVICE)
    checkpoint = torch.load(f"{OUTPUT_DIR}/best_model_baseline.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluate_model(model, test_loader, class_names, tag="baseline")
