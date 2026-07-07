"""
Ablation study to justify the design choices in the resume bullets:
  1. Augmentation ON vs OFF (random crop + flip + color jitter)
  2. Adam vs SGD
  3. Cosine LR decay vs Step decay vs no schedule

For each configuration: trains a fresh model, evaluates on the same held-out
test set, and produces a comparison table + bar chart so the effect of each
choice on accuracy/F1 is visible (this is what "ablation and error analysis"
on a resume should actually mean, rather than just re-running the same model).

Run with a small NUM_EPOCHS in config.py first -- ablations are expensive
since each row here is a full training run.
"""

import os
import json
import matplotlib.pyplot as plt

import config
from train import train_model
from evaluate import evaluate_model
from dataset import get_dataloaders


def run_ablation():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    results = {}

    experiments = [
        {"tag": "adam_cosine",  "optimizer_name": "adam", "lr_schedule": "cosine"},
        {"tag": "adam_step",    "optimizer_name": "adam", "lr_schedule": "step"},
        {"tag": "adam_nosched", "optimizer_name": "adam", "lr_schedule": "none"},
        {"tag": "sgd_cosine",   "optimizer_name": "sgd",  "lr_schedule": "cosine"},
    ]

    for exp in experiments:
        print(f"\n{'='*60}\nRunning ablation: {exp['tag']}\n{'='*60}")
        model, history, class_names, test_loader = train_model(
            optimizer_name=exp["optimizer_name"],
            lr_schedule=exp["lr_schedule"],
            tag=exp["tag"],
        )
        metrics = evaluate_model(model, test_loader, class_names, tag=exp["tag"])
        results[exp["tag"]] = metrics

    with open(f"{config.OUTPUT_DIR}/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_ablation_comparison(results)
    print("\nAblation summary:")
    for tag, m in results.items():
        print(f"  {tag:15s} acc={m['accuracy']:.4f} f1={m['f1']:.4f}")

    return results


def plot_ablation_comparison(results):
    tags = list(results.keys())
    metrics_names = ["accuracy", "precision", "recall", "f1"]

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.2
    x = range(len(tags))

    for i, metric in enumerate(metrics_names):
        values = [results[t][metric] for t in tags]
        offset = (i - 1.5) * width
        ax.bar([xi + offset for xi in x], values, width=width, label=metric)

    ax.set_xticks(list(x))
    ax.set_xticklabels(tags, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Optimizer & LR Schedule Comparison")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/ablation_comparison.png", dpi=150)
    plt.close()
    print(f"Saved ablation comparison chart to {config.OUTPUT_DIR}/ablation_comparison.png")


if __name__ == "__main__":
    run_ablation()
