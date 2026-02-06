import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns


# =====================================================
# Plot ROC curves (One-vs-Rest)
# =====================================================
def plot_roc_curves(y_true, y_prob, n_classes=4):
    """
    Plot One-vs-Rest ROC curves for multi-class classification.
    
    Parameters
    ----------
    y_true : array-like, shape (N,)
        Ground-truth class labels.
    y_prob : array-like, shape (N, n_classes)
        Predicted class probabilities.
    n_classes : int
        Number of classes.
    """

    plt.figure(figsize=(7, 6))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"Class {i} (AUC = {roc_auc:.3f})"
        )

    # Random classifier baseline
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig("roc_curve.png", dpi=300)
    plt.show()


# =====================================================
# Plot confusion matrix
# =====================================================
def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix as a heatmap.
    """

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()


# =====================================================
# Main
# =====================================================
def main():
    """
    Load saved predictions from train.py and generate figures.
    """

    print("📂 Loading prediction results...")

    y_true = np.load("y_true.npy")
    y_prob = np.load("y_prob.npy")

    print("y_true shape:", y_true.shape)
    print("y_prob shape:", y_prob.shape)

    # Convert probabilities to predicted labels
    y_pred = np.argmax(y_prob, axis=1)

    # Plot ROC curves
    plot_roc_curves(y_true, y_prob, n_classes=y_prob.shape[1])

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)

    print("✅ Figures saved: roc_curve.png & confusion_matrix.png")


if __name__ == "__main__":
    main()
