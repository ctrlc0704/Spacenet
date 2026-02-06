import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from model import SpaceNet
from utils import preprocess_data

def main(epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 Training with {epochs} epochs")

    # ===== LOAD DATA =====
    df = pd.read_csv("synthetic_data.csv")  # hoặc data.csv
    y = df["label"].values
    X = df.drop(columns=["label"]).values

    # ===== PREPROCESSING =====
    X, y, selected_idx = preprocess_data(
        X, y, top_k=25
    )

    print("Selected feature indices:", selected_idx)

    # ===== 5-FOLD CV =====
    skf = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    aucs = []

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}")

        model = SpaceNet(
            num_features=X.shape[1],
            num_classes=4
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X[tr], dtype=torch.float32),
                torch.tensor(y[tr], dtype=torch.long)
            ),
            batch_size=64,
            shuffle=True
        )

        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X[te], dtype=torch.float32),
                torch.tensor(y[te], dtype=torch.long)
            ),
            batch_size=64
        )

        # ----- TRAIN -----
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

    #     # ----- EVALUATE -----
    #     model.eval()
    #     probs, labels = [], []
    #     with torch.no_grad():
    #         for xb, yb in test_loader:
    #             p = F.softmax(model(xb.to(device)), dim=1)
    #             probs.append(p.cpu().numpy())
    #             labels.append(yb.numpy())

    #     auc = roc_auc_score(
    #         np.concatenate(labels),
    #         np.concatenate(probs),
    #         multi_class="ovr"
    #     )
    #     aucs.append(auc)
    #     print(f"AUC: {auc:.4f}")

    # print("Mean AUC:", np.mean(aucs))
    # print("Std AUC:", np.std(aucs))
  # ===== EVALUATE =====
        model.eval()
        probs, labels = [], []

        with torch.no_grad():
            for xb, yb in test_loader:
                p = torch.softmax(model(xb.to(device)), dim=1)
                probs.append(p.cpu().numpy())
                labels.append(yb.numpy())

        y_prob = np.concatenate(probs)
        y_true = np.concatenate(labels)

        # 👉 LƯU KẾT QUẢ TỪ FOLD NÀY
        all_probs.append(y_prob)
        all_labels.append(y_true)

        auc = roc_auc_score(
            y_true, y_prob, multi_class="ovr"
        )
        auc_scores.append(auc)
        print(f"AUC (Fold {fold + 1}): {auc:.4f}")

    # ===== SAVE FOR ROC PLOTTING =====
    y_prob_all = np.concatenate(all_probs)
    y_true_all = np.concatenate(all_labels)

    np.save("y_prob.npy", y_prob_all)
    np.save("y_true.npy", y_true_all)

    print("\n📊 FINAL RESULTS")
    print("Mean ROC-AUC:", np.mean(auc_scores))
    print("Std  ROC-AUC:", np.std(auc_scores))
    print("✅ ROC data saved: y_prob.npy & y_true.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SpaceNet with configurable epochs"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )

    args = parser.parse_args()

    main(epochs=args.epochs)





