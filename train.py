import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from model import SpaceNet
from utils import preprocess_data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== LOAD REAL DATA =====
    df = pd.read_csv("data.csv")

    # Giả sử cột label tên là 'label'
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
        for epoch in range(20):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        # ----- EVALUATE -----
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                p = F.softmax(model(xb.to(device)), dim=1)
                probs.append(p.cpu().numpy())
                labels.append(yb.numpy())

        auc = roc_auc_score(
            np.concatenate(labels),
            np.concatenate(probs),
            multi_class="ovr"
        )
        aucs.append(auc)
        print(f"AUC: {auc:.4f}")

    print("Mean AUC:", np.mean(aucs))
    print("Std AUC:", np.std(aucs))

if __name__ == "__main__":
    main()





