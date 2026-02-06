# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from model import SpaceNet
from utils import preprocess_data

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === LOAD DATA (replace with real data) ===
    X = np.random.randn(1860, 44)
    y = np.random.randint(0, 4, size=1860)

    X, y, feat_idx, scaler = preprocess_data(X, y)

    skf = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    auc_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}")

        model = SpaceNet(num_features=X.shape[1]).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-3, weight_decay=1e-5
        )
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(
            torch.tensor(X[train_idx], dtype=torch.float32),
            torch.tensor(y[train_idx], dtype=torch.long)
        )
        test_ds = TensorDataset(
            torch.tensor(X[test_idx], dtype=torch.float32),
            torch.tensor(y[test_idx], dtype=torch.long)
        )

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64)

        # ---- Training ----
        for epoch in range(20):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        # ---- Evaluation ----
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
        auc_scores.append(auc)
        print(f"AUC: {auc:.4f}")

    print("Mean AUC:", np.mean(auc_scores))
    print("Std AUC:", np.std(auc_scores))

if __name__ == "__main__":
    train()
