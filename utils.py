# utils.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import shap
import torch

# -------- IQR Outlier Removal --------
def iqr_filter(X):
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    mask = np.all(
        (X >= Q1 - 1.5 * IQR) &
        (X <= Q3 + 1.5 * IQR),
        axis=1
    )
    return mask

# -------- Full Preprocessing --------
def preprocess_data(X, y, top_k=25):
    # 1. IQR
    mask = iqr_filter(X)
    X, y = X[mask], y[mask]

    # 2. Standard normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3. SMOTE
    X, y = SMOTE(random_state=42).fit_resample(X, y)

    # 4. Random Forest feature selection
    rf = RandomForestClassifier(
        n_estimators=200, random_state=42
    )
    rf.fit(X, y)
    idx = np.argsort(
        rf.feature_importances_
    )[::-1][:top_k]

    return X[:, idx], y, idx, scaler


def explain_with_shap(model, X_background, X_test):
    """
    model: trained SpaceNet
    X_background: small subset (e.g. 100 samples)
    X_test: samples to explain
    """
    model.eval()

    explainer = shap.DeepExplainer(
        model,
        torch.tensor(X_background, dtype=torch.float32)
    )

    shap_values = explainer.shap_values(
        torch.tensor(X_test, dtype=torch.float32)
    )

    return shap_values

