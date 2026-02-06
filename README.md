# SpaceNet: Explainable Deep Learning for Space Tourism Demand Prediction

This repository contains the implementation of **SpaceNet**, a deep learning framework for modeling and predicting space tourism demand, following the methodology described in our research paper.

The pipeline integrates:
- Advanced data preprocessing (outlier removal, normalization, class balancing)
- Feature selection
- Deep neural network modeling
- Cross-validation
- Explainable AI (SHAP)
- Robust evaluation with ROC curves and confusion matrices

---

## 📌 Pipeline Code (Google Colab)

```python
!git clone https://github.com/ctrlc0704/Spacenet
%cd Spacenet
!ls
!pip install -r requirements.txt
!python data_statistics.py
!python train.py --epochs 50
!python plot_results.py


