# Week 2 – Student Performance Prediction
This project uses a **synthetic educational dataset** (600 generated records; no real student data).

- Regression: predict final score (0–100) using Linear Regression and Random Forest Regressor.
- Classification: predict pass/fail (score >= 60) using Logistic Regression and Random Forest Classifier.
- Preprocessing: median imputation, StandardScaler and OneHotEncoder.
- Metrics: MAE, RMSE, R², accuracy, precision, recall, F1 and confusion matrix.

## Windows PowerShell
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_models.py
```
If activation is blocked:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
Outputs are saved in `outputs/`. Synthetic data is for learning only, not real student assessment.
