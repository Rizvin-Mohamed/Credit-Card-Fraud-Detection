# 🛡️ SafeSpend — Explainable Fraud Detection

**Live app:**  https://credit-card-fraud-detection-rizvin.streamlit.app

SafeSpend checks a bank statement for suspicious transactions and explains every alert in plain language, e.g.:

> 🔴 **₹48,500 at QuickPay Wallet XJ** · Delhi · 02:47 AM
> - ₹48,500 is 58× your usual spend of ₹832
> - Made at 02:47 AM; you usually pay between 10:00 and 19:00
> - Made in Delhi; you usually transact in Chennai

## The problem
Card fraud is rare — only **0.17%** of transactions in the dataset — so a model that always says "genuine" is 99.83% accurate and catches nothing. The challenge is catching fraud without flooding customers with false alarms, and explaining each alert so a person can act on it.

## How it works
SafeSpend combines two models:

| | Card fraud classifier | Personal anomaly detector |
|---|---|---|
| Type | Supervised (XGBoost) | Unsupervised (Isolation Forest) |
| Data | 284,807 real card transactions (Kaggle) | The user's own statement |
| Learns | What fraud looks like | What's *normal for this person* |
| Explained by | SHAP waterfall charts | SHAP-ranked plain-language reasons |

## Results

**Model selection** — 5-fold stratified cross-validation on training data only:

| Model | PR-AUC | Recall | Precision |
|---|---|---|---|
| **XGBoost + class weight** | **0.845 ± 0.026** | 0.83 | 0.91 |
| XGBoost | 0.842 ± 0.027 | 0.78 | 0.95 |
| Random Forest | 0.834 ± 0.031 | 0.78 | 0.94 |
| Logistic Regression + class weight | 0.754 ± 0.027 | 0.91 | 0.06 |

**Final test** — held-out test set (56,746 transactions, 95 frauds), used once:

| Threshold | Caught | Missed | False alarms | Precision | Recall |
|---|---|---|---|---|---|
| 0.5 (default) | 72 | 23 | 5 | 0.935 | 0.758 |
| 0.014 (tuned on out-of-fold predictions) | 78 | 17 | 45 | 0.634 | 0.821 |

Test PR-AUC: **0.826** (CV estimate 0.843).

**Personal anomaly detector** — synthetic 6-month statement (600 genuine + 8 injected frauds): **8/8 frauds caught, 0 false alarms**. On the same statement with the frauds removed: 0 high-risk alerts.

## Key findings
- **Accuracy is misleading** on imbalanced data; PR-AUC is the main metric. Logistic Regression had the *highest* ROC-AUC (0.982) but only 6% precision.
- **Class weights and SMOTE shift the threshold; they don't make a model smarter** — PR-AUC stayed ~0.68 for all Logistic Regression variants.
- **Recall plateaus at ~83%.** About 17% of frauds look genuine to the model at every threshold, so threshold tuning buys little recall at a large precision cost. This motivated the personal anomaly detector, which uses each user's own history.
- SHAP shows **V14, V4, V12, V10, V11** drive the classifier — the same features EDA correlation found.

## Project structure
```
safespend/
├── app/            Streamlit app + demo data
├── models/         Trained XGBoost model, scaler, thresholds
├── notebooks/      01 EDA → 08 app files, in order
├── src/            safespend.py — reusable features, scoring, explanations
└── requirements.txt
```

## Run it locally
```bash
git clone https://github.com/YOUR-USERNAME/safespend.git
cd safespend
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```
To retrain the models, download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) into `data/` and run the notebooks in order.

## Limitations
- The anomaly detector looks at one transaction at a time; fraud that matches your usual amount, time and place won't be caught.
- The test statement is synthetic and its frauds are deliberately obvious.
- The Kaggle features (V1–V28) are anonymised, so the classifier can't be applied to real users' statements directly.

## Tech
Python · pandas · scikit-learn · XGBoost · imbalanced-learn · SHAP · Streamlit · Altair

## Data and privacy
Card data: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group). The sample bank statement is synthetic. Uploaded statements are processed in memory and never stored.