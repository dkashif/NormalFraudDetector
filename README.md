# 💳 Credit Card Fraud Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://normalfrauddetector-efx5avwlbv7fflqigwhmfl.streamlit.app/)

A machine learning pipeline that detects fraudulent financial transactions trained on **6.3 million real transactions**. Deployed as an interactive web app.

---

## 🔍 Live Demo
**[Try it here →](https://normalfrauddetector-efx5avwlbv7fflqigwhmfl.streamlit.app/)**

Enter transaction details and get a real-time fraud probability score.

---

## 📊 Results

| Metric | Logistic Regression (baseline) | XGBoost + SMOTE (final) |
|---|---|---|
| Precision (Fraud) | 0.02 | **0.51** |
| Recall (Fraud) | 0.94 | **0.99** |
| F1 (Fraud) | 0.04 | **0.68** |
| ROC-AUC | 0.99 | **0.9992** |
| **AUC-PR** | 0.56 | **0.9984** |

The model catches **1,640 out of 1,643 fraud cases** in the test set (only 3 missed across 1.27M transactions).

---

## 🧠 Technical Highlights

**Why not just use accuracy?**
The dataset is severely imbalanced — only 0.13% of transactions are fraud. A model that predicts "not fraud" on everything achieves 99.87% accuracy while being completely useless. This project uses AUC-PR (area under the precision-recall curve) as the primary metric, which is the industry standard for imbalanced classification.

**Handling class imbalance — two-pronged approach:**
- **SMOTE** (Synthetic Minority Oversampling) generates synthetic fraud examples during training to balance the classes
- **XGBoost's `scale_pos_weight`** further penalises the model for missing real fraud cases (set to 774x)

**Feature engineering:**
Rather than feeding raw transaction fields into the model, domain-specific features were engineered to capture known fraud patterns:

- `origDrained` — did the sender's account get completely emptied? (97.6% of fraud vs 23.8% of legit)
- `amountEqualsOldBalance` — does the transfer amount exactly match the sender's balance? (97.8% of fraud vs 0% of legit)
- `destWasEmpty` — was the receiver account empty before the transfer? (mule account pattern)
- `destBalanceMismatch` — did the destination balance not increase by the expected amount?
- `logAmount` — log-transformed transaction amount to reduce skew

**Threshold tuning:**
XGBoost outputs a fraud probability (0–1). Rather than using the default 0.5 cutoff, the decision threshold was optimised to maximise F1 score, landing at **0.85** — only flagging transactions the model is highly confident about.

---

## 🛠️ Stack

- **Python** — pandas, numpy, scikit-learn, XGBoost, imbalanced-learn
- **Modelling** — XGBoost classifier inside a scikit-learn Pipeline
- **Imbalance handling** — SMOTE via imbalanced-learn
- **Deployment** — Streamlit, Streamlit Cloud

---

## 📁 Project Structure

```
├── fraud_detection_v2.ipynb   # Full analysis + modelling notebook
├── app.py                     # Streamlit app
├── fraud_detection_v2.pkl     # Saved pipeline + threshold
└── requirements.txt
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/yourusername/NormalFraudDetector
cd NormalFraudDetector
pip install -r requirements.txt
streamlit run app.py
```

> **Note:** Dataset not included due to size. Download from [Kaggle — PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) and place as `AIML Dataset.csv` in the project root to rerun the notebook.

---

## 📈 Dataset

- **Source:** PaySim synthetic mobile money transaction simulator
- **Size:** 6,362,620 transactions
- **Fraud rate:** 0.13% (8,213 fraudulent transactions)
- **Features:** Transaction type, amount, sender/receiver balances
