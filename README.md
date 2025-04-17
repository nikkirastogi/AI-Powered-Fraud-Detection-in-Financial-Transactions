# AI-Powered-Fraud-Detection-in-Financial-Transactions

Traditional rule-based systems often fail to catch sophisticated fraud patterns involving fake identities, bots, and stolen credentials. 
My objective is to build a scalable, interpretable, and accurate ML-based fraud detection system capable of identifying such fraud in real-time.

---

## 🎯 Objectives

- Detect fraudulent transactions using ML/DL models
- Handle class imbalance effectively using SMOTE and class weighting
- Evaluate models with fraud-focused metrics (Precision, Recall, AUC)
- Enable real-time fraud flagging capability
- Provide model interpretability and feature insight

---

## 📁 Dataset

- **Source:** IEEE-CIS Fraud Detection Challenge (Kaggle)
- **Files Used:** `train_transaction.csv`, `train_identity.csv`
- **Size:** ~590,000 records, 434 features after merging
- **Imbalance:** Only ~3.5% transactions are labeled as fraud
- train_transaction.csv : https://www.kaggle.com/c/ieee-fraud-detection/data?select=train_transaction.csv
- train_identity.csv : https://www.kaggle.com/c/ieee-fraud-detection/data?select=train_identity.csv

---

## 🛠️ Data Processing Steps

- Merged transaction and identity data on `TransactionID`
- Removed features with >20% missing values
- Imputed remaining missing values (Median for numeric, Mode for categorical)
- Removed low-variance features (>90% single value)
- Final shape: 590540 × 182
- One-hot encoded categorical variables
- Scaled numerical features using `StandardScaler`
- Balanced the training set using **SMOTE**
- Applied stratified sampling for train/test splits

---

### Modeling Strategy

- Models Evaluated:
  SMOTE
   - Logistic Regression (with/without SMOTE)
   - Random Forest (all + selected features)
  Stratified Sampling
   - XGBoost
   - LightGBM (baseline + tuned) using Stratified Sampling
   - Deep Neural Network (100 & 400 epochs) using Stratified Sampling
     
- Highlights:
  - SMOTE oversampling
  - Stratified Sampling
  - Feature selection with Random Forest
  - Evaluation: Precision, Recall, F1, AUC-PR

---

### SMOTE – Synthetic Minority Oversampling

- Balances the dataset by generating synthetic examples from minority class
- Interpolates between nearest neighbors
- Improves recall, reduces overfitting

---
### Stratified Sampling
- Maintains the same fraud-to-non-fraud ratio in train/test sets
- Ensures consistent evaluation without data leakage
- This dataset have only 3.5% fraud

### Feature Selection

- Top 20 features selected via Random Forest
- Key features: `TransactionAmt`, `card1`, `C13`, `C14`, `card2`
- Helps reduce dimensionality and increase interpretability

---

## Evaluation Metrics
- Focused on recall to catch as many frauds as possible

![image](https://github.com/user-attachments/assets/41c413b4-6741-438a-89cd-36f198834d8c)

## Observations:
- Random Forest performs well but struggles with recall.
- XGBoost has balanced recall and precision but isn't the best in F1-score.
- LGBM performs better after tuning, but precision drops.
- DNN achieves a good balance but needs more tuning for higher precision.

---
### 🔍 Final Recommendation

- **Best Overall:** LightGBM (Tuned) – Balanced fraud detection
- **High Recall Needs:** LightGBM (Tuned)
- **Balanced Use Case:** XGBoost
- **Fast & Interpretable:** Random Forest
- **Powerful GPUs & Deep Learning:** DNN

## 🚀 Deploying with Streamlit

This project includes a Streamlit web app to interactively test the fraud detection model.

### 🔧 Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/nikkirastogi/AI-Powered-Fraud-Detection-in-Financial-Transactions.git
   cd AI-Powered-Fraud-Detection-in-Financial-Transactions
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

4. **Interact with the App**
   - Enter transaction details in the sidebar
   - Get prediction and probability of fraud

### 📁 App Folder Structure

```
.
├── app.py                        # Streamlit app interface
├── model/
│   ├── fraud_model.pkl           # Trained classifier (e.g., XGBoost)
│   └── scaler.pkl                # StandardScaler used in preprocessing
├── requirements.txt             # App dependencies
└── README.md
```

---

## Contact

Feel free to reach out with questions or collaboration ideas:  
**Nikki Rastogi** – nikkirastogi1998@gmail.com

---
