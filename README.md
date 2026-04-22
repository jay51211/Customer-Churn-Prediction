# 📉 Customer Churn Prediction

> Predicting telecom customer churn using an end-to-end ML pipeline — with a live Streamlit app for business teams to use without writing a single line of code.

---

## 🧩 Problem Statement

Telecom companies lose significant revenue every year due to customer churn. The challenge is identifying **which customers are likely to leave before they actually do**, so the business can take proactive retention action.

This project builds a production-ready machine learning pipeline that predicts churn probability for each customer based on their demographics, service usage, and account history.

---

## 🚀 Live Demo

🔗 **[Try the Streamlit App](https://customer-churn-prediction-4ippd6emidve4v9dckebwh.streamlit.app/)**

Business users can enter customer details and instantly get a churn prediction — no coding required.

---

## 📊 Dataset

- **Source:** Telecom Customer Churn dataset
- **Records:** ~7,000 customers
- **Features:** Demographics, contract type, tenure, monthly charges, service subscriptions, payment method
- **Target:** `Churn` — Yes / No

---

## 🔍 Key Findings from EDA

- **Month-to-Month contract customers** churn at 3x the rate of yearly contract customers
- **Customers in the first 12 months** are the highest churn risk — early engagement is critical
- **Higher monthly charges** correlate strongly with churn, especially when paired with low tenure
- **Customers without tech support or online security** churn significantly more

---

## 💡 Business Recommendations

1. **Target new customers (0–12 months)** with loyalty incentives and onboarding support — this is the highest-risk window
2. **Offer contract upgrade discounts** to Month-to-Month customers, since long-term contracts drastically reduce churn
3. **Bundle tech support and security services** for high-value customers — these features are strong churn reducers
4. **Flag high monthly charge + low tenure customers** as priority for retention calls

---

## 🛠️ Tech Stack

| Area | Tools |
|------|-------|
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Pipeline | Scikit-learn (ColumnTransformer, OneHotEncoder, StandardScaler) |
| Model | RandomForestClassifier |
| Class Imbalance | Imbalanced-learn (SMOTE) |
| Deployment | Streamlit, Pickle |

---

## 🏗️ Project Structure

```
Customer-Churn-Prediction/
│
├── customer_churn.ipynb       # Full EDA + model training notebook
├── app.py                     # Streamlit web app
├── pipeline.pkl               # Trained & serialized ML pipeline
├── telecom_customer_churn.csv # Dataset
├── requirements.txt           # Dependencies
└── README.md
```

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/jay51211/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

---

## 📈 Model Performance

The **RandomForestClassifier** was selected after comparing multiple models. It was chosen for its:
- Strong accuracy on imbalanced data (after SMOTE balancing)
- Robustness to noisy/missing features
- Interpretability via feature importance scores

---

## 👤 Author

**Jay Kumbhar**
📧 jaykumbhar518@gmail.com
💼 [LinkedIn](https://linkedin.com/in/jaykumbhar5121) | 💻 [GitHub](https://github.com/jay51211)
