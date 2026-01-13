# 📉 Customer Churn Prediction

This is a **Machine Learning + Streamlit web app** that predicts whether a telecom customer will **Churn (Leave)** or **Stay** using the **K-Nearest Neighbors (KNN) Classifier**.

---

## ✅ Project Overview

Customer churn means when a customer stops using a service.  
This project helps predict churn so that companies can take action to retain customers.

The app allows users to enter customer details using a simple Streamlit UI and get a churn prediction instantly.

---

## 🎯 Objective

To build an end-to-end ML project that:

✅ Cleans and preprocesses telecom customer data  
✅ Trains a **KNN Classifier** model  
✅ Evaluates performance using Accuracy, Precision, Recall, F1 Score  
✅ Deploys prediction using a **Streamlit web application**

---

## 🧾 Dataset

- Dataset Name: **Telco Customer Churn**
- File Used: `Telco_customer_churn.xlsx`
- Target Column: **Churn Value**
  - `1` → Customer will churn
  - `0` → Customer will stay

---

## ⚙️ Technologies Used

- Python 🐍  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  

---

## 🔍 Machine Learning Workflow

### ✅ Steps Performed

1. Load dataset  
2. Remove unnecessary / leakage columns:
   - CustomerID, Churn Label, Churn Score, Churn Reason, CLTV, City, Zip Code, etc.
3. Handle missing values (Total Charges)
4. Encode categorical features (Label Encoding + One Hot Encoding)
5. Split data into Train/Test set
6. Feature scaling using StandardScaler (Required for KNN)
7. Train KNN Classifier and find best `k`
8. Evaluate model using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - Confusion Matrix

---

## 📊 Model Performance (Example)

- Accuracy ≈ **78%**
- Precision, Recall, F1 score calculated using test dataset

*(Your score may vary slightly depending on split and k value.)*

---

## 🖥️ Streamlit App Features

✅ Sidebar input form for customer details  
✅ Dropdown selections for categorical values  
✅ Number input for numeric values  
✅ Prediction result displayed clearly:
- ✅ Customer will STAY  
- ❌ Customer will CHURN  

---

## 📂 Project Structure

Customer-Churn-Prediction/
│
├── app.py
├── customer_churn.ipynb
├── Telco_customer_churn.xlsx
├── knn_model.pkl
├── scaler.pkl
├── columns.pkl
└── README.md

