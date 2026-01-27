# Customer Churn Prediction Project

This project builds an end-to-end **machine learning pipeline** to predict customer churn using Python, Scikit-learn, and a trained **RandomForestClassifier**. The pipeline includes data preprocessing, class balancing, model training, evaluation, and deployment integration.

---

## Project Overview

The goal of this project is to predict whether a customer will **churn (leave the service)** based on their demographic, service usage, and account information.

The system includes:

* Data preprocessing
* Feature encoding
* Scaling
* Model training
* Model evaluation
* Model serialization
* Deployment integration (Streamlit)

---

## Model Used

**Best Model Selected:** `RandomForestClassifier`

### Why Random Forest?

* High accuracy
* Handles non-linear relationships well
* Robust to noise
* Reduces overfitting vs single decision trees
* Works well with mixed feature types

---

## Project Structure

```
project-folder/
│
├── app.py                      
├── customer_churn.ipynb  
├── pipeline.pkl      
├── requirements.txt         
└── README.md                  
```

---

### Run the app

```bash
python -m streamlit run app.py
```

---

##  Model Pipeline Components

* `ColumnTransformer`
* `OneHotEncoder`
* `StandardScaler`
* `RandomForestClassifier`
* 
---

## Model Performance

The Random Forest model was selected based on:

* Accuracy
* Stability
* Generalization performance
* Lower overfitting

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* Matplotlib / Seaborn
* Pickle
* Streamlit

---

## Features

* End-to-end ML pipeline
* Automatic preprocessing
* Class imbalance handling
* Production-ready model
* Deployable architecture
* Scalable design

---
