# Customer Churn Prediction Project

This project builds an end-to-end **machine learning pipeline** to predict customer churn using Python, Scikit-learn, and a trained **RandomForestClassifier**. The pipeline includes data preprocessing, class balancing, model training, evaluation, and deployment integration.

---

## Project Overview

The goal of this project is to predict whether a customer will **churn (leave the service)** based on their demographic, service usage, and account information.

The system includes:

* Data preprocessing
* Feature encoding
* Scaling
* Class imbalance handling (SMOTE)
* Model training
* Model evaluation
* Model serialization
* Deployment integration (Flask/Streamlit ready)

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

## Important Note About Model File

> The file `pipeline.pkl` is **NOT included** in this repository because of its large size.

### How to generate it:

You can easily recreate the model file by running the Jupyter notebook:

**`customer_churn.ipynb`**

At the end of the notebook, the trained pipeline is saved using:

```python
with open("pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)
```

This will automatically generate:

```
pipeline.pkl
```

---

### Train the model

Open Jupyter Notebook:

```bash
jupyter notebook
```

Run:

```
customer_churn.ipynb
```

This will:

* Preprocess data
* Train model
* Evaluate performance
* Save the trained pipeline as `pipeline.pkl`

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

All steps are wrapped in **one unified pipeline**, making deployment safe and consistent.

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
