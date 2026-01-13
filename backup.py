import streamlit as st
import pandas as pd
import pickle

#set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

#read excel file
df = pd.read_excel("Telco_customer_churn.xlsx")

df = df.drop(columns = ["CustomerID", "Count", "Country", "State", "City", "Zip Code", "Latitude", "Longitude",
                        "Churn Label", "Churn Score", "CLTV", "Lat Long", "Churn Reason"])

#Import knn model and columns
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
trained_columns = pickle.load(open("columns.pkl", "rb"))

st.title("📉 Customer Churn Prediction Dashboard")
st.subheader("Predict whether customer will Churn or Stay")

#sidebar
st.sidebar.title("Customer Input Form")
#customer details
st.sidebar.subheader("👤 Customer Details")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependent = st.sidebar.selectbox("Dependents", ["Yes", "No"])
#subscription
st.sidebar.subheader("📄 Subscription")
tenure_month = st.sidebar.slider("Tenure Month", 0, 72, 12)
contract = st.sidebar.selectbox("Contact Number", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method",["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
paperless_bill = st.sidebar.selectbox("Paperless Billing",["Yes", "No"])
#services
st.sidebar.subheader("🌐 Services")
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No"])
#charges
st.sidebar.subheader("💰 Charges")
monthly_charges = st.sidebar.number_input("Monthly charges", min_value = 0)
total_charges = st.sidebar.number_input("Total charges", min_value = 0)
st.sidebar.markdown("---")
st.sidebar.button("Predict")

