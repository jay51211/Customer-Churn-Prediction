import streamlit as st
import pandas as pd
import pickle

st.title("Customer Churn Prediction")

with open("pipeline.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["pipeline"]

st.caption("Predicting Customer Churn Using ML")

st.sidebar.header("User Inputs")
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
age = st.sidebar.number_input("Age")
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.number_input("No of Dependents")
referral = st.sidebar.number_input("No of Referrals")
tenure = st.sidebar.number_input("Tenure in Months")
offer = st.sidebar.text_input("Offer")
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
long_charge = st.sidebar.number_input("Avg Monthly Long Distance Charges")
muti_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No"])
internet_service = st.sidebar.selectbox("Internet Service", ["Yes", "No"])
internet_type = st.sidebar.text_input("Internet Type")
monthly_download = st.sidebar.number_input("Avg Monthly GB Download")
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No"])
protection_plan = st.sidebar.selectbox("Device Protection Plan", ["Yes", "No"])
premium_support = st.sidebar.selectbox("Premium Tech Support", ["Yes", "No"])
tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No"])
movie = st.sidebar.selectbox("Streaming Movie", ["Yes", "No"])
music = st.sidebar.selectbox("Streaming Music", ["Yes", "No"])
unlimited_data = st.sidebar.selectbox("Unlimited Data", ["Yes", "No"])
contract = st.sidebar.selectbox("Contract", ['One Year', 'Month-to-Month', 'Two Year'])
paperless_bill = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
pay_method = st.sidebar.selectbox("Payment Method", ['Credit Card', 'Bank Withdrawal', 'Mailed Check'])
monthly_charge = st.sidebar.number_input("Monthly Charge")
total_charges = st.sidebar.number_input("Total Charges")
total_refund = st.sidebar.number_input("Total Refund")
total_extra_data_charge = st.sidebar.number_input("Total Extra Data Charges")
total_long_dis_charge = st.sidebar.number_input("Total Long Distance Charges")
total_revenue = st.sidebar.number_input("Total Revenue")

user_input = {
    "Gender" : gender,
    "Age" : age,
    "Married" : married,
    "Number of Dependents" : dependents,
    "Number of Referrals" : referral,
    "Tenure in Months" : tenure,
    "Offer" : offer,
    "Phone Service" : phone,
    "Avg Monthly Long Distance Charges" : monthly_charge,
    "Multiple Lines" : muti_lines,
    "Internet Service" : internet_service,
    "Internet Type" : internet_type,
    "Avg Monthly GB Download" : monthly_download,
    "Online Security" : online_security,
    "Online Backup": online_backup,
    "Device Protection Plan" : protection_plan,
    "Premium Tech Support" : premium_support,
    "Streaming TV" : tv,
    "Streaming Movies" : movie,
    "Streaming Music" : music,
    "Unlimited Data" : unlimited_data,
    "Contract" : contract,
    "Paperless Billing" : paperless_bill,
    "Payment Method" : pay_method,
    "Monthly Charge" : monthly_charge,
    "Total Charges" : total_charges,
    "Total Refunds" : total_refund,
    "Total Extra Data Charges" : total_extra_data_charge,
    "Total Long Distance Charges" : total_long_dis_charge,
    "Total Revenue" : total_revenue
}

input_df = pd.DataFrame([user_input])

if st.button("Predict"):
    pred = model.predict(input_df)
    st.write(input_df)

    if pred == 1:
        st.error("Customer will Churn (Leave)")

    else:
        st.success("Customer will not Churn (Stay)")