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

#Import knn model and columns
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
trained_columns = pickle.load(open("columns.pkl", "rb"))

st.title("📉 Customer Churn Prediction Dashboard")
st.subheader("Predict whether customer will Churn or Stay")

#sidebar
st.sidebar.title("Customer Input Form")


drop_col = ["CustomerID", "Count", "Country", "State", "City", "Zip Code", "Latitude", "Longitude", "Churn Value"
            ,"Churn Label", "Churn Score", "CLTV", "Lat Long", "Churn Reason",]

feature_df = df.drop(columns=[c for c in drop_col if c in df.columns], errors="ignore")
feature_df["Total Charges"] = feature_df["Total Charges"].astype(str).str.strip()
feature_df["Total Charges"] = pd.to_numeric(feature_df["Total Charges"], errors="coerce")
feature_df["Total Charges"] = feature_df["Total Charges"].fillna(feature_df["Total Charges"].median())

categorical_features = feature_df.select_dtypes(include="object").columns.tolist()
numerical_features = feature_df.select_dtypes(include="number").columns.tolist()

user_input = {}

st.sidebar.subheader("Customer Information")
for col in categorical_features:
    options = sorted(feature_df[col].dropna().astype(str).unique().tolist())
    if len(options) > 0:
        user_input[col] = st.sidebar.selectbox(col, options)
    elif len(options) <= 0:
        user_input[col] = None

for col in numerical_features:
    user_input[col] = st.sidebar.number_input(col, min_value=0.0)

#Convert user input to dataframe
input_df = pd.DataFrame([user_input])

#Get every column data of user input
input_encoded = pd.get_dummies(input_df)

#Align column with column of feature_df
for col in trained_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

input_encoded = input_encoded[trained_columns]

#Display data of customer
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Your Input Data")
    st.write(input_df)

with col2:
    st.subheader("✅ Prediction Result")

    submit = st.button("Predict")

    if submit:
        # ✅ Scale data
        input_scaled = scaler.transform(input_encoded)

        # ✅ Predict
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.error("❌ Customer will CHURN (Leave)")
        else:
            st.success("✅ Customer will STAY (Not Churn)")

st.markdown("---")
st.write("This project is for educational purposes only")