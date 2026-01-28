import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

@st.cache_data
def load_data():
    return pd.read_csv("telecom_customer_churn.csv")

graph_df = load_data()
graph_df = graph_df[graph_df["Monthly Charge"] >= 0]

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🧑‍💼",
    layout="wide"
)

with open("pipeline.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["pipeline"]
report = model_data["report"]

st.markdown(
    """
    <h1 style="text-align:center;">🧑‍💼 Customer Churn Prediction</h1>
    <p style="text-align:center; font-size:18px; color:gray;">
        Predict whether an customer is likely to leave the company
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

i1, i2, i3, i4, i5 = st.columns(5)
with i1:
    st.image("images/churn_by_contract.jpg", caption="Churn by Contract")
with i2:
    st.image("images/count_by_gender.jpg", caption = "Churn by Gender")
with i3:
    st.image("images/count_by_internet_type.png", caption = "Churn by Internet Type")
with i4:
    st.image("images/count_of_customer_by_total_charge.jpg", caption = "Churn by Total Charge")
with i5:
    st.image("images/age_scatter_monthly.jpg", caption = "Age vs Monthly Charge")

p1, p2 = st.columns(2)
with p1:
    st.markdown("### Input Features")
    with st.form("User Inputs"):
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.expander("Personal Data", expanded=True):
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                age = st.number_input("Age", step = 1)
                married = st.selectbox("Married", ["Yes", "No"])
                dependents = st.number_input("No of Dependents", step = 1)
                referral = st.number_input("No of Referrals", step = 1)
                tenure = st.number_input("Tenure in Months", step = 1)
                offer = st.selectbox("Offer", ['Offer A', 'Offer B', 'Offer C', 'Offer D', 'Offer E'])
                phone = st.selectbox("Phone Service", ["Yes", "No"])
                muti_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
                internet_service = st.selectbox("Internet Service", ["Yes", "No"])
        with c2:
            with st.expander("Service", expanded=True):
                contract = st.selectbox("Contract", ['One Year', 'Month-to-Month', 'Two Year'])
                internet_type = st.selectbox("Internet Type", ['Cable', 'Fiber Optic', 'DSL', 'No'])
                monthly_download = st.number_input("Avg Monthly GB Download", step = 1)
                online_security = st.selectbox("Online Security", ["Yes", "No"])
                online_backup = st.selectbox("Online Backup", ["Yes", "No"])
                protection_plan = st.selectbox("Device Protection Plan", ["Yes", "No"])
                premium_support = st.selectbox("Premium Tech Support", ["Yes", "No"])
                tv = st.selectbox("Streaming TV", ["Yes", "No"])
                movie = st.selectbox("Streaming Movie", ["Yes", "No"])
                music = st.selectbox("Streaming Music", ["Yes", "No"])
        with c3:
            with st.expander("Charges", expanded=True):
                unlimited_data = st.selectbox("Unlimited Data", ["Yes", "No"])
                paperless_bill = st.selectbox("Paperless Billing", ["Yes", "No"])
                long_charge = st.number_input("Avg Monthly Long Distance Charges")
                pay_method = st.selectbox("Payment Method", ['Credit Card', 'Bank Withdrawal', 'Mailed Check'])
                monthly_charge = st.number_input("Monthly Charge")
                total_charges = st.number_input("Total Charges")
                total_refund = st.number_input("Total Refund")
                total_extra_data_charge = st.number_input("Total Extra Data Charges")
                total_long_dis_charge = st.number_input("Total Long Distance Charges")
                total_revenue = st.number_input("Total Revenue")
        submit = st.form_submit_button("🔮 Predict Churn")

pred = None
if submit:
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

    pred = model.predict(input_df)

    if pred == 1:
        st.error("Customer will Churn (Leave)")
    else:
        st.success("Customer will not Churn (Stay)")

st.markdown("---")
st.subheader("📊 Model Performance (Test Data)")

report_df = pd.DataFrame(report).transpose()
report_df[["precision", "recall", "f1-score"]] *= 100
report_df = report_df.round(2)

st.dataframe(
    report_df,
    use_container_width=True
)

with p2:
    st.markdown("### 📍 Customer Churn Prediction")
    fig = px.scatter(
        graph_df,
        x="Age",
        y="Monthly Charge",
        color = "Customer Status",
        opacity = 0.8,
        labels = {
            "Age": "Age",
            "Monthly Charge": "Monthly Charge",
        },
        title = "Customer Distribution (Age vs Monthly Charge)",
    )
    if submit:
        fig.add_scatter(
            x = [age],
            y = [monthly_charge],
            mode = "markers",
            marker = dict(
                color = "red" if pred == 1 else "green",
                symbol = "star",
                size = 5
            ),
            name = "Current Point"
        )
    st.plotly_chart(fig, use_container_width=True)


    fig = px.scatter(
        graph_df,
        x = "Age",
        y = "Total Charges",
        color = "Customer Status",
        labels = {
            "Age": "Age",
            "Total Charges": "Total Charges",
        },
        title = "Customer Distribution (Age vs Total Charges)",
    )
    if submit:
        fig.add_scatter(
            x = [age],
            y = [total_charges],
            mode = "markers",
            marker = dict(
                color = "red" if pred == 1 else "green",
                symbol = "star",
                size = 5
            ),
            name = "Current Point"
        )
    st.plotly_chart(fig, use_container_width = True)
