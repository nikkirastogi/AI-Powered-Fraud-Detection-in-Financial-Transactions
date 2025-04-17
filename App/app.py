import streamlit as st
import pandas as pd
import joblib
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Detection Simulator", page_icon="🕵️‍♂️")

st.title("🕵️‍♂️ Real-Time Fraud Detection Simulator")
st.markdown("Simulate a transaction below to check if it gets flagged as fraud by the AI model.")

# --- 💳 Card Info ---
st.header("💳 Card Details")
card_number = st.text_input("Card Number (Fake)", max_chars=16, help="Use a fake card number for testing (e.g., 4111111111111111)")
cvv = st.text_input("CVV (Fake)", max_chars=3, help="Use a fake CVV for testing (e.g., 123)")
amount = st.number_input("Transaction Amount ($)", min_value=0.0, step=1.0, help="Enter the dollar amount for this transaction.")
card_type = st.selectbox("Card Brand", ["Visa", "Mastercard", "Discover"])
card_kind = st.selectbox("Card Type", ["Credit", "Debit"])

# --- 🛒 Purchase Info ---
st.header("🛒 Purchase Info")
product = st.radio("Type of Product", ["H - Electronics", "R - Retail", "S - Services", "W - Wholesale"], help="Choose the type of product or service being purchased.")
days_since = st.slider("How many days since your last transaction?", 0, 365)

# --- 📍 Billing Address ---
st.header("📍 Billing Address")
address = st.text_input("Street Address", help="Fake address (e.g., 123 Main Street)")
city = st.text_input("City", value="New York")
state = st.selectbox("State", ["NY", "CA", "TX", "FL", "IL"])
zip_code = st.text_input("ZIP Code", max_chars=5)

# Mapping region based on city (for demo)
region_map = {
    "New York": 325.0,
    "Los Angeles": 299,
    "Houston": 321,
    "Chicago": 181.0,
    "Miami": 335
}

model_lgbm_best = joblib.load('fraud_model_lgbm.pkl')

region = region_map.get(city, 300)  # default to 300 if city not in map

# --- 📧 Contact Info ---
st.header("📧 Email Info")
email = st.selectbox("Email Domain", [
    "gmail.com", "yahoo.com", "aol.com", "hotmail.com", "anonymous.com"
], help="Select the email domain associated with the transaction.")

# --- 🔍 Show Summary ---
st.divider()
st.subheader("📝 Summary of Transaction")
st.write(f"💰 Amount: ${amount}")
st.write(f"💳 Card: {card_type} ({card_kind}) | Number: {card_number} | CVV: {cvv}")
st.write(f"📦 Product: {product}")
st.write(f"📅 Days Since Last Txn: {days_since}")
st.write(f"🏠 Address: {address}, {city}, {state} {zip_code} | Region Code: {region}")
st.write(f"📧 Email: {email}")



# Button to trigger backend prediction
if st.button("🚀 Check Transaction"):
    st.success("Input captured! Ready for model evaluation...")

    test_transaction = pd.read_csv('test_transaction.csv')
    # st.write(model_lgbm_best.predict_proba(test_transaction))
    test_transaction['addr1'] = region
    # print(region)
    test_transaction['TransactionAmt'] = amount
    test_transaction['card1'] = 10545
    test_transaction['card2'] = 489
    test_transaction['P_emaildomain_gmail.com'] = 1

    values = model_lgbm_best.predict(test_transaction)

    # print(test_transaction)
    # st.write(values)
    # st.write(model_lgbm_best.predict_proba(test_transaction)[:, 1]*100)
    if(model_lgbm_best.predict_proba(test_transaction)[:, 1]>=.5):
        st.write("Alert Fraud Detected with Probability " + str(model_lgbm_best.predict_proba(test_transaction)[:, 1][0] *100 ))
    if(model_lgbm_best.predict_proba(test_transaction)[:, 1]<=.5):
        st.write("Fraud Not Detected with Probability " + str(model_lgbm_best.predict_proba(test_transaction)[:, 0][0] *100 ))
    # TODO: Map these values into model input template



