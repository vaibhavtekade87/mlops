import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Load model from Hugging Face Model Hub
model_path = hf_hub_download(repo_id="tekadevaibhav/tourism-package-model", filename="best_model.pkl")
model = joblib.load(model_path)

st.title("Wellness Tourism Package Prediction")
st.write("Predict whether a customer will purchase the Wellness Tourism Package")

# Get inputs from user
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    num_trips = st.number_input("Number of Trips", min_value=1, max_value=20, value=3)

with col2:
    passport = st.selectbox("Passport", [0, 1])
    own_car = st.selectbox("Own Car", [0, 1])
    num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=10000, max_value=100000, value=25000)
    pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    num_followups = st.number_input("Number of Followups", min_value=1, max_value=10, value=3)
    duration_of_pitch = st.number_input("Duration of Pitch (mins)", min_value=5, max_value=60, value=15)

# Save inputs into dataframe (matching training data column order)
input_data = pd.DataFrame({
    'Unnamed: 0': [0],
    'Age': [float(age)],
    'TypeofContact': [type_of_contact],
    'CityTier': [city_tier],
    'DurationOfPitch': [float(duration_of_pitch)],
    'Occupation': [occupation],
    'Gender': [gender],
    'NumberOfPersonVisiting': [num_person_visiting],
    'NumberOfFollowups': [float(num_followups)],
    'ProductPitched': [product_pitched],
    'PreferredPropertyStar': [float(preferred_property_star)],
    'MaritalStatus': [marital_status],
    'NumberOfTrips': [float(num_trips)],
    'Passport': [passport],
    'PitchSatisfactionScore': [pitch_satisfaction_score],
    'OwnCar': [own_car],
    'NumberOfChildrenVisiting': [float(num_children_visiting)],
    'Designation': [designation],
    'MonthlyIncome': [float(monthly_income)]
})

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("Customer is likely to PURCHASE the Wellness Tourism Package!")
    else:
        st.error("Customer is NOT likely to purchase the package.")
