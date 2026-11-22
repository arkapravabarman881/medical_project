
import streamlit as st
import pandas as pd
import joblib
from datetime import time

# ---------------------------------
# 1. Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Smart Hospital Waiting-Time Predictor",
    page_icon="🏥",
    layout="centered"
)

# ---------------------------------
# 2. Load Trained Model and Preprocessor
# ---------------------------------

@st.cache_resource
def load_models():
    model = joblib.load("models/gradient_boosting_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_models()

# ---------------------------------
# 3. Application Header
# ---------------------------------

st.title("🏥 Smart Hospital Patient Waiting-Time Prediction")

st.write(
    "Enter patient and arrival information to estimate "
    "the waiting time before the first healthcare provider contact."
)

st.info(
    "Educational and research demonstration only. "
    "This tool must not be used for clinical decisions "
    "or patient-care prioritization."
)

st.divider()

# ---------------------------------
# 4. Patient Information
# ---------------------------------

st.header("Patient Information")

age = st.number_input(
    "Patient Age (years)",
    min_value=0,
    max_value=94,
    value=30,
    step=1
)

sex = st.selectbox(
    "Patient Sex",
    options=[1, 2],
    format_func=lambda x: "Female (1)" if x == 1 else "Male (2)"
)
# ---------------------------------
# 5. Visit Information
# ---------------------------------

st.header("Visit Information")

month_names = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

month_name = st.selectbox(
    "Month of Visit",
    month_names
)

vmonth = month_names.index(month_name) + 1

day_names = [
    "Sunday", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday"
]

day_name = st.selectbox(
    "Day of Visit",
    day_names
)

vdayr = day_names.index(day_name) + 1

arrival_time = st.time_input(
    "Patient Arrival Time",
    value=time(10, 30),
    format="12h"
)

arr_minutes = arrival_time.hour * 60 + arrival_time.minute

# ---------------------------------
# 6. Arrival Method
# ---------------------------------

st.header("Arrival Information")

arrival_method = st.selectbox(
    "Did the patient arrive by ambulance?",
    options=["Yes", "No", "Unknown"]
)

arrems_mapping = {
    "Yes": "1",
    "No": "2",
    "Unknown": "Unknown"
}

arrems = arrems_mapping[arrival_method]

transfer_status = st.selectbox(
    "Was the patient transferred from another hospital or urgent care facility?",
    options=[
        "Not applicable",
        "Yes",
        "No",
        "Unknown"
    ]
)

transfer_mapping = {
    "Not applicable": "Not applicable",
    "Yes": "1",
    "No": "2",
    "Unknown": "Unknown"
}

ambtransfer = transfer_mapping[transfer_status]

residence_status = st.selectbox(
    "Patient Residence",
    options=[
        "Private residence",
        "Nursing home",
        "Homeless / homeless shelter",
        "Other",
        "Unknown"
    ]
)

residence_mapping = {
    "Private residence": "1",
    "Nursing home": "2",
    "Homeless / homeless shelter": "3",
    "Other": "4",
    "Unknown": "Unknown"
}

residnce = residence_mapping[residence_status]

residnce = residence_mapping[residence_status]

# ---------------------------------
# 7. Prediction
# ---------------------------------

st.divider()

if st.button("Predict Waiting Time", type="primary", use_container_width=True):

    input_data = pd.DataFrame([{
        "VMONTH": vmonth,
        "VDAYR": vdayr,
        "AGE": age,
        "SEX": sex,
        "ARREMS": arrems,
        "AMBTRANSFER": ambtransfer,
        "RESIDNCE": residnce,
        "ARR_MINUTES": arr_minutes
    }])

    categorical_features = [
        "VMONTH",
        "VDAYR",
        "SEX",
        "ARREMS",
        "AMBTRANSFER",
        "RESIDNCE"
    ]

    input_data[categorical_features] = (
        input_data[categorical_features].astype(str)
    )

    processed_input = preprocessor.transform(input_data)

    prediction = model.predict(processed_input)[0]

    st.subheader("Prediction Result")

    st.metric(
        label="Estimated Waiting Time",
        value=f"{prediction:.1f} minutes"
    )

    st.caption(
        "This is an experimental model estimate, not a guaranteed "
        "or clinically validated waiting time."
    )
