
import streamlit as st
import pandas as pd
import joblib
from datetime import time

# ---------------------------------
# 1. Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Hospital Waiting Time Predictor",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------------
# 2. Custom CSS
# ---------------------------------

st.markdown("""
<style>
    .main {
        background-color: #f5f8fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    .main-title {
        font-size: 32px;
        font-weight: 750;
        color: #163a5f;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #64748b;
        text-align: center;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .section-heading {
        color: #163a5f;
        font-size: 21px;
        font-weight: 650;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .result-card {
        background-color: #eaf4ff;
        padding: 25px;
        border-radius: 14px;
        text-align: center;
        border: 1px solid #c9e0fa;
        margin-top: 20px;
    }

    .result-label {
        color: #315b82;
        font-size: 16px;
        font-weight: 600;
    }

    .result-value {
        color: #1261a0;
        font-size: 42px;
        font-weight: 750;
    }

    .footer {
        text-align: center;
        color: #8492a6;
        font-size: 13px;
        margin-top: 35px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# 3. Load Model and Preprocessor
# ---------------------------------

@st.cache_resource
def load_models():
    model = joblib.load("models/gradient_boosting_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_models()

# ---------------------------------
# 4. Application Header
# ---------------------------------

st.markdown(
    '<div class="main-title">🏥 Smart Hospital Patient Waiting-Time Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Emergency Department Waiting-Time Estimation</div>',
    unsafe_allow_html=True
)

st.info(
    "Educational and research demonstration only. "
    "This experimental model is not clinically validated and must not "
    "be used for medical decisions or patient-care prioritization."
)

st.divider()

# ---------------------------------
# 5. Patient Information
# ---------------------------------

st.markdown(
    '<div class="section-heading">👤 Patient Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Patient Age (years)",
        min_value=0,
        max_value=94,
        value=30,
        step=1
    )

with col2:
    sex = st.selectbox(
        "Patient Sex",
        options=[1, 2],
        format_func=lambda x: "Female (1)" if x == 1 else "Male (2)"
    )

st.divider()

# ---------------------------------
# 6. Visit Information
# ---------------------------------

st.markdown(
    '<div class="section-heading">📅 Visit Information</div>',
    unsafe_allow_html=True
)

col3, col4, col5 = st.columns(3)

month_names = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

day_names = [
    "Sunday", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday"
]

with col3:
    month_name = st.selectbox("Month of Visit", month_names)
    vmonth = month_names.index(month_name) + 1

with col4:
    day_name = st.selectbox("Day of Visit", day_names)
    vdayr = day_names.index(day_name) + 1

with col5:
    arrival_time = st.time_input(
        "Patient Arrival Time",
        value=time(10, 30),
        format="12h"
    )

arr_minutes = arrival_time.hour * 60 + arrival_time.minute

st.divider()

# ---------------------------------
# 7. Arrival Information
# ---------------------------------

st.markdown(
    '<div class="section-heading">🚑 Arrival Information</div>',
    unsafe_allow_html=True
)

col6, col7 = st.columns(2)

with col6:
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

with col7:
    transfer_status = st.selectbox(
        "Was the patient transferred from another hospital or urgent care facility?",
        options=["Not applicable", "Yes", "No", "Unknown"]
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

st.divider()

# ---------------------------------
# 8. Prediction
# ---------------------------------

st.markdown(
    '<div class="section-heading">📊 Waiting-Time Prediction</div>',
    unsafe_allow_html=True
)

if st.button(
    "🔍 Predict Waiting Time",
    type="primary",
    use_container_width=True
):

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

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Estimated Waiting Time</div>
            <div class="result-value">{prediction:.1f} minutes</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "This is an experimental model estimate, not a guaranteed "
        "or clinically validated waiting time."
    )

# ---------------------------------
# 9. Footer
# ---------------------------------

st.markdown("""
<div class="footer">
    Smart Hospital Patient Waiting-Time Prediction |
    Machine Learning Project | Educational Demonstration
</div>
""", unsafe_allow_html=True)
