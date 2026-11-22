
# 🏥 Smart Hospital Patient Waiting-Time Prediction

## 📌 Project Overview

This project uses Machine Learning techniques to analyze and predict patient waiting times in hospital Emergency Departments (EDs).

The main objective is to estimate the waiting time before a patient receives their first contact with a healthcare provider, using patient characteristics, arrival information, and visit-related features.

The project includes data preprocessing, exploratory data analysis, model training, evaluation, feature importance analysis, and a Streamlit-based prediction interface.

## 📊 Dataset

- **Source:** CDC National Hospital Ambulatory Medical Care Survey (NHAMCS), 2022 Emergency Department dataset.
- **Original records:** 16,025
- **Valid records used:** 13,272
- **Target variable:** `WAITTIME`
- **Target description:** Waiting time in minutes before first contact with a healthcare provider.

Records with blank and not-applicable waiting-time codes were excluded.

## ⚙️ Project Workflow

1. Dataset collection and exploration.
2. Data cleaning and preprocessing.
3. Feature selection and feature engineering.
4. Exploratory data analysis.
5. Training multiple regression models.
6. Model evaluation and comparison.
7. Feature importance analysis.
8. Development of a Streamlit prediction application.

## 🧠 Machine Learning Models

Three regression algorithms were implemented:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Gradient Boosting was selected as the final model based on its overall RMSE and R² performance.

## 📈 Model Evaluation

The models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 33.6558 | 62.9840 | 0.0336 |
| Random Forest | 37.2440 | 66.7391 | -0.0851 |
| Gradient Boosting | 34.0500 | 62.9241 | 0.0354 |

### Evaluation Summary

- Gradient Boosting achieved the highest R² score and lowest RMSE.
- Linear Regression achieved the lowest MAE.
- All models demonstrated limited predictive performance.

The low R² scores indicate that the selected features explain only a small proportion of the variation in waiting times.

## 🔍 Important Features

The Gradient Boosting model identified the following influential features:

- Arrival time
- Month of visit
- Patient age
- Ambulance arrival status
- Hospital transfer status

Feature importance represents model-specific predictive contribution and does not establish causation.

## 💻 Streamlit Application

A Streamlit web application was developed to demonstrate the trained machine learning model.

### Application Features

- Patient age and sex input.
- Month and day of visit selection.
- Patient arrival time input in 12-hour format.
- Ambulance arrival information.
- Hospital transfer information.
- Patient residence category.
- Estimated waiting-time prediction.

### Run the Application Locally

Activate your virtual environment and run:

```bash
streamlit run app.py
```

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook
- Streamlit

## 📁 Project Structure

```text
medical_project/
│
├── dataset/
│   └── raw/
│
├── models/
│   ├── gradient_boosting_model.pkl
│   └── preprocessor.pkl
│
├── results/
│   ├── model_comparison.csv
│   └── feature_importance.csv
│
├── analysis.ipynb
├── app.py
├── README.md
└── .gitignore
```

## ⚠️ Limitations

- The models have relatively low R² scores.
- The available features may not capture important factors such as emergency department workload and patient urgency.
- Historical data does not guarantee accurate predictions for individual future patients.
- Feature importance does not establish causal relationships.
- The model has not been clinically validated.

## 🚀 Future Improvements

- Explore additional relevant features.
- Perform hyperparameter tuning.
- Experiment with additional regression algorithms.
- Improve model interpretability.
- Investigate additional factors affecting hospital waiting times.
- Improve the prediction interface.

## ⚕️ Disclaimer

This project is developed strictly for educational and research purposes.

The predictions are experimental estimates and must not be used for clinical decision-making, emergency triage, or patient-care prioritization.
