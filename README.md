# Smart Hospital Patient Waiting-Time Prediction

## Project Overview

This project uses Machine Learning techniques to analyze and predict patient waiting times in hospital Emergency Departments (EDs).

The objective is to investigate how patient characteristics, arrival times, and visit-related information are associated with waiting time before the first healthcare provider contact.

## Dataset

- **Source:** CDC National Hospital Ambulatory Medical Care Survey (NHAMCS), 2022 Emergency Department dataset.
- **Original dataset:** 16,025 records.
- **Records used:** 13,272 valid waiting-time records.
- **Target variable:** `WAITTIME` (minutes).

Records with blank or not-applicable waiting-time codes were excluded.

## Machine Learning Models

Three regression algorithms were implemented:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

## Model Evaluation

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 33.6558 | 62.9840 | 0.0336 |
| Random Forest | 37.2440 | 66.7391 | -0.0851 |
| Gradient Boosting | 34.0500 | 62.9241 | 0.0354 |

Gradient Boosting achieved the highest R² and lowest RMSE, while Linear Regression achieved the lowest MAE.

The models showed limited predictive performance, indicating that the selected features explain only a small proportion of waiting-time variation.

## Important Features

The Gradient Boosting model identified these influential features:

- Arrival time
- Month of visit
- Patient age
- Ambulance arrival status
- Hospital transfer status

Feature importance indicates predictive contribution, not causation.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook

## Project Structure

```text
Smart-Hospital-Patient-Waiting-Time-Prediction/
│
├── dataset/
│   └── raw/
│
├── notebooks/
│   └── analysis.ipynb
│
├── models/
│   ├── gradient_boosting_model.pkl
│   └── preprocessor.pkl
│
├── results/
│   ├── model_comparison.csv
│   └── feature_importance.csv
│
├── README.md
└── .gitignore
```

## Limitations

- The models have relatively low R² scores.
- The available features may not capture important factors such as emergency department workload and patient urgency.
- The dataset represents historical hospital visits and does not guarantee accurate predictions for individual future patients.
- The results demonstrate statistical associations, not causal relationships.

## Future Improvements

- Explore additional relevant features.
- Perform hyperparameter tuning.
- Experiment with additional regression algorithms.
- Improve model interpretability.
- Develop a simple prediction interface.

## Disclaimer

This project is for educational and research purposes only. Its predictions should not be used for clinical decision-making or patient-care prioritization.