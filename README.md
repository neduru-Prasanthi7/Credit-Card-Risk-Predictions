Credit Card Default Prediction Project
Project Overview

The Credit Card Default Prediction System predicts the likelihood of a credit card user defaulting on payment. The project is implemented using Python and Object-Oriented Programming (OOP) concepts (classes, objects, functions).

It includes data preprocessing, feature selection, model training, evaluation, and deployment via a web interface.

Features

Data Preprocessing

Checked and removed null values.

Handled missing values using random sampling techniques.

Detected and treated outliers in the dataset.

Removed irrelevant columns using feature selection.

Balanced the dataset to handle class imbalance.

Machine Learning Models

K-Nearest Neighbors (KNN)

Naive Bayes

Logistic Regression

Decision Tree

Random Forest

AdaBoost

Model Evaluation

Test Accuracy

Classification Report

Confusion Matrix

ROC Curve & AUC Score

Best Model: Logistic Regression, selected based on ROC-AUC and overall performance.

Deployment

Model serialized using pickle.

Web interface created using Flask for real-time prediction.

Technologies Used

Python: pandas, numpy, scikit-learn, matplotlib, seaborn

Machine Learning: Classification algorithms

Web Deployment: Flask

Model Serialization: Pickle

Sample Screenshots
Web Interface


Enter user details to get prediction of credit card default.
Credit_Card_Default_Prediction/
│
├── app.py                     # Flask app for deployment
├── credit_model.pkl            # Trained Logistic Regression model
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html              # Web page for user input
├── static/
│   └── style.css               # Styling for the web interface
├── data/
│   └── credit_data.csv         # Dataset
├── src/
│   ├── preprocessing.py        # Data cleaning, missing value handling, outlier treatment
│   ├── feature_selection.py
│   ├── model_training.py
│   └── evaluation.py
└── README.md


Confusion Matrix

ROC Curve
