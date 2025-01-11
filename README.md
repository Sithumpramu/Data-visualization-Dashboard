Heart Attack Data Visualization and Prediction Dashboard
This project creates a data visualization and prediction dashboard for analyzing heart attack trends and predicting risks using machine learning. The dashboard uses a dataset from Kaggle, preprocesses the data, trains a predictive model, and displays insights and evaluation metrics in an interactive Streamlit app.

Project Objectives
Data Cleaning and Preprocessing:

Handle missing and inconsistent data using Pandas.
Normalize numerical columns and encode categorical features for machine learning.
Data Visualization:

Create insightful visualizations (e.g., trends by age, gender, BMI) using Matplotlib and Seaborn to explore relationships in the data.
Machine Learning Predictions:

Use Logistic Regression from scikit-learn to train and test a model predicting heart attack risks.
Evaluate the model using:
Confusion Matrix
ROC Curve
Feature Importance
Interactive Dashboard:

Build an intuitive dashboard with Streamlit to display:
Pre-trained visualizations.
Prediction metrics.
Model evaluation insights.
Technologies Used
Python 3.8+
Libraries:
Pandas and NumPy for data preprocessing.
Matplotlib and Seaborn for visualization.
scikit-learn for machine learning.
Streamlit for building the dashboard.
Dataset
The dataset is sourced from Kaggle and contains information about heart attack cases in Germany (2015–2023) for youth (under 25) and adults (25+). Features include health, lifestyle, and environmental factors.

Key Features:
Health Factors: BMI, Cholesterol Level, Hypertension, etc.
Lifestyle Factors: Smoking Status, Alcohol Consumption, Diet Quality.
Environmental Factors: Air Pollution Index, Region.
Socioeconomic Factors: Education Level, Employment Status, Income Level.
Target Variable: Heart_Attack_Incidence (0 = No Heart Attack, 1 = Heart Attack).
Workflow
Data Cleaning and Preprocessing:

Handle missing values and outliers.
Encode categorical variables.
Normalize numerical variables using StandardScaler.
Data Visualization:

Analyze heart attack trends by age, gender, BMI, and smoking habits.
Create region-wise and year-wise heart attack trends.
Model Training and Testing:

Split the data into training and testing sets.
Train a Logistic Regression model using fit and evaluate predictions with predict and predict_proba.
Model Evaluation:

Generate the following evaluation metrics:
Confusion Matrix: Show correct and incorrect predictions.
ROC Curve: Measure the model's ability to classify cases.
Feature Importance: Display the impact of features on predictions.
Streamlit Dashboard:

Display EDA graphs (trends and relationships).
Present model predictions and performance metrics interactively.

Sample Visualizations
EDA Graphs:

Heart attack trends by age and gender.
BMI distribution and its impact on heart attack incidence.
Yearly and regional heart attack trends.
Evaluation Metrics:

Confusion Matrix
ROC Curve (with AUC score)
Feature Importance Plot

Future Enhancements
allow users to dynamically filter and customize the displayed data 
Deploy the dashboard to a cloud platform like Streamlit Cloud or AWS.
