# Heart Attack Data Visualization and Prediction Dashboard

This project creates an interactive dashboard for analyzing heart attack trends and predicting risks using machine learning. The dashboard processes a comprehensive dataset from Kaggle, trains a predictive model, and presents insights through an interactive Streamlit interface.

## Overview

The dashboard combines data visualization with machine learning to provide insights into heart attack risks and trends, utilizing data from Germany (2015-2023) across different age groups.

## Project Objectives

- **Data Cleaning and Preprocessing**: Handle missing and inconsistent data using Pandas and normalize features for machine learning
- **Data Visualization**: Create insightful visualizations using Matplotlib and Seaborn to explore relationships in the data
- **Machine Learning Predictions**: Implement Logistic Regression to predict heart attack risks
- **Interactive Dashboard**: Build an intuitive Streamlit interface for data exploration and predictions

## Technologies Used

- Python 3.8+
- **Libraries**:
  - Pandas & NumPy: Data preprocessing
  - Matplotlib & Seaborn: Visualization
  - scikit-learn: Machine learning
  - Streamlit: Dashboard creation

## Dataset

The dataset contains heart attack cases from Germany (2015-2023), covering both youth (under 25) and adults (25+).

### Key Features

- **Health Factors**: BMI, Cholesterol Level, Hypertension
- **Lifestyle Factors**: Smoking Status, Alcohol Consumption, Diet Quality
- **Environmental Factors**: Air Pollution Index, Region
- **Socioeconomic Factors**: Education Level, Employment Status, Income Level
- **Target Variable**: Heart_Attack_Incidence (0 = No Heart Attack, 1 = Heart Attack)

## Workflow

### 1. Data Preprocessing
- Handle missing values and outliers
- Encode categorical variables
- Normalize numerical variables using StandardScaler

### 2. Data Visualization
- Analyze heart attack trends by demographics
- Create region-wise and year-wise analysis
- Explore feature correlations

### 3. Model Development
- Split data into training and testing sets
- Train Logistic Regression model
- Generate predictions and probabilities

### 4. Model Evaluation
- Generate confusion matrix
- Plot ROC curve
- Calculate feature importance

### 5. Dashboard Implementation
- Display interactive EDA graphs
- Present model predictions
- Show performance metrics

## Visualizations

The dashboard includes:

- Heart attack trends by age and gender
- BMI distribution analysis
- Yearly and regional trend analysis
- Model evaluation metrics:
  - Confusion Matrix
  - ROC Curve with AUC score
  - Feature Importance Plot

## Future Enhancements

1. Interactive Data Filtering
   - Implement user interface controls (checkboxes, sliders, dropdowns)
   - Enable dynamic data filtering and visualization customization

2. Cloud Deployment
   - Deploy dashboard to Streamlit Cloud or AWS
 
