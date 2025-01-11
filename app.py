import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
data = pd.read_csv('heart_attack_germany.csv')

st.title("Germany Heart Attack Analysis Dashboard")

# Function: Heart Attack Incidence by Age Group
def heart_attack_by_age(data):
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='Age_Group', hue='Heart_Attack_Incidence', ax=ax)
    ax.set_title("Heart Attack Incidence by Age Group")
    return fig

# Function: Heart Attack Incidence by Gender
def heart_attack_by_gender(data):
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='Gender', hue='Heart_Attack_Incidence', ax=ax)
    ax.set_title("Heart Attack Incidence by Gender")
    return fig

# Function: BMI Distribution
def BMI_distribution(data):
    fig, ax = plt.subplots()
    sns.histplot(data=data, x='BMI', hue='Heart_Attack_Incidence', kde=True, bins=30, ax=ax)
    ax.set_title("Distribution of BMI")
    return fig

# Function: Heart Attack Trends Over the Years
def heart_attack_trends(data):
    fig, ax = plt.subplots()
    yearly_data = data.groupby('Year')['Heart_Attack_Incidence'].sum().reset_index()
    sns.lineplot(data=yearly_data, x='Year', y='Heart_Attack_Incidence', ax=ax)
    ax.set_title("Heart Attack Trends Over the Years")
    return fig

# Function: Regional Heart Attack Rates
def regional_heart_attack_rates(data):
    fig, ax = plt.subplots()
    regional_data = data.groupby('State')['Heart_Attack_Incidence'].mean().reset_index()
    sns.barplot(data=regional_data, x='State', y='Heart_Attack_Incidence', ax=ax)
    ax.set_title("Regional Heart Attack Rates")
    return fig

# Function: Cholesterol Levels vs Heart Attack Incidence
def cholesterol_vs_heart_attack(data):
    fig, ax = plt.subplots()
    sns.boxplot(data=data, x='Heart_Attack_Incidence', y='Cholesterol_Level', ax=ax)
    ax.set_title("Cholesterol Levels vs Heart Attack Incidence")
    return fig

# Function: Alcohol Consumption and Heart Attack Risk
def alcohol_and_heart_attack(data):
    fig, ax = plt.subplots()
    sns.violinplot(data=data, x='Heart_Attack_Incidence', y='Alcohol_Consumption', ax=ax)
    ax.set_title("Alcohol Consumption and Heart Attack Risk")
    return fig

# Display graphs in a 3-column layout
st.subheader("Explore the Data with Visualizations")

# Row 1
col1, col2, col3 = st.columns(3)
with col1:
    st.pyplot(heart_attack_by_age(data))
with col2:
    st.pyplot(heart_attack_by_gender(data))
with col3:
    st.pyplot(BMI_distribution(data))

# Row 2
col4, col5, col6 = st.columns(3)
with col4:
    st.pyplot(heart_attack_trends(data))
with col5:
    st.pyplot(regional_heart_attack_rates(data))
with col6:
    st.pyplot(cholesterol_vs_heart_attack(data))

# Row 3
col7, col8, col9= st.columns(3) 
with col8:
    st.pyplot(alcohol_and_heart_attack(data))



# Load and preprocess the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('heart_attack_germany.csv')

    categorical_columns = [
        'State', 'Age_Group', 'Gender', 'Smoking_Status', 'Physical_Activity_Level',
        'Diet_Quality', 'Urban_Rural', 'Socioeconomic_Status', 'Stress_Level',
        'Healthcare_Access', 'Education_Level', 'Employment_Status'
    ]
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    numerical_columns = [
        'BMI', 'Alcohol_Consumption', 'Cholesterol_Level', 'Air_Pollution_Index',
        'Region_Heart_Attack_Rate'
    ]
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    return data

# Training and Evaluation
@st.cache_resource
def train_model(data):
    x = data.drop(columns=['Heart_Attack_Incidence'])
    y = data['Heart_Attack_Incidence']

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(x_train, y_train)

    # Predictions and probabilities
    y_predict = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)[:, 1]

    return model, x_test, y_test, y_predict, y_pred_prob

# Visualization: Confusion Matrix
def plot_confusion_matrix(y_test, y_predict):
    """Plots the confusion matrix."""
    conf_matrix = confusion_matrix(y_test, y_predict)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["No Heart Attack", "Heart Attack"], 
                yticklabels=["No Heart Attack", "Heart Attack"], ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    return fig

# Visualization: ROC Curve
def plot_roc_curve(y_test, y_pred_prob):
    """Plots the ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="darkorange", lw=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    return fig

# Visualization: Feature Importance
def plot_feature_importance(model, feature_names):
    """Plots feature importance from model coefficients."""
    coefficients = model.coef_[0]
    feature_importance = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})
    feature_importance = feature_importance.sort_values(by="Coefficient", key=abs, ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(data=feature_importance, x="Coefficient", y="Feature", palette="viridis", ax=ax)
    ax.set_title("Feature Importance")
    ax.axvline(0, color="gray", linestyle="--")
    return fig


data = load_data()
model, x_test, y_test, y_predict, y_pred_prob = train_model(data)

st.title("Heart Attack Prediction")
st.subheader("Evaluation Metrics and Insights")

col1,col2 = st.columns(2)
with col1:
   st.subheader("Confusion Matrix")
   conf_matrix_fig = plot_confusion_matrix(y_test, y_predict)
   st.pyplot(conf_matrix_fig)

with col2:
   st.subheader("ROC Curve")
   roc_curve_fig = plot_roc_curve(y_test, y_pred_prob)
   st.pyplot(roc_curve_fig)

col3,col4 = st.columns(2)
with col3:
   st.subheader("Feature Importance")
   feature_importance_fig = plot_feature_importance(model, x_test.columns)
   st.pyplot(feature_importance_fig)


