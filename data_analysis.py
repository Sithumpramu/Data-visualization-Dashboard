import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

file = 'heart_attack_germany.csv'
data = pd.read_csv(file)

print(data.head())
print(data.info())
print(data.describe().transpose())

null_values = data.isnull().sum()
print(f'null values count {null_values}')

duplicates = data.duplicated().sum()
print(f'No of duplicates : {duplicates}')

#replace negative values
cholesterol_median = data['Cholesterol_Level'][data['Cholesterol_Level'] > 0].median()
data.loc[data['Cholesterol_Level'] < 0, 'Cholesterol_Level'] = cholesterol_median
print("done replacing")

print(data.describe().transpose())

#visualize Data

#Heart Attack Incidence by Age Group
sns.countplot(data=data, x='Age_Group', hue='Heart_Attack_Incidence')
plt.title("Heart Attack Incidence by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.show()

#Heart Attack Incidence by Gender
sns.countplot(data=data, x='Gender', hue='Heart_Attack_Incidence')
plt.title("Heart Attack Incidence by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title='Heart Attack', labels=['No', 'Yes'])
plt.show()

# Distribution of BMI
sns.histplot(data=data, x='BMI', hue='Heart_Attack_Incidence', kde=True, bins=30)
plt.title("BMI Distribution and Heart Attack Incidence")
plt.xlabel("BMI")
plt.ylabel("Count")
plt.show()

# Heart Attack Trends Over the Years
yearly_data = data.groupby('Year')['Heart_Attack_Incidence'].sum().reset_index()
sns.lineplot(data=yearly_data, x='Year', y='Heart_Attack_Incidence')
plt.title("Heart Attack Trends Over the Years")
plt.xlabel("Year")
plt.ylabel("Heart Attack Cases")
plt.show()

#Regional Heart Attack Rates
regional_data = data.groupby('State')['Heart_Attack_Incidence'].mean().reset_index()
sns.barplot(data=regional_data, x='State', y='Heart_Attack_Incidence')
plt.title("Average Heart Attack Incidence by State")
plt.xticks(rotation=45)
plt.xlabel("State")
plt.ylabel("Heart Attack Incidence")
plt.show()

#Cholesterol Levels vs Heart Attack Incidence
sns.boxplot(data=data, x='Heart_Attack_Incidence', y='Cholesterol_Level')
plt.title("Cholesterol Levels by Heart Attack Incidence")
plt.xlabel("Heart Attack Incidence (0 = No, 1 = Yes)")
plt.ylabel("Cholesterol Level")
plt.show()

#Alcohol Consumption and Heart Attack Risk
sns.violinplot(data=data, x='Heart_Attack_Incidence', y='Alcohol_Consumption')
plt.title("Alcohol Consumption by Heart Attack Incidence")
plt.xlabel("Heart Attack Incidence (0 = No, 1 = Yes)")
plt.ylabel("Alcohol Consumption")
plt.show()


# Columns to encode
categorical_columns = [
    'State', 'Age_Group', 'Gender', 'Smoking_Status', 'Physical_Activity_Level',
    'Diet_Quality', 'Urban_Rural', 'Socioeconomic_Status', 'Stress_Level',
    'Healthcare_Access', 'Education_Level', 'Employment_Status'
]

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

#normalization
numerical_columns = [
    'BMI', 'Alcohol_Consumption', 'Cholesterol_Level', 'Air_Pollution_Index',
    'Region_Heart_Attack_Rate'
]

scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

x = data.drop(columns=['Heart_Attack_Incidence'])
y = data['Heart_Attack_Incidence']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LogisticRegression(max_iter=500 ,random_state=42)
model.fit(x_train,y_train)

y_predict = model.predict(x_test)
y_pred_prob = model.predict_proba(x_test)[:, 1]

#evaluation
print(classification_report(y_test, y_predict, target_names=["No Heart Attack", "Heart Attack"]))

conf_matrix = confusion_matrix(y_test, y_predict)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["No Heart Attack", "Heart Attack"], 
            yticklabels=["No Heart Attack", "Heart Attack"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="darkorange", lw=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()


# Feature Importance (Coefficients)
coefficients = model.coef_[0]
feature_importance = pd.DataFrame({"Feature": x.columns, "Coefficient": coefficients})
feature_importance = feature_importance.sort_values(by="Coefficient", key=abs, ascending=False)

sns.barplot(data=feature_importance, x="Coefficient", y="Feature", palette="viridis")
plt.title("Feature Importance")
plt.axvline(0, color="gray", linestyle="--")
plt.show()
