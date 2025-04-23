
pip install pandas scikit-learn openpyxl

"""# **Load the dataset**"""

import pandas as pd

# Load the dataset
from google.colab import files
uploaded = files.upload()

"""# **Read the dataset**"""

data = pd.read_csv('HRMS CSV.csv')

print("Dataset Shape:", data.shape)
print(data.info())

"""# **Check the dataset and entries provided using relevant graphs**"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('HRMS CSV.csv')

# Data Exploration
print(df.head())
print(df.info())
print(df.describe())

# Data Cleaning
df['Health Risk Score'].fillna(0, inplace=True)

# Bar Plot: Distribution of Health Conditions
health_conditions = df[['Asthma', 'Heart Disease', 'Hyper Tension', 'Diabetes']].apply(pd.value_counts).fillna(0)
# Convert 'Yes' and 'No' to 1 and 0 respectively for plotting
health_conditions = health_conditions.replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, True: 1, False: 0})

health_conditions.plot(kind='bar', figsize=(10, 6))
plt.title('Distribution of Health Conditions')
plt.xlabel('Health Condition')
plt.ylabel('Count')
plt.show()

# Histogram: Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Box Plot: Cholesterol Level by Gender
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Cholesterol Level', data=df)
plt.title('Cholesterol Level by Gender')
plt.xlabel('Gender')
plt.ylabel('Cholesterol Level')
plt.show()

# Heatmap: Correlation Matrix
# Select only numerical features for correlation calculation
numerical_df = df.select_dtypes(include=['number'])
corr = numerical_df.corr()  # Calculate correlation for numerical features only

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Scatter Plot: Age vs. Blood Pressure
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Blood Pressure', hue='Gender', data=df)
plt.title('Age vs. Blood Pressure')
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.show()

# Count Plot: Breathing Problem by Gender
plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', hue='Breathing Problem', data=df)
plt.title('Breathing Problem by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

"""# **Basic Idea for the risk score calculations**"""

class HealthRiskCalculator:
    def __init__(self):
        # Original mappings
        self.gender_map = {
            'male': 'M',
            'female': 'F'
        }

        self.bp_map = {
            'low': (90, 60),
            'normal': (120, 80),
            'high': (140, 90)
        }

        self.cholesterol_map = {
            'low': 150,
            'normal': 200,
            'high': 240
        }

        # New lifestyle factor weights
        self.lifestyle_weights = {
            'smoking': 15,
            'alcohol': 10,
            'physical_activity': -10,  # Negative because it reduces risk
            'family_history': 10,
            'surgery': 5
        }

    def get_user_input(self):
        """Get all required inputs from user with validation"""
        print("\n=== Health Risk Assessment Form ===\n")

        # Get age
        while True:
            try:
                age = int(input("Enter age: "))
                if 0 <= age <= 120:
                    break
                print("Please enter a valid age between 0 and 120.")
            except ValueError:
                print("Please enter a valid number.")

        # Get city
        city = input("Enter your residing city: ").strip()

        # Get gender
        while True:
            gender = input("Enter gender (male/female): ").lower()
            if gender in self.gender_map:
                break
            print("Please enter either 'male' or 'female'.")

        # Get blood pressure level
        while True:
            bp = input("Enter blood pressure level (low/normal/high): ").lower()
            if bp in self.bp_map:
                break
            print("Please enter 'low', 'normal', or 'high'.")

        # Get cholesterol level
        while True:
            cholesterol = input("Enter cholesterol level (low/normal/high): ").lower()
            if cholesterol in self.cholesterol_map:
                break
            print("Please enter 'low', 'normal', or 'high'.")

        # Get yes/no conditions
        conditions = [
            "asthma",
            "heart disease",
            "hypertension",
            "diabetes",
            "breathing problem",
            "smoking habit",
            "alcoholic habit",
            "regular physical activity",
            "family history of chronic disease",
            "previous surgery"
        ]

        condition_values = {}
        for condition in conditions:
            while True:
                response = input(f"Do you have {condition}? (yes/no): ").lower()
                if response in ['yes', 'no']:
                    condition_values[condition] = (response == 'yes')
                    break
                print("Please enter 'yes' or 'no'.")

        # Prepare data dictionary
        patient_data = {
            'age': age,
            'city': city,
            'gender': self.gender_map[gender],
            'blood_pressure': self.bp_map[bp],
            'cholesterol': self.cholesterol_map[cholesterol],
            'asthma': condition_values['asthma'],
            'heart_disease': condition_values['heart disease'],
            'hypertension': condition_values['hypertension'],
            'diabetes': condition_values['diabetes'],
            'breathing_problem': condition_values['breathing problem'],
            'smoking': condition_values['smoking habit'],
            'alcohol': condition_values['alcoholic habit'],
            'physical_activity': condition_values['regular physical activity'],
            'family_history': condition_values['family history of chronic disease'],
            'surgery': condition_values['previous surgery']
        }

        return patient_data

    def calculate_risk_score(self, patient_data):
        """Calculate risk score based on input data"""
        risk_score = 0

        # Age scoring
        age = patient_data['age']
        if age < 30:
            risk_score += 5
        elif age < 45:
            risk_score += 10
        elif age < 60:
            risk_score += 15
        else:
            risk_score += 25

        # Gender scoring
        if patient_data['gender'] == 'M':
            risk_score += 5

        # Medical condition scoring
        condition_weights = {
            'asthma': 10,
            'heart_disease': 15,
            'hypertension': 10,
            'diabetes': 10,
            'breathing_problem': 10
        }

        for condition, weight in condition_weights.items():
            if patient_data[condition]:
                risk_score += weight

        # Lifestyle factor scoring
        if patient_data['smoking']:
            risk_score += self.lifestyle_weights['smoking']
        if patient_data['alcohol']:
            risk_score += self.lifestyle_weights['alcohol']
        if patient_data['physical_activity']:
            risk_score += self.lifestyle_weights['physical_activity']  # Reduces risk
        if patient_data['family_history']:
            risk_score += self.lifestyle_weights['family_history']
        if patient_data['surgery']:
            risk_score += self.lifestyle_weights['surgery']

        # Blood pressure scoring
        systolic, _ = patient_data['blood_pressure']
        if systolic >= 140:
            risk_score += 15
        elif systolic >= 120:
            risk_score += 10
        elif systolic >= 90:
            risk_score += 5

        # Cholesterol scoring
        if patient_data['cholesterol'] >= 240:
            risk_score += 15
        elif patient_data['cholesterol'] >= 200:
            risk_score += 10
        elif patient_data['cholesterol'] >= 170:
            risk_score += 5

        # Ensure score stays within 0-100 range
        return max(0, min(100, risk_score))

    def get_risk_level(self, risk_score):
        """Determine risk level based on score"""
        if risk_score >= 75:
            return "Very High Risk"
        elif risk_score >= 50:
            return "High Risk"
        elif risk_score >= 25:
            return "Moderate Risk"
        else:
            return "Low Risk"

    def get_recommendations(self, patient_data, risk_level):
        """Generate recommendations based on risk factors"""
        recommendations = []

        if risk_level in ["High Risk", "Very High Risk"]:
            recommendations.append("Please consult a healthcare provider for a detailed evaluation.")

        if patient_data['smoking']:
            recommendations.append("Consider smoking cessation programs")
        if patient_data['alcohol']:
            recommendations.append("Consider reducing alcohol consumption")
        if not patient_data['physical_activity']:
            recommendations.append("Start regular physical activity (consult doctor before starting)")

        return recommendations

    def display_results(self, patient_data, risk_score, risk_level):
        """Display the assessment results"""
        print("\n=== Health Risk Assessment Results ===\n")
        print(f"Risk Score: {risk_score}/100")
        print(f"Risk Level: {risk_level}")
        print("\nPersonal Information:")
        print(f"Age: {patient_data['age']}")
        print(f"City: {patient_data['city']}")
        print(f"Gender: {'Male' if patient_data['gender'] == 'M' else 'Female'}")
        print(f"Blood Pressure: {next(k for k, v in self.bp_map.items() if v == patient_data['blood_pressure']).title()}")
        print(f"Cholesterol: {next(k for k, v in self.cholesterol_map.items() if v == patient_data['cholesterol']).title()}")

        print("\nHealth Conditions:")
        conditions = ['asthma', 'heart_disease', 'hypertension',
                     'diabetes', 'breathing_problem']
        for condition in conditions:
            print(f"{condition.title()}: {'Yes' if patient_data[condition] else 'No'}")

        print("\nLifestyle Factors:")
        print(f"Smoking: {'Yes' if patient_data['smoking'] else 'No'}")
        print(f"Alcohol Consumption: {'Yes' if patient_data['alcohol'] else 'No'}")
        print(f"Regular Physical Activity: {'Yes' if patient_data['physical_activity'] else 'No'}")
        print(f"Family History of Chronic Disease: {'Yes' if patient_data['family_history'] else 'No'}")
        print(f"Previous Surgery: {'Yes' if patient_data['surgery'] else 'No'}")

        recommendations = self.get_recommendations(patient_data, risk_level)
        if recommendations:
            print("\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")

def main():
    calculator = HealthRiskCalculator()

    print("Welcome to the Health Risk Calculator!")

    while True:
        patient_data = calculator.get_user_input()
        risk_score = calculator.calculate_risk_score(patient_data)
        risk_level = calculator.get_risk_level(risk_score)
        calculator.display_results(patient_data, risk_score, risk_level)

        while True:
            again = input("\nWould you like to calculate another risk score? (yes/no): ").lower()
            if again in ['yes', 'no']:
                break
            print("Please enter 'yes' or 'no'.")

        if again == 'no':
            print("\nThank you for using the Health Risk Calculator!")
            break

if __name__ == "__main__":
    main()

"""## **Train the dataset**"""

from sklearn.model_selection import train_test_split

# Define features and target variable
X = data.drop(columns=['Health Risk Score'])   # Features: all columns except target score.
y = data['Health Risk Score']                    # Target: Health Risk Score.

# Convert categorical variables to numerical format using one-hot encoding or label encoding as necessary.
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train, X_test, y_train, y_test)

# Preprocessing
binary_columns = ['Asthma', 'Heart Disease', 'Hyper Tension', 'Diabetes', 'Breathing Problem']
data[binary_columns] = data[binary_columns].fillna(0)

if 'Age' in data.columns:
    data['Age'] = data['Age'].fillna(data['Age'].mean())

data[binary_columns] = data[binary_columns].replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, True: 1, False: 0})
data[binary_columns] = data[binary_columns].astype(int)

if 'Gender' in data.columns:
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1}).fillna(0).astype(int)

data['Health Risk Score'] = (
    (data['Age'] / 10) +
    (data['Asthma'] * 2) +
    (data['Heart Disease'] * 3) +
    (data['Hyper Tension'] * 2) +
    (data['Diabetes'] * 4) +
    (data['Breathing Problem'] * 2)
)

# Prepare the dataset for modeling
X = data.drop(columns=['Health Risk Score'])
y = data['Health Risk Score']

# Handle categorical columns with one-hot encoding
categorical_columns = X.select_dtypes(include=['object']).columns
if not categorical_columns.empty:
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

"""# **ML model implementation using XGBoost**"""

import xgboost as xgb
from sklearn.model_selection import train_test_split

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost Regressor
model = xgb.XGBRegressor(random_state=42)
model.fit(X_train, y_train)

"""# **Calculate the accuracy, precision etc. for the ML Model**"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, learning_curve

# 1. Basic Evaluation Metrics
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics for train data
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Calculate metrics for test data
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("Model Performance:")
    print("------------------")
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print("\n")
    print(f"Testing RMSE: {test_rmse:.2f}")
    print(f"Testing MAE: {test_mae:.2f}")
    print(f"Testing R²: {test_r2:.4f}")

    # Check for overfitting
    print("\nOverfitting Check:")
    print(f"RMSE difference (train-test): {train_rmse - test_rmse:.2f}")
    print(f"R² difference (train-test): {train_r2 - test_r2:.4f}")

    return {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'y_test': y_test,
        'y_test_pred': y_test_pred
    }

# 2. Cross-Validation
def perform_cross_validation(model, X, y, cv=5):
    print("\nCross-Validation Results:")
    print("-----------------------")
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    print(f"CV RMSE Scores: {-cv_scores}")
    print(f"Mean CV RMSE: {-cv_scores.mean():.2f}")
    print(f"Std Dev CV RMSE: {cv_scores.std():.2f}")

    cv_r2 = cross_val_score(model, X, y, cv=cv, scoring='r2')
    print(f"CV R² Scores: {cv_r2}")
    print(f"Mean CV R²: {cv_r2.mean():.4f}")
    print(f"Std Dev CV R²: {cv_r2.std():.4f}")

# 3. Feature Importance
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

    print("\nFeature Importance:")
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

# 4. Prediction vs Actual Plot
def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)

    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs Actual')
    plt.grid(True)
    plt.show()

# 5. Learning Curve to check for overfitting
def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title("Learning Curve")
    plt.xlabel("Training Examples")
    plt.ylabel("R² Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

# Run all evaluations
feature_names = X.columns  # Make sure to define this based on your dataframe
results = evaluate_model(model, X_train, X_test, y_train, y_test)
perform_cross_validation(model, X, y)
plot_feature_importance(model, feature_names)
plot_predictions(results['y_test'], results['y_test_pred'])
plot_learning_curve(model, X, y)

"""# **Creating interactive Interface for the user to access the risk score calculator**"""

pip install gradio

!pip install reportlab

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

import os
import gradio as gr

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

class HealthRiskCalculator:
    def __init__(self):
        # Original mappings
        self.gender_map = {
            'male': 'M',
            'female': 'F'
        }

        self.bp_map = {
            'low': (90, 60),
            'normal': (120, 80),
            'high': (140, 90)
        }

        self.cholesterol_map = {
            'low': 150,
            'normal': 200,
            'high': 240
        }

        # New lifestyle factor weights
        self.lifestyle_weights = {
            'smoking': 15,
            'alcohol': 10,
            'physical_activity': -10,  # Negative because it reduces risk
            'family_history': 10,
            'surgery': 5
        }

    def process_gradio_input(self, name, age, city, gender, bp, cholesterol, asthma, heart_disease,
                           hypertension, diabetes, breathing_problem, smoking,
                           alcohol, physical_activity, family_history, surgery):
        """Process inputs from Gradio interface"""

        # Convert yes/no strings to boolean
        convert_to_bool = lambda x: x.lower() == 'yes'

        # Prepare data dictionary
        patient_data = {
            'name': name,
            'age': age,
            'city': city,
            'gender': self.gender_map[gender.lower()],
            'blood_pressure': self.bp_map[bp.lower()],
            'cholesterol': self.cholesterol_map[cholesterol.lower()],
            'bp_label': bp,  # Store original label directly
            'cholesterol_label': cholesterol,  # Store original label directly
            'asthma': convert_to_bool(asthma),
            'heart_disease': convert_to_bool(heart_disease),
            'hypertension': convert_to_bool(hypertension),
            'diabetes': convert_to_bool(diabetes),
            'breathing_problem': convert_to_bool(breathing_problem),
            'smoking': convert_to_bool(smoking),
            'alcohol': convert_to_bool(alcohol),
            'physical_activity': convert_to_bool(physical_activity),
            'family_history': convert_to_bool(family_history),
            'surgery': convert_to_bool(surgery)
        }

        return patient_data

    def calculate_risk_score(self, patient_data):
        """Calculate risk score based on input data"""
        risk_score = 0

        # Age scoring
        age = patient_data['age']
        if age < 30:
            risk_score += 5
        elif age < 45:
            risk_score += 10
        elif age < 60:
            risk_score += 15
        else:
            risk_score += 25

        # Gender scoring
        if patient_data['gender'] == 'M':
            risk_score += 5

        # Medical condition scoring
        condition_weights = {
            'asthma': 10,
            'heart_disease': 15,
            'hypertension': 10,
            'diabetes': 10,
            'breathing_problem': 10
        }

        for condition, weight in condition_weights.items():
            if patient_data[condition]:
                risk_score += weight

        # Lifestyle factor scoring
        if patient_data['smoking']:
            risk_score += self.lifestyle_weights['smoking']
        if patient_data['alcohol']:
            risk_score += self.lifestyle_weights['alcohol']
        if patient_data['physical_activity']:
            risk_score += self.lifestyle_weights['physical_activity']  # Reduces risk
        if patient_data['family_history']:
            risk_score += self.lifestyle_weights['family_history']
        if patient_data['surgery']:
            risk_score += self.lifestyle_weights['surgery']

        # Blood pressure scoring
        systolic, _ = patient_data['blood_pressure']
        if systolic >= 140:
            risk_score += 15
        elif systolic >= 120:
            risk_score += 10
        elif systolic >= 90:
            risk_score += 5

        # Cholesterol scoring
        if patient_data['cholesterol'] >= 240:
            risk_score += 15
        elif patient_data['cholesterol'] >= 200:
            risk_score += 10
        elif patient_data['cholesterol'] >= 170:
            risk_score += 5

        # Ensure score stays within 0-100 range
        return max(0, min(100, risk_score))

    def get_risk_level(self, risk_score):
        """Determine risk level based on score"""
        if risk_score >= 75:
            return "Very High Risk"
        elif risk_score >= 50:
            return "High Risk"
        elif risk_score >= 25:
            return "Moderate Risk"
        else:
            return "Low Risk"

    def get_recommendations(self, patient_data, risk_level):
        """Generate recommendations based on risk factors and risk level"""
        recommendations = []

        # Recommendations based on risk level
        if risk_level == "Very High Risk":
            recommendations.append("Urgent: Schedule an appointment with your healthcare provider within the next 7 days")
            recommendations.append("Consider more frequent health monitoring")
        elif risk_level == "High Risk":
            recommendations.append("Please consult a healthcare provider for a detailed evaluation within 30 days")
            recommendations.append("Implement lifestyle changes immediately")
        elif risk_level == "Moderate Risk":
            recommendations.append("Schedule a regular check-up with your healthcare provider")
            recommendations.append("Consider making gradual lifestyle improvements")
        else:  # Low Risk
            recommendations.append("Continue your healthy habits")
            recommendations.append("Schedule routine annual check-ups")

        # Lifestyle specific recommendations
        if patient_data['smoking']:
            recommendations.append("Consider smoking cessation programs or nicotine replacement therapy")
        else:
            recommendations.append("Continue to avoid tobacco products")

        if patient_data['alcohol']:
            recommendations.append("Consider reducing alcohol consumption to recommended levels")
        else:
            recommendations.append("Continue to limit or avoid alcohol consumption")

        if patient_data['physical_activity']:
            recommendations.append("Maintain your regular exercise routine of at least 150 minutes per week")
        else:
            recommendations.append("Start regular physical activity (consult doctor before starting)")

        # Nutrition recommendations
        recommendations.append("Maintain a balanced diet rich in fruits, vegetables, and whole grains")

        # Blood pressure recommendations
        bp_label = patient_data.get('bp_label', '').lower()
        if bp_label == 'high':
            recommendations.append("Monitor your blood pressure regularly and consider dietary changes")
        elif bp_label == 'low':
            recommendations.append("Monitor your blood pressure and stay hydrated")
        else:  # normal
            recommendations.append("Continue to monitor your blood pressure annually")

        # Cholesterol recommendations
        cholesterol_label = patient_data.get('cholesterol_label', '').lower()
        if cholesterol_label == 'high':
            recommendations.append("Consider dietary changes to reduce cholesterol and follow up with your doctor")
        elif cholesterol_label == 'normal' or cholesterol_label == 'low':
            recommendations.append("Continue to monitor your cholesterol levels annually")

        # Age-based recommendations
        age = patient_data.get('age', 0)
        if age >= 45:
            recommendations.append("Consider more frequent screenings for age-related conditions")

        # Mental health recommendation
        recommendations.append("Incorporate stress management techniques into your daily routine")

        # Return only unique recommendations
        return list(set(recommendations))

    def generate_pdf_report(self, patient_data):
        """Generate a PDF report of the risk assessment with watermark and footer"""
        import io
        import os
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader

        # Calculate risk score and level
        risk_score = self.calculate_risk_score(patient_data)
        risk_level = self.get_risk_level(risk_score)
        recommendations = self.get_recommendations(patient_data, risk_level)

        # Create a buffer for the PDF
        buffer = io.BytesIO()

        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter, title="Health Risk Assessment Report")
        styles = getSampleStyleSheet()
        elements = []

        # Create custom styles for the report
        styles.add(ParagraphStyle(name='ReportTitle',
                                parent=styles['Heading1'],
                                fontSize=16,
                                alignment=1,
                                spaceAfter=12,
                                underline=True))

        styles.add(ParagraphStyle(name='SectionHeader',
                                parent=styles['Heading2'],
                                fontSize=12,
                                underline=True,
                                spaceAfter=6))

        styles.add(ParagraphStyle(name='NormalText',
                                parent=styles['Normal'],
                                fontSize=11,
                                spaceAfter=6))

        styles.add(ParagraphStyle(name='Footer',
                                parent=styles['Normal'],
                                fontSize=8,
                                alignment=1))  # Centered

        # Add the report title
        elements.append(Paragraph("Health Risk Assessment Report", styles['ReportTitle']))
        elements.append(Spacer(1, 0.2*inch))

        # Add patient details
        patient_info = [
            ("Name:", patient_data.get('name', '')),
            ("Age:", patient_data.get('age', '')),
            ("Residing City:", patient_data.get('city', '')),
            ("Gender:", patient_data.get('gender', ''))
        ]

        for label, value in patient_info:
            elements.append(Paragraph(f"<b>{label}</b> {value}", styles['NormalText']))

        elements.append(Spacer(1, 0.2*inch))

        # Lifestyle Habits section
        elements.append(Paragraph("Lifestyle Habit:", styles['SectionHeader']))

        # Process lifestyle data
        lifestyle_text = ""
        if patient_data.get('smoking', False):
            lifestyle_text += f"Smoking: Yes<br/>"
        else:
            lifestyle_text += f"Smoking: No<br/>"

        if patient_data.get('physical_activity', False):
            lifestyle_text += f"Exercise: Regular<br/>"
        else:
            lifestyle_text += f"Exercise: None<br/>"

        if patient_data.get('alcohol', False):
            lifestyle_text += f"Alcohol Consumption: Yes<br/>"
        else:
            lifestyle_text += f"Alcohol Consumption: No<br/>"

        elements.append(Paragraph(lifestyle_text, styles['NormalText']))
        elements.append(Spacer(1, 0.2*inch))

        # Medical History section
        elements.append(Paragraph("Medical History:", styles['SectionHeader']))

        # Process medical history data
        medical_text = ""

        # Chronic conditions
        chronic_conditions = []
        if patient_data.get('asthma', False):
            chronic_conditions.append("Asthma")
        if patient_data.get('heart_disease', False):
            chronic_conditions.append("Heart Disease")
        if patient_data.get('hypertension', False):
            chronic_conditions.append("Hypertension")
        if patient_data.get('diabetes', False):
            chronic_conditions.append("Diabetes")
        if patient_data.get('breathing_problem', False):
            chronic_conditions.append("Breathing Problems")

        if chronic_conditions:
            medical_text += f"Chronic Conditions: {', '.join(chronic_conditions)}<br/>"
        else:
            medical_text += f"Chronic Conditions: None<br/>"

        # Family history
        if patient_data.get('family_history', False):
            medical_text += f"Family History: Positive for chronic diseases<br/>"
        else:
            medical_text += f"Family History: No significant history<br/>"

        # Surgery history
        if patient_data.get('surgery', False):
            medical_text += f"Previous Surgery: Yes<br/>"
        else:
            medical_text += f"Previous Surgery: No<br/>"

        # Blood pressure and cholesterol
        medical_text += f"Blood Pressure: {patient_data.get('bp_label', 'Not provided')}<br/>"
        medical_text += f"Cholesterol: {patient_data.get('cholesterol_label', 'Not provided')}<br/>"

        elements.append(Paragraph(medical_text, styles['NormalText']))
        elements.append(Spacer(1, 0.2*inch))

        # Risk Score
        elements.append(Paragraph(f"<b>Risk Score:</b> {risk_score} ({risk_level})", styles['NormalText']))
        elements.append(Spacer(1, 0.1*inch))

        # Recommendations
        elements.append(Paragraph("<b>Recommendations:</b>", styles['SectionHeader']))
        for recommendation in recommendations:
            elements.append(Paragraph(f"• {recommendation}", styles['NormalText']))

        # Custom canvas for watermark and footer
        class WatermarkAndFooterCanvas(canvas.Canvas):
            def __init__(self, *args, **kwargs):
                canvas.Canvas.__init__(self, *args, **kwargs)
                self.pages = []

            def showPage(self):
                self.pages.append(dict(self.__dict__))
                self._startPage()

            def save(self):
                page_count = len(self.pages)
                for page in self.pages:
                    self.__dict__.update(page)
                    self._drawWatermark()
                    self._drawFooter(page_count)
                    canvas.Canvas.showPage(self)
                canvas.Canvas.save(self)

            def _drawWatermark(self):
                # Save the current state
                self.saveState()

                # Center the watermark on the page
                page_width, page_height = letter

                # Set transparency
                self.setFillAlpha(0.1)  # 10% opacity

                # Try to load the logo image
                try:
                    # Draw the watermark in the center of the page
                    logo_path = "/content/drive/MyDrive/Gohn_s_bakery_studio-removebg-preview.png"
                    if os.path.exists(logo_path):
                        logo = ImageReader(logo_path)
                        logo_width, logo_height = 200, 200  # Adjust size as needed
                        self.drawImage(logo,
                            (page_width - logo_width) / 2,
                            (page_height - logo_height) / 2,
                            width=logo_width,
                            height=logo_height,
                            mask='auto')
                except Exception as e:
                    # If logo loading fails, use text watermark instead
                    self.setFont("Helvetica", 40)
                    self.setFillColorRGB(0.8, 0.8, 0.8)  # Light gray
                    self.drawCentredString(page_width/2, page_height/2, "HEALTH RISK REPORT")

                # Restore the state
                self.restoreState()

            def _drawFooter(self, page_count):
                self.saveState()

                # Footer text
                footer_text = "Health Risk Management System (HRMS) © 2025 | Confidential Medical Information"
                page_num_text = f"Page {self._pageNumber} of {page_count}"

                page_width, page_height = letter

                # Draw the footer
                self.setFont("Helvetica", 8)
                self.setFillColorRGB(0, 0, 0)  # Black text

                # Draw footer text and page number
                self.drawCentredString(page_width/2, 0.5*inch, footer_text)
                self.drawCentredString(page_width/2, 0.3*inch, page_num_text)

                # Add a line above the footer
                self.setLineWidth(0.5)
                self.line(0.5*inch, 0.75*inch, page_width - 0.5*inch, 0.75*inch)

                self.restoreState()

        # Build the PDF with custom canvas
        doc.build(elements, canvasmaker=WatermarkAndFooterCanvas)
        buffer.seek(0)

        return buffer
import base64

def gradio_risk_assessment(name, age, city, gender, bp, cholesterol, asthma, heart_disease,
                         hypertension, diabetes, breathing_problem, smoking,
                         alcohol, physical_activity, family_history, surgery):
    # Rest of the function remains the same
    """Function to handle Gradio interface"""
    try:
        calculator = HealthRiskCalculator()

        # Process inputs
        patient_data = calculator.process_gradio_input(
            name, age, city, gender, bp, cholesterol, asthma,
            heart_disease, hypertension, diabetes, breathing_problem,
            smoking, alcohol, physical_activity, family_history, surgery
        )

        # Generate PDF report
        pdf_buffer = calculator.generate_pdf_report(patient_data)

        # Get risk score and level for display
        risk_score = calculator.calculate_risk_score(patient_data)
        risk_level = calculator.get_risk_level(risk_score)

        # Create base64 data for download link
        pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
        download_link = f'<a href="data:application/pdf;base64,{pdf_base64}" download="health_risk_report.pdf" class="download-button">Download PDF Report</a>'

        # Create a summary HTML with black text
        html_result = f"""
        <div style="padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: black; color: white;">
            <h3>Health Risk Assessment Summary</h3>
            <p><strong>Risk Score:</strong> {risk_score}/100</p>
            <p><strong>Risk Level:</strong> {risk_level}</p>
            <p>A comprehensive report has been generated with detailed analysis and recommendations.</p>
            <div style="margin-top: 15px;">
                {download_link}
            </div>
        </div>
        <style>
            .download-button {{
                display: inline-block;
                padding: 10px 15px;
                background-color: #000000;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                font-weight: bold;
            }}
            .download-button:hover {{
                background-color: #333333;
            }}
        </style>
        """

        return html_result
    except Exception as e:
        # Return an error message to the user
        return f"""
        <div style="padding: 15px; border: 1px solid #f8d7da; border-radius: 5px; background-color: #f8d7da; color: #721c24;">
            <h3>Error</h3>
            <p>An error occurred while processing your health risk assessment: {str(e)}</p>
            <p>Please try again or contact support if the issue persists.</p>
        </div>
        """

# Create the Gradio Interface
def create_interface():
    with gr.Blocks(title="Health Risk Assessment") as demo:
        gr.HTML("<h1>Health Risk Management System</h1>")
        gr.HTML("<p>Enter your information below to calculate your health risk and generate a detailed PDF report.</p>")

        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>Personal Details</h3>")
                name = gr.Textbox(label="Name", value="Vardhan")
                age = gr.Number(label="Age", value=30, minimum=0, maximum=120)
                city = gr.Textbox(label="City", value="New York")
                gender = gr.Radio(label="Gender", choices=["Male", "Female"], value="Male")

            with gr.Column(scale=1):
                # Health conditions
                gr.HTML("<h3>Health Conditions</h3>")
                asthma = gr.Radio(label="Asthma", choices=["Yes", "No"], value="No")
                heart_disease = gr.Radio(label="Heart Disease", choices=["Yes", "No"], value="No")
                hypertension = gr.Radio(label="Hypertension", choices=["Yes", "No"], value="No")
                diabetes = gr.Radio(label="Diabetes", choices=["Yes", "No"], value="No")
                breathing_problem = gr.Radio(label="Breathing Problem", choices=["Yes", "No"], value="No")
                bp = gr.Radio(label="Blood Pressure Level", choices=["Low", "Normal", "High"], value="Normal")
                cholesterol = gr.Radio(label="Cholesterol Level", choices=["Low", "Normal", "High"], value="Normal")


            with gr.Column(scale=1):
                # Lifestyle factors
                gr.HTML("<h3>Lifestyle Factors</h3>")
                smoking = gr.Radio(label="Smoking Habit", choices=["Yes", "No"], value="No")
                alcohol = gr.Radio(label="Alcohol Consumption", choices=["Yes", "No"], value="No")
                physical_activity = gr.Radio(label="Regular Physical Activity", choices=["Yes", "No"], value="Yes")
                family_history = gr.Radio(label="Family History of Chronic Disease", choices=["Yes", "No"], value="No")
                surgery = gr.Radio(label="Previous Surgery", choices=["Yes", "No"], value="No")

        with gr.Row():
            submit_button = gr.Button("Calculate Risk Score and Generate Report", variant="primary")

        with gr.Row():
            output = gr.HTML()

        submit_button.click(
            fn=gradio_risk_assessment,
            inputs=[
                name, age, city, gender, bp, cholesterol,
                asthma, heart_disease, hypertension, diabetes, breathing_problem,
                smoking, alcohol, physical_activity, family_history, surgery
            ],
            outputs=output
        )

    return demo

# For running in Colab
if __name__ == "__main__":
    # Install required packages if in Colab
    try:
        import google.colab
        !pip install gradio reportlab matplotlib pandas
    except:
        pass

    # Create and launch the interface
    demo = create_interface()
    demo.launch(share=True)
