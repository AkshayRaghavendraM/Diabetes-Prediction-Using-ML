import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the diabetes dataset (replace with your actual path)
data = pd.read_csv("C:\\Users\\Asus\\OneDrive\\Documents\\Diabetes-prediction-using-ML\\diabetes.csv")

# Exploratory data analysis (optional)
# sns.heatmap(data.isnull())  # Check for missing values
# correlation = data.corr()
# print(correlation)
# sns.heatmap(correlation)  # Explore correlations

# Separate features (X) and target variable (Y)
X = data.drop("Outcome", axis=1)
Y = data["Outcome"]

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Function to get user input and make prediction
def predict_diabetes():
    print("Enter the following information:")
    preg = float(input("Pregnancies: "))
    glucose = float(input("Blood glucose level (mg/dL): "))
    pressure = float(input("Blood pressure (diastolic in mm Hg): "))
    skinthickness = float(input("Skin thickness (mm): "))
    insulin = float(input("Serum insulin (mu IU/mL): "))
    bmi = float(input("Body mass index (BMI): "))
    diabetespedigreefunction = float(input("Diabetes pedigree function: "))
    age = float(input("Age (years): "))

    # Create a DataFrame from user input
    user_data = pd.DataFrame({
        "Pregnancies": [preg],
        "Glucose": [glucose],
        "BloodPressure": [pressure],
        "SkinThickness": [skinthickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetespedigreefunction],
        "Age": [age]
    })

    # Make prediction using the trained model
    prediction = model.predict(user_data)[0]

    if prediction == 0:
        print("Predicted outcome: Non-diabetic (Negative)")
    else:
        print("Predicted outcome: Diabetic (Positive)")
        print("**Disclaimer:** This is a preliminary prediction and should not replace a medical diagnosis. Please consult a healthcare professional for a comprehensive assessment.")

# Call the prediction function
predict_diabetes()
