#This is the preparation of data for further analysis

import pandas as pd
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import LabelEncoder

# I downloaded date from website pima-indians-diabetes
data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

df = pd.read_csv(data_url, names=columns)

df.to_csv('diabetes_data_zyg.csv', index=False)

imputer = SimpleImputer(strategy='mean')
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = imputer.fit_transform(df[num_cols])

label_encoders = {}
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

df.to_csv('processed_data_diabetes_zyg.csv', index=False)
