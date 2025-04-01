#This is the preparation of data for further analysis

import pandas as pd
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import LabelEncoder

file_path = 'diabetes_data_zyg.csv'
df = pd.read_csv(file_path)

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

