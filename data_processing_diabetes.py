import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'diabetes_data.csv'
df = pd.read_csv(file_path)

# Impute missing data
imputer = SimpleImputer(strategy='mean')
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = imputer.fit_transform(df[num_cols])

# Encoding categorical variables
label_encoders = {}
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Saving processed data to a new file
df.to_csv('processed_diabetes_data.csv', index=False)

print("Preprocessing complete, data saved to file.")
