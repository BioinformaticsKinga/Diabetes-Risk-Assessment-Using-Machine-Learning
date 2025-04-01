import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'diabetes_data.csv'
data = pd.read_csv(file_path)

# Handle missing values (mean imputation for numerical columns)
imputer = SimpleImputer(strategy='mean')
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

# Encode categorical variables
label_encoders = {}
categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column].astype(str))
    label_encoders[column] = encoder

# Save the processed data to a file
output_path = 'processed_diabetes_data.csv'
data.to_csv(output_path, index=False)

print("Processing complete. Data saved to file.")
