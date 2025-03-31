import pandas as pd
import plotly.express as px

# Load the processed data
df = pd.read_csv('processed_diabetes_data.csv')

# Visualizing the distribution of the target variable (Outcome)
fig = px.bar(df, x='Outcome', title="Distribution of Outcome Variable")
fig.show()

# Visualizing correlation matrix
fig_corr = px.imshow(df.corr(), text_auto=True, title="Correlation Matrix")
fig_corr.show()
