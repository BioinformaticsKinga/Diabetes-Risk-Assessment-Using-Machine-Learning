import pandas as pd
import plotly.express as px

df = pd.read_csv('processed_data_diabetes_zyg.csv')

# Visualizing the distribution of the target variable (Outcome)
vis = px.bar(df, x='Outcome', title="Distribution of Outcome Variable")
vis.show()

# Matrix Correlation
fig_corr = px.imshow(df.corr(), text_auto=True, title="Correlation Matrix")
fig_corr.show()
