import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the processed data
df = pd.read_csv('processed_diabetes_data.csv')

# Prepare features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model
best_knn_model = grid_search.best_estimator_

# Predictions
y_pred_best = best_knn_model.predict(X_test)

# Model Evaluation
print("Optimized Model Classification Report:\n", classification_report(y_test, y_pred_best))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))

# Cross-validation evaluation
cross_val_scores = cross_val_score(best_knn_model, X, y, cv=5)
print(f"Cross-validation Accuracy Scores: {cross_val_scores}")
print(f"Average Cross-validation Score: {cross_val_scores.mean()}")
