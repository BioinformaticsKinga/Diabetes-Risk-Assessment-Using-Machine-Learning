import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('processed_diabetes_data_zyg.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# It is the best model
the_best_model = grid_search.best_estimator_

# It is prediction
y_the_best_prediction = best_knn_model.predict(X_test)

# It is model evaluation
print("Optimized Model Classification Report:\n", classification_report(y_test, y_the_best_prediction))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_the_best_prediction))

# It is cross-validation evaluation
cross_val_scores = cross_val_score(best_knn_model, X, y, cv=5)
print(f"Cross-validation Accuracy Scores: {cross_val_scores}")
print(f"Average Cross-validation Score: {cross_val_scores.mean()}")
