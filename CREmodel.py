import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the data
df_auroraprice = pd.read_csv('')

# Encode the categorical variable "Submarket"
label_encoder = LabelEncoder()
df_auroraprice['Submarket']= label_encoder.fit_transform(df_auroraprice['Submarket'])

# Create the features matrix
X = df_auroraprice.drop(columns=['Address', 'Lat.', 'Long.', 'Avg. Asking Rent Per Door'])

# Create the target vector
y = df_auroraprice['Avg. Asking Rent Per Door']

# Generate polynomial features
poly = PolynomialFeatures(2)
X_poly = poly.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2)

# Define the model
model = DecisionTreeRegressor()

# Define the hyperparameters to tune
params = {
    'max_depth': [3, 5, 7, 9, 11],
    'min_samples_leaf': [1, 3, 5, 7]
}

# Use grid search to tune the hyperparameters
grid_search = GridSearchCV(model, params, cv=3)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(grid_search, X_train, y_train, cv=3)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {np.mean(cv_scores)}")

# Use the best hyperparameters to make predictions on the test set
y_pred = grid_search.predict(X_test)

# Evaluate the model on the test set
test_score = grid_search.score(X_test, y_test)
print(f"Test set score: {test_score}")
