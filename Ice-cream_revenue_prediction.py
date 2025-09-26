import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

try:
    icecream = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Ice%20Cream.csv')
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

y = icecream['Revenue']
X = icecream[['Temperature']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"Intercept (b₀): {model.intercept_}")
print(f"Coefficient (b₁): {model.coef_[0]}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
print(f"Mean Absolute Percentage Error: {mean_absolute_percentage_error(y_test, y_pred):.2%}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
