import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset from CSV file
df = pd.read_csv("new.csv")  # Ensure the file exists and has 'Area' & 'Price' columns

# Display the first few rows of the dataset
print(df.head())
a
# Handle missing values (if any)
df = df.dropna()

# Define features and target variable
X = df[["Area"]]  # Features (Independent Variable)
y = df["Price"]   


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")


joblib.dump(model, "house_price_model.pkl")


loaded_model = joblib.load("house_price_model.pkl")


new_house = pd.DataFrame([[3500]], columns=["Area"]) 
predicted_price = loaded_model.predict(new_house)
print(f"Predicted Price for 3500 sq ft: {predicted_price[0]}")


plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')  # Perfect fit line
plt.show()
