
import os
import joblib

from sklearn.linear_model import LinearRegression


# Training data: house area (square feet)
# Demo prices are in lakh rupees.
X = [
    [500],
    [700],
    [900],
    [1100],
    [1300],
    [1500],
    [1700],
    [1900],
    [2100],
    [2300],
]

y = [
    20,
    28,
    36,
    44,
    52,
    60,
    68,
    76,
    84,
    92,
]


# Train the regression model
model = LinearRegression()
model.fit(X, y)


# Save the trained model
output_path = os.path.join(
    os.path.dirname(__file__),
    "house_price_regression.joblib",
)

joblib.dump(model, output_path)

print("Regression model trained successfully!")
print(f"Saved model at: {output_path}")

# Test prediction for a 1200 sq ft house
prediction = model.predict([[1200]])[0]
print(f"Demo prediction for 1200 sq ft: {prediction:.2f} lakh")
