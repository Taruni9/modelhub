from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load Iris dataset
iris = load_iris()

# Create model
model = RandomForestClassifier(
    n_estimators=10,
    random_state=42
)

# Train model
model.fit(iris.data, iris.target)

# Save the trained model
joblib.dump(model, "iris_classifier.joblib")

print("Model saved successfully!")