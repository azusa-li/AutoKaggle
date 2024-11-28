import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
submission = pd.read_csv("./input/sample_submission.csv")

# One-hot encode categorical variables
train = pd.get_dummies(
    train, columns=train.select_dtypes(include=["object"]).columns.drop("NObeyesdad")
)
test = pd.get_dummies(test, columns=test.select_dtypes(include=["object"]).columns)

# Align test set with train set
test = test.reindex(columns=train.columns.drop("NObeyesdad"), fill_value=0)

# Encode target variable
target_encoder = LabelEncoder()
train["NObeyesdad"] = target_encoder.fit_transform(train["NObeyesdad"])

# Separate features and target
X = train.drop(["NObeyesdad", "id"], axis=1)
y = train["NObeyesdad"]
X_test = test.drop(["id"], axis=1)

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LightGBM model
model = LGBMClassifier(random_state=42)

# Evaluate with cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"Cross-validated Accuracy: {np.mean(cv_scores):.4f}")

# Train on training set and evaluate on validation set
model.fit(X_train, y_train)
val_preds = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Train on full training set and predict on the test set
model.fit(X, y)
test_preds = model.predict(X_test)

# Prepare submission file
submission["NObeyesdad"] = target_encoder.inverse_transform(test_preds)
submission.to_csv("./working/submission.csv", index=False)
