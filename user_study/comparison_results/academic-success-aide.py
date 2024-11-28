import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score

# Load the data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")

# Prepare the data
X = train.drop(["Target", "id"], axis=1)
y = train["Target"].map({"Dropout": 0, "Enrolled": 1, "Graduate": 2})
X_test = test.drop(["id"], axis=1)

# Initialize LightGBM model
model = lgb.LGBMClassifier()

# Define parameter grid for RandomizedSearchCV
param_grid = {
    "num_leaves": [31, 50, 70],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200, 300],
    "boosting_type": ["gbdt", "dart"],
}

# Randomized search
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42,
)

# Fit the random search model
random_search.fit(X, y)

# Best model
best_model = random_search.best_estimator_

# 5-fold cross-validation with the best model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Train the best model
    best_model.fit(X_train, y_train)

    # Validate the model
    y_val_pred = best_model.predict(X_val)
    score = accuracy_score(y_val, y_val_pred)
    cv_scores.append(score)

# Print cross-validation accuracy
print(
    "5-fold cross-validation accuracy with hyperparameter tuning: {:.4f}".format(
        sum(cv_scores) / len(cv_scores)
    )
)

# Train on the entire training data
best_model.fit(X, y)

# Predict on test set
y_pred = best_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test["id"], "Target": y_pred})
submission["Target"] = submission["Target"].map(
    {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
)
submission.to_csv("./working/submission.csv", index=False)
