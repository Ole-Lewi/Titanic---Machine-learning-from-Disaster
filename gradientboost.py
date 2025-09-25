import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Save PassengerId from test set (needed for submission)
test_ids = test['PassengerId']

# Preprocess categorical & numerical features
train["Sex"] = train["Sex"].map({"male":0, "female":1})
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

train["Age"].fillna(train["Age"].median(), inplace=True)
test["Age"].fillna(test["Age"].median(), inplace=True)

test["Fare"].fillna(test["Fare"].median(), inplace=True)

# Define features and target
X = train[["Pclass", "Sex", "Age", "Fare"]]
y = train["Survived"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Validation Accuracy: {accuracy:.2f}')

# Make predictions for submission
test_features = test[["Pclass", "Sex", "Age", "Fare"]]
predictions = model.predict(test_features)

# Create submission file
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Survived": predictions
})
submission.to_csv("submission 02.csv", index=False)
print("Submission file saved as submission.csv")