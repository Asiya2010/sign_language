import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# ---- Load data ----
df = pd.read_csv('dataset/landmarks.csv')
print(f"Total samples: {len(df)}")
print(f"Samples per sign:\n{df['label'].value_counts()}")

X = df.drop('label', axis=1)
y = df['label']

# ---- Split into train/test ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Train ----
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---- Evaluate ----
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---- Save the model ----
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/sign_classifier.pkl')
print("\nModel saved to models/sign_classifier.pkl")