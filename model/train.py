import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load your dataset (adjust the path as needed)
df = pd.read_csv(r'\data\diabetes.csv')

# Separate features and target variable
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']               # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the scaler and fit it on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define individual models with best parameters from previous tuning (or default if necessary)
svm_model = SVC(kernel='linear', C=1, max_iter=3000, probability=True, random_state=42)
nb_model = GaussianNB(var_smoothing=1e-09)
rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, random_state=42)
xgb_model = XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=50, subsample=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss')

# Create an ensemble model using VotingClassifier (with soft voting)
ensemble_model = VotingClassifier(
    estimators=[
        ('svm', svm_model),
        ('nb', nb_model),
        ('rf', rf_model),
        ('xgb', xgb_model)
    ],
    voting='soft',  # Soft voting for probability-based voting
    weights=[0.63, 0.64, 0.65, 0.63]  # Adjust these weights based on prior model performance
)

# Train the ensemble model
ensemble_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = ensemble_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Ensemble Model Accuracy: {accuracy:.4f}")
print(f"Ensemble Model F1 Score: {f1:.4f}")

# Create the 'model' directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the scaler and ensemble model using joblib
scaler_path = os.path.join('model', 'scaler.pkl')
ensemble_model_path = os.path.join('model', 'ensemble_model.pkl')

joblib.dump(scaler, scaler_path)
joblib.dump(ensemble_model, ensemble_model_path)

print(f"Scaler saved to: {scaler_path}")
print(f"Ensemble model saved to: {ensemble_model_path}")
