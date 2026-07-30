import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_and_export():
    print("=" * 60)
    print("      Machine Learning Model Training Pipeline (Wine Dataset)")
    print("=" * 60)

    # 1. Load Dataset
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target

    print(f"\n[+] Dataset Loaded successfully: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[+] Target Classes: {list(wine.target_names)}")

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Preprocessing (Feature Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Model Training
    print("\n[+] Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # 5. Model Evaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[OK] Model Training Complete!")
    print(f"[OK] Test Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))

    # 6. Export Model and Scaler Artifacts
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "model.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    features_path = os.path.join(models_dir, "feature_names.joblib")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(list(wine.feature_names), features_path)

    print(f"[OK] Saved model artifact to: {model_path}")
    print(f"[OK] Saved scaler artifact to: {scaler_path}")
    print(f"[OK] Saved feature names to:  {features_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_and_export()
