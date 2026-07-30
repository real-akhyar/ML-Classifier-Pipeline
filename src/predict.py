import os
import joblib
import numpy as np

def predict(input_features=None):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, "models", "model.joblib")
    scaler_path = os.path.join(base_dir, "models", "scaler.joblib")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            "Model artifacts not found! Please run `python src/train.py` first."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    class_names = ['class_0', 'class_1', 'class_2']

    # Sample data point if none provided (Wine dataset format: 13 features)
    if input_features is None:
        input_features = [
            13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050.0
        ]

    features_array = np.array(input_features).reshape(1, -1)
    scaled_features = scaler.transform(features_array)

    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]

    print("=" * 50)
    print("        ML Model Inference Execution")
    print("=" * 50)
    print(f"Input Features: {input_features}")
    print(f"Predicted Class Index : {prediction}")
    print(f"Predicted Class Name  : {class_names[prediction]}")
    print(f"Class Probabilities   : {dict(zip(class_names, [round(p, 4) for p in probabilities]))}")
    print("=" * 50)

    return {
        "class_index": int(prediction),
        "class_name": class_names[prediction],
        "probabilities": {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    }

if __name__ == "__main__":
    predict()
