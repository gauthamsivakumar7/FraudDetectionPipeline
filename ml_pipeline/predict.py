# ml_pipeline/predict.py
import joblib
import pandas as pd

# A real model is not done until I can load it and run inference on new data

def predict(model_path, input_dict, threshold=0.5):
    model = joblib.load(model_path)
    X = pd.DataFrame([input_dict])

    prob = model.predict_proba(X)[:, 1][0]
    pred = int(prob >= threshold)

    return {
        "probability": float(prob),
        "prediction": pred,
        "threshold": threshold
    }


if __name__ == "__main__":
    sample = {
        "feature_1": 0.25,
        "feature_2": 3.7
    }
    print(predict("artifacts/logreg_pipeline.joblib", sample, threshold=0.5))