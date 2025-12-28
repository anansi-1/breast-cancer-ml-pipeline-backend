from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np



app = FastAPI(title="Breast Cancer Prediction API")


try:
    logistic_model = joblib.load("models/logistic_model.joblib")
    decision_tree_model = joblib.load("models/decision_tree_model.joblib")
    knn_model = joblib.load("models/knn_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
except Exception as e:
    print("Error loading models:", e)
    raise RuntimeError("Models could not be loaded. Check paths and files.")


class Features(BaseModel):
    features: list[float]  

@app.get("/")
def home():
    return {"message": "Breast Cancer Prediction API is running"}



@app.post("/predict")
def predict(data: Features):
    # Check feature length
    if len(data.features) != 30:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid number of features: expected 30, got {len(data.features)}"
        )

    try:
        X = np.array(data.features, dtype=float).reshape(1, -1)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="All features must be numeric values."
        )

    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error during feature scaling: {str(e)}"
        )

    try:
        return {
            "logistic_regression": int(logistic_model.predict(X_scaled)[0]),
            "decision_tree": int(decision_tree_model.predict(X_scaled)[0]),
            "knn": int(knn_model.predict(X_scaled)[0])
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )


app = FastAPI(title="Breast Cancer Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

