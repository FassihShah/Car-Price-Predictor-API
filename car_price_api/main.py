from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model, encoders and feature list
model = joblib.load("car_price_model_tuned.pkl")
encoders = joblib.load("label_encoders.pkl")
features = joblib.load("model_features.pkl")

app = FastAPI()

# Input schema
class CarData(BaseModel):
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    seats: int
    car_age: int

@app.post("/predict")
def predict_price(data: CarData):
    # Convert input to dict
    input_dict = data.dict()

    # Apply label encoders to categorical features
    for col in ['fuel', 'seller_type', 'transmission', 'owner']:
        encoder = encoders[col]
        input_dict[col] = encoder.transform([input_dict[col]])[0]

    # Convert to DataFrame with same column order as training
    input_df = pd.DataFrame([input_dict])[features]

    # Predict
    pred = model.predict(input_df)[0]
    return {"predicted_price": round(pred, 2)}
