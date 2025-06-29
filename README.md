# Car Price Prediction API

This project predicts the **selling price of used cars** using a machine learning model and serves predictions via a **FastAPI**-powered API. The model is trained on a real-world dataset containing details like mileage, engine capacity, fuel type, and more.

---

## Project Workflow

1. **Data Preprocessing**
   - Handled missing values, converted text to numeric, and encoded categorical features with `LabelEncoder`

2. **Model Training**
   - Used `RandomForestRegressor` for prediction
   - Applied `RandomizedSearchCV` to tune key hyperparameters
   - Selected the best performing model based on RMSE and R² score

3. **Model Evaluation**
   - RMSE: ~107,000  
   - R² Score: ~0.98

4. **API Deployment**
   - Trained model, encoders, and feature schema saved using `joblib`
   - Built an API with FastAPI to serve predictions via a `/predict` endpoint

---

## API Prediction Endpoint

**POST** `/predict`  
Takes car features as input and returns the predicted price.

---

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```
### Start API Server

```bash
cd car_price_api
uvicorn main:app --reload
```
- Visit: http://127.0.0.1:8000/docs to access the Swagger UI

---

## Project Structure

```bash
car-price-predictor/
├── Car_Price_Prediction.ipynb        # Training + tuning notebook
├── requirements.txt                  # Dependencies
└── car_price_api/
    ├── main.py                       # FastAPI app
    ├── car_price_model_tuned.pkl     # Trained RF model
    ├── label_encoders.pkl            # Encoders for categorical variables
    └── model_features.pkl            # Feature column order
```

---
