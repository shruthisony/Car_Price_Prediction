from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Load models
with open("CarPricePred.pkl", "rb") as file:
    model = pickle.load(file)

with open("scalar.pkl","rb") as file:
    scalar = pickle.load(file)

# FastAPI instance
app = FastAPI()

# Request model
class CarInput(BaseModel):
    year: int
    present_price: float
    kms_driven: int
    owner: int
    fuel_type: str  # "Petrol" or "Diesel"
    seller_type: str  # "Dealer" or "Individual"
    transmission: str  # "Manual" or "Automatic"

@app.post("/predict")
def predict_price(car: CarInput):
    try:
        # one-hot encoding
        fuel_type_petrol = 1 if car.fuel_type == "Petrol" else 0
        fuel_type_diesel = 1 if car.fuel_type == "Diesel" else 0
        seller_type_individual = 1 if car.seller_type == "Individual" else 0
        transmission_manual = 1 if car.transmission == "Manual" else 0

        input_array = np.array([[car.year, car.present_price, car.kms_driven, car.owner,
                                 fuel_type_petrol, fuel_type_diesel,
                                 seller_type_individual, transmission_manual]])
        
        # Apply the same scaler used during training
        input_scaled = scalar.transform(input_array)

        prediction = model.predict(input_scaled)

        return {"predicted_price": float(prediction[0][0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

