from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import predict

app = FastAPI()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def get_prediction(input_data: IrisInput):
    data = input_data.dict()
    prediction = predict(data)
    return {"prediction": int(prediction)}