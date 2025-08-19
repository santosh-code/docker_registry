import joblib
import pandas as pd

model = joblib.load("app/model.pkl")

def predict(data):
    df = pd.DataFrame([data])
    return model.predict(df)[0]
