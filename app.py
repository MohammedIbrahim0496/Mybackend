from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
from typing import List
from email.message import EmailMessage
import smtplib
import os
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
try:
    diabmodel = joblib.load("diabetes_model.pkl")
    diabscaler = joblib.load("diabetes_scaler.pkl")
    carmodel = joblib.load("Carmodel.pkl")
except Exception as e:
    diabmodel = diabscaler = carmodel = None

@app.get("/")
def home():
    return "Backend Running Successfully"
@app.post("/predict")
def predict(data: List[float] = Body(...)):
    try:
        model = diabmodel
        scaler = diabscaler
        x = np.array(data, dtype=float).reshape(1, -1)
        x = scaler.transform(x)
        prediction = model.predict(x)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/carpredict")
def carpredict(data: List[float] = Body(...)):
    try:
        model = carmodel
        x = np.array(data, dtype=float).reshape(1, -1)
        prediction = model.predict(x)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/email")
def email(data: List[str] = Body(...)):
    try:
        msg=EmailMessage()
        coninfo=f"Name :\n{data[0]} \nEmail :\n{data[1]} \nMessage :\n{data[2]}"
        msg["Subject"]=f"New Contact Submission From {data[0]}"
        msg["From"]=os.getenv("EMAIL_USER")
        msg["To"]=os.getenv("EMAIL_TO")
        msg.set_content(coninfo)
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com",465) as server:
                server.login(os.getenv("EMAIL_USER"),os.getenv("EMAIL_PASSWORD"))
                re=server.send_message(msg)
                print(re,flush=True)
                return {"email":int(0)}
        except Exception as e:
            return {"email":int(1)}
    except Exception as e:
        return {"email":int(1)}
        raise HTTPException(status_code=500,detail=str(e))