from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
from typing import List
import resend
import os
from pydantic import BaseModel,EmailStr
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
        resend.api_key = os.getenv("RESEND_API_KEY")
        email_to = os.getenv("EMAIL_TO")
        coninfo=f"Name :\n{data[0]} \nEmail :\n{data[1]} \nMessage :\n{data[2]}"
        print("SCreating Message",flush=True)
        params = {
            "from": "onboarding@resend.dev",
            "to": email_to,
            "subject": f"New Contact Submission From {data[0]}",
            "text": coninfo
        }
        response = resend.Emails.send(params)
        print("sent successfully",response,flush=True)
        return {"email":int(0)}
    #except Exception as e:
     #   return {"email":int(1)}
    except Exception as e:
        str(e)
        return {"email":str(e)}
        raise HTTPException(status_code=500,detail=str(e))