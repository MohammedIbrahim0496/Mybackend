from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
from typing import List
from email.message import EmailMessage
import smtplib
import os
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
        coninfo=f"Name :{data[0]} \nEmail :{data[1]} \nMessage :{data[2]}"
        msg["Subject"]=f"New Contact Submission From {data[0]}"
        msg["From"]="EMAIL_USER"
        msg["To"]="EMAIL_TO"
        msg.set_content(coninfo)
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com",465) as server:
                server.login(os.environ["EMAIL_USER"],os.environ["EMAIL_PASSWORD"])
                server.send_message(msg)
                return {"email":int(0)}
        except Exception as e:
            strr=f"1{e}"
            return {"email":str(strr)}     
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))          
    """try:
        msg=EmailMessage()
        coninfo=f"Name :{sname.value} \nEmail :{semail.value} \nMessage :{smass.value}"
        msg["Subject"]=f"New Contact Submission From {sname.value}"
        msg["From"]="n.ibrahim.04092006@gmail.com"
        msg["To"]="n.mohammedibrahim19472006@gmail.com"
        msg.set_content(coninfo)
        with smtplib.SMTP_SSL("smtp.gmail.com",465) as server:
            server.login("n.ibrahim.04092006@gmail.com","uxlpbejwlswofocw")
            server.send_message(msg)   
        #status_text.value ="Email Sent Succesfully!!"
        #status_text.color="green"
        #await asyncio.sleep(5)   
        #status_text.value =""
        #status_text.color=ft.Colors.TRANSPARENT
    except:
        #status_text.value ="Failed To Send Email !!"
        #status_text.color="red"
        #await asyncio.sleep(5)   
        #status_text.value =""
        #status_text.color=ft.Colors.TRANSPARENT """  