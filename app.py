from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
from typing import List
import resend
import os
import torch 
import torch.nn as nn
import re
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
        params = {
            "from": "onboarding@resend.dev",
            "to": email_to,
            "subject": f"New Contact Submission From {data[0]}",
            "text": coninfo
        }
        response = resend.Emails.send(params)
        return {"email":int(0)}
    #except Exception as e:
     #   return {"email":int(1)}
    except Exception as e:
        str(e)
        return {"email":str(e)}
        raise HTTPException(status_code=500,detail=str(e))
@app.post("/spam")
def spam(data: List[str] = Body(...)):
    try:
        class SpamLSTM(nn.Module): 
            def __init__( 
                self, 
                vocab_size, 
                embedding_dim=128, 
                hidden_dim=128 
            ):
                super().__init__() 
                self.embedding = nn.Embedding( 
                    vocab_size, 
                    embedding_dim, 
                    padding_idx=0 
                ) 
                self.lstm = nn.LSTM( 
                    input_size=embedding_dim, 
                    hidden_size=hidden_dim, 
                    batch_first=True 
                ) 
                self.fc = nn.Linear( 
                    hidden_dim, 
                    1 
                ) 
            def forward(self, x):
                x = self.embedding(x) 
                output, (hidden, cell) = self.lstm(x) 
                x = hidden[-1] 
                x = self.fc(x) 
                return x.squeeze(1)   
        device = torch.device( 
            "cuda" if torch.cuda.is_available() else "cpu" 
        )  
        checkpoint=torch.load("spam_model.pth",map_location=device)
        vocab= checkpoint["vocab"]
        vocab_size=checkpoint["vocab_size"]
        embedding_dim=checkpoint["embedding_dim"]
        hidden_dim=checkpoint["hidden_dim"]
        MAX_LENGTH=checkpoint["max_length"] 
        model = SpamLSTM( 
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, 
        ).to(device) 
        model.load_state_dict(checkpoint["model_state_dist"])
        def clean_text(text): 
            text = text.lower() 
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text) 
            return text.split() 
        def encode_text(tokens):
            return [ 
                vocab.get(word, vocab["<UNK>"]) 
                for word in tokens 
            ] 
        def pad_sequence(sequence): 
            # Truncate 
            sequence = sequence[:MAX_LENGTH] 
            # Padding 
            if len(sequence) < MAX_LENGTH: 
                sequence += [vocab["<PAD>"]] * ( 
                    MAX_LENGTH - len(sequence) 
                ) 
            return sequence
        def predict_email(email):
            model.eval()
            # Clean 
            tokens = clean_text(email) 
            # Encode 
            encoded = encode_text(tokens)
            # Pad 
            encoded = pad_sequence(encoded)
            # Tensor 
            tensor = torch.tensor( 
                [encoded], 
                dtype=torch.long 
            ).to(device)
            with torch.no_grad():
                output = model(tensor)
                probability = torch.sigmoid(output).item() 
                return (round(probability,4))
        a=predict_email(data)  
        return {"prediction": float(a)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))