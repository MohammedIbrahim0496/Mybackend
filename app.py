
import os
import re
from pathlib import Path
from typing import List

import joblib
import numpy as np
import resend
import torch
import torch.nn as nn

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI(
    title="Prediction API",
    description="Diabetes, car price, email, and spam prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use CPU inference; no CUDA GPU is required.
DEVICE = torch.device("cpu")


# --------------------------------------------------
# Load scikit-learn models
# --------------------------------------------------

def load_joblib_model(filename: str):
    model_path = BASE_DIR / filename

    if not model_path.is_file():
        print(f"Model file missing: {model_path}")
        return None

    try:
        return joblib.load(model_path)
    except Exception as exc:
        print(f"Failed to load {filename}: {exc}")
        return None


diabmodel = load_joblib_model("diabetes_model.pkl")
diabscaler = load_joblib_model("diabetes_scaler.pkl")
carmodel = load_joblib_model("Carmodel.pkl")


# --------------------------------------------------
# PyTorch LSTM spam model
# --------------------------------------------------

class SpamLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]
        x = self.fc(x)
        return x.squeeze(1)


spam_model = None
vocab = None
MAX_LENGTH = None
spam_model_error = None


def load_spam_model():
    global spam_model, vocab, MAX_LENGTH, spam_model_error

    model_path = BASE_DIR / "spam_model.pth"

    if not model_path.is_file():
        spam_model_error = "spam_model.pth is missing"
        print(spam_model_error)
        return

    try:
        # Use weights_only=False only for a trusted checkpoint
        # created by you. The checkpoint contains a vocabulary
        # dictionary as well as the model weights.
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=False,
        )

        loaded_vocab = checkpoint["vocab"]
        embedding_dim = int(checkpoint["embedding_dim"])
        hidden_dim = int(checkpoint["hidden_dim"])
        max_length = int(checkpoint["max_length"])

        model = SpamLSTM(
            vocab_size=len(loaded_vocab),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )

        # Retain the checkpoint key from your original code.
        state_dict = checkpoint.get("model_state_dist")

        if state_dict is None:
            raise KeyError(
                "Missing 'model_state_dist' in spam_model.pth. "
                "Verify the key used when saving the checkpoint."
            )

        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        spam_model = model
        vocab = loaded_vocab
        MAX_LENGTH = max_length
        spam_model_error = None

        print("PyTorch spam model loaded successfully.")

    except Exception as exc:
        spam_model = None
        vocab = None
        MAX_LENGTH = None
        spam_model_error = str(exc)
        print(f"Failed to load spam model: {exc}")


load_spam_model()


# --------------------------------------------------
# Home and health
# --------------------------------------------------

@app.get("/")
def home():
    return {"message": "Backend Running Successfully"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "diabetes_model_loaded": diabmodel is not None,
        "diabetes_scaler_loaded": diabscaler is not None,
        "car_model_loaded": carmodel is not None,
        "spam_model_loaded": spam_model is not None,
    }


# --------------------------------------------------
# Diabetes prediction
# --------------------------------------------------

@app.post("/predict")
def predict(data: List[float] = Body(...)):
    if diabmodel is None or diabscaler is None:
        raise HTTPException(
            status_code=503,
            detail="Diabetes model or scaler is unavailable.",
        )

    try:
        x = np.asarray(data, dtype=float).reshape(1, -1)
        x = diabscaler.transform(x)
        result = diabmodel.predict(x)

        return {"prediction": int(result[0])}

    except Exception as exc:
        print(f"Diabetes prediction error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Diabetes prediction failed. Check the input features.",
        ) from exc


# --------------------------------------------------
# Car price prediction
# --------------------------------------------------

@app.post("/carpredict")
def carpredict(data: List[float] = Body(...)):
    if carmodel is None:
        raise HTTPException(
            status_code=503,
            detail="Car model is unavailable.",
        )

    try:
        x = np.asarray(data, dtype=float).reshape(1, -1)
        result = carmodel.predict(x)

        return {"prediction": float(result[0])}

    except Exception as exc:
        print(f"Car prediction error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Car prediction failed. Check the input features.",
        ) from exc


# --------------------------------------------------
# Contact email
# --------------------------------------------------

@app.post("/email")
def email(data: List[str] = Body(...)):
    if len(data) < 3:
        raise HTTPException(
            status_code=422,
            detail="Expected [name, email, message].",
        )

    api_key = os.getenv("RESEND_API_KEY")
    email_to = os.getenv("EMAIL_TO")

    if not api_key or not email_to:
        raise HTTPException(
            status_code=503,
            detail="Email service is not configured.",
        )

    try:
        name, sender_email, message = data[:3]

        resend.api_key = api_key

        params = {
            "from": "onboarding@resend.dev",
            "to": email_to,
            "subject": f"New Contact Submission From {name}",
            "text": (
                f"Name:\n{name}\n"
                f"Email:\n{sender_email}\n"
                f"Message:\n{message}"
            ),
        }

        resend.Emails.send(params)

        return {"email": 0}

    except Exception as exc:
        print(f"Email sending error: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Unable to send the email.",
        ) from exc


# --------------------------------------------------
# Spam detection helpers
# --------------------------------------------------

def clean_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.split()


def predict_spam_probability(email_text: str) -> float:
    if spam_model is None or vocab is None or MAX_LENGTH is None:
        raise HTTPException(
            status_code=503,
            detail="PyTorch spam model is unavailable.",
        )

    if "<UNK>" not in vocab or "<PAD>" not in vocab:
        raise HTTPException(
            status_code=500,
            detail="Spam vocabulary is missing <UNK> or <PAD>.",
        )

    tokens = clean_text(email_text)

    encoded = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    encoded = encoded[:MAX_LENGTH]

    if len(encoded) < MAX_LENGTH:
        encoded.extend(
            [vocab["<PAD>"]] * (MAX_LENGTH - len(encoded))
        )

    tensor = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.inference_mode():
        output = spam_model(tensor)
        probability = torch.sigmoid(output).item()

    return round(float(probability), 4)


# --------------------------------------------------
# Spam prediction
# --------------------------------------------------

@app.post("/spam")
def spam(data: str = Body(...)):
    try:
        probability = predict_spam_probability(data)
        return {"prediction": probability}

    except HTTPException:
        raise

    except Exception as exc:
        print(f"Spam prediction error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Spam prediction failed.",
        ) from exc
