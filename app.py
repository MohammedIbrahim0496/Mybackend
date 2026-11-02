
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
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env when running locally.
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Prediction API",
    description="Diabetes, car price, contact email, and spam prediction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Load scikit-learn models safely at startup/import.
# --------------------------------------------------

def load_joblib_model(filename: str):
    path = BASE_DIR / filename

    if not path.is_file():
        print(f"Model file not found: {path}")
        return None

    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"Could not load {filename}: {exc}")
        return None


diabmodel = load_joblib_model("diabetes_model.pkl")
diabscaler = load_joblib_model("diabetes_scaler.pkl")
carmodel = load_joblib_model("Carmodel.pkl")


# --------------------------------------------------
# Spam LSTM model definition.
# --------------------------------------------------

class SpamLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
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


# Use CPU on the hosted API; no CUDA GPU is required.
DEVICE = torch.device("cpu")

spam_model = None
spam_vocab = None
spam_max_length = None
spam_load_error = None


def load_spam_model():
    global spam_model, spam_vocab, spam_max_length, spam_load_error

    checkpoint_path = BASE_DIR / "spam_model.pth"

    if not checkpoint_path.is_file():
        spam_load_error = "spam_model.pth was not found."
        print(spam_load_error)
        return

    try:
        # Only load trusted model files that you control.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )

        vocab = checkpoint["vocab"]
        embedding_dim = int(checkpoint["embedding_dim"])
        hidden_dim = int(checkpoint["hidden_dim"])
        max_length = int(checkpoint["max_length"])

        model = SpamLSTM(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )

        # Preserve compatibility with the checkpoint key in your code.
        state_dict = checkpoint.get("model_state_dist")
        if state_dict is None:
            raise KeyError(
                "Checkpoint does not contain 'model_state_dist'. "
                "Check the key used when saving your trained model."
            )

        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        spam_model = model
        spam_vocab = vocab
        spam_max_length = max_length
        spam_load_error = None

        print("Spam model loaded successfully.")

    except Exception as exc:
        spam_model = None
        spam_vocab = None
        spam_max_length = None
        spam_load_error = str(exc)
        print(f"Could not load spam model: {exc}")


load_spam_model()


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def home():
    return {"message": "Backend Running Successfully"}


@app.get("/health")
def health():
    """Check application availability and model loading status."""
    return {
        "status": "ok",
        "diabetes_model_loaded": diabmodel is not None,
        "diabetes_scaler_loaded": diabscaler is not None,
        "car_model_loaded": carmodel is not None,
        "spam_model_loaded": spam_model is not None,
    }


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
        prediction = diabmodel.predict(x)

        return {"prediction": int(prediction[0])}

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Diabetes prediction failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Diabetes prediction failed. Check the input features.",
        ) from exc


@app.post("/carpredict")
def carpredict(data: List[float] = Body(...)):
    if carmodel is None:
        raise HTTPException(
            status_code=503,
            detail="Car prediction model is unavailable.",
        )

    try:
        x = np.asarray(data, dtype=float).reshape(1, -1)
        prediction = carmodel.predict(x)

        return {"prediction": float(prediction[0])}

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Car prediction failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Car prediction failed. Check the input features.",
        ) from exc


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

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Email sending failed: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Unable to send the email.",
        ) from exc


# --------------------------------------------------
# Spam text preprocessing and prediction
# --------------------------------------------------

def clean_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.split()


def predict_spam_probability(email_text: str) -> float:
    if spam_model is None or spam_vocab is None:
        raise HTTPException(
            status_code=503,
            detail="Spam model is unavailable.",
        )

    tokens = clean_text(email_text)

    unknown_token = spam_vocab.get("<UNK>")
    padding_token = spam_vocab.get("<PAD>")

    if unknown_token is None or padding_token is None:
        raise HTTPException(
            status_code=500,
            detail="Spam vocabulary is missing required special tokens.",
        )

    encoded = [
        spam_vocab.get(word, unknown_token)
        for word in tokens
    ]

    encoded = encoded[:spam_max_length]

    if len(encoded) < spam_max_length:
        encoded += [padding_token] * (
            spam_max_length - len(encoded)
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


@app.post("/spam")
def spam(data: str = Body(...)):
    try:
        probability = predict_spam_probability(data)
        return {"prediction": probability}

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Spam prediction failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Spam prediction failed.",
        ) from exc
