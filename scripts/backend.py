import logging
from transformers import pipeline
import torch
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from database import Prediction, SessionLocal, init_db
from sqlalchemy.exc import SQLAlchemyError

model_dir = Path(__file__).resolve().parent.parent / "models" / "bert_best_2"

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Инициализация базы данных при старте


@app.on_event("startup")
def on_startup():
    init_db()


class TextRequest(BaseModel):
    text: str


id2label = {0: "negative", 1: "positive", 2: "trash", 3: "neutral"}


def load_models():
    pipe = pipeline("text-classification", model=str(model_dir),
                    device=0 if torch.cuda.is_available() else -1)
    return pipe


pipe = load_models()


@app.get("/")
async def root():
    return {
        "message": "Hello! Welcome to the Emotional Classification API. Use /predict/ to classify text"
    }


@app.post("/predict/")
def predict(request: TextRequest):
    text = request.text
    logging.info(f"Received text for prediction: {text}")

    logging.info("Making prediction...")
    result = pipe(text, truncation=True, max_length=512)[0]
    predicted_class = result["label"]
    score = result["score"]
    logging.info(f"Predicted class: {predicted_class} (score={score:.3f})")

    # Сохраняем в базу данных
    db = SessionLocal()
    try:
        db_obj = Prediction(comment=text, predicted_class=predicted_class)
        db.add(db_obj)
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        logging.error(f"Ошибка при сохранении в БД: {e}")
    finally:
        db.close()

    return {"predicted_class": predicted_class, "score": score}
