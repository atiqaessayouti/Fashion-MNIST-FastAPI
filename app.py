from pathlib import Path
import time

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "fashion_mnist.keras"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

app = FastAPI(
    title="Fashion-MNIST API",
    description="Classification d'une image Fashion-MNIST de 28x28 pixels.",
    version="1.0.0",
)

# Le modèle est chargé une seule fois au démarrage du module/serveur.
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Modèle introuvable : {MODEL_PATH}. Placez fashion_mnist.keras à côté de app.py."
    )

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Modèle chargé : {MODEL_PATH}")
print(f"Entrée : {model.input_shape} | Sortie : {model.output_shape}")


class ImageInput(BaseModel):
    pixels: list[float] = Field(
        ...,
        description="Liste de 784 pixels, dans l'ordre ligne par ligne.",
        min_length=1,
    )

    @field_validator("pixels")
    @classmethod
    def validate_finite_values(cls, values):
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Les pixels doivent être des nombres finis.")
        return values


@app.get("/")
def home():
    return {
        "message": "Bienvenue dans Fashion-MNIST API",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_PATH.name,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
    }


@app.post("/predict")
def predict(data: ImageInput):
    if len(data.pixels) != 784:
        raise HTTPException(
            status_code=400,
            detail="L'entrée doit contenir exactement 784 pixels.",
        )

    image = np.asarray(data.pixels, dtype=np.float32)

    if np.any(image < 0) or np.any(image > 255):
        raise HTTPException(
            status_code=400,
            detail="Les pixels doivent être compris entre 0 et 255.",
        )

    image = image.reshape(1, 28, 28)

    # Même prétraitement que le notebook : pixels dans [0, 1].
    if float(np.max(image)) > 1.0:
        image = image / 255.0

    # Le chronomètre mesure l'appel d'inférence du modèle.
    start = time.perf_counter()
    outputs = model(image, training=False)
    # Synchroniser le résultat TensorFlow avant d'arrêter le chronomètre.
    logits = outputs.numpy()
    latency_ms = (time.perf_counter() - start) * 1000.0

    # Le modèle du notebook produit des logits (Dense(10), sans softmax).
    probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
    class_id = int(np.argmax(probabilities))

    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": float(probabilities[class_id]),
        "latency_ms": round(float(latency_ms), 3),
    }
