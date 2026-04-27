from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io
import os

IMG_SIZE = 224
model = None

def load_model():
    global model
    if model is None:
        try:
            import tensorflow as tf
            model_path = os.path.join(os.getcwd(), "RURALOPS.keras")
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(lifespan=lifespan)

def preprocess(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
def home():
    return {"message": "RURALOPS API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if model is None:
            return {"error": "Model not loaded"}

        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img = preprocess(image)

        prediction = model.predict(img)[0][0]

        if prediction >= 0.5:
            label = "dirty"
            confidence = float(prediction)
        else:
            label = "clean"
            confidence = float(1 - prediction)

        return {"class": label, "confidence": round(confidence, 4)}

    except Exception as e:
        return {"error": str(e)}