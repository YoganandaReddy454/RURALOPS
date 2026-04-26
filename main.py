from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = FastAPI()

model = None  # load later

IMG_SIZE = 224

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("RURALOPS.keras")

def preprocess(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        load_model()  # load here

        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img = preprocess(image)

        prediction = model.predict(img, verbose=0)[0][0]

        if prediction >= 0.5:
            label = "dirty"
            confidence = float(prediction)
        else:
            label = "clean"
            confidence = float(1 - prediction)

        return {
            "class": label,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return {"error": str(e)}