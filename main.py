from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import io

app = FastAPI(title="Smart Agriculture Multi-Model API")

# =========================================
# GLOBAL VARIABLES (Empty at start)
# =========================================

disease_model = None
disease_classes = None
disease_info = None
recommendation_model = None
label_encoder = None

# =========================================
# LOAD MODELS ON STARTUP (IMPORTANT FIX)
# =========================================

@app.on_event("startup")
def load_models():
    global disease_model, disease_classes, disease_info
    global recommendation_model, label_encoder

    print("Loading models...")

    # ---- Load Disease Model (.keras) ----
    disease_model = tf.keras.models.load_model("models/crop_disease_model.keras")

    with open("data/class_names.pkl", "rb") as f:
        disease_classes = pickle.load(f)

    with open("data/disease_info.pkl", "rb") as f:
        disease_info = pickle.load(f)

    # ---- Load Recommendation Model (.pkl) ----
    with open("models/crop_recommendation_model.pkl", "rb") as f:
        recommendation_model = pickle.load(f)

    with open("encoders/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    print("Models loaded successfully!")

# =========================================
# HOME ROUTE
# =========================================

@app.get("/")
def home():
    return {"status": "✅ Smart Agriculture API Running Successfully"}

# =========================================
# 1️⃣ CROP DISEASE PREDICTION (IMAGE)
# =========================================

@app.post("/predict-disease")
async def predict_disease(file: UploadFile = File(...)):

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = disease_model.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    disease_name = disease_classes[predicted_index]

    return {
        "model": "Crop Disease Detection",
        "disease": disease_name,
        "confidence": round(confidence, 2),
        "precautions": disease_info[disease_name]["precautions"],
        "medicine": disease_info[disease_name]["medicine"]
    }

# =========================================
# 2️⃣ CROP RECOMMENDATION (NUMERICAL)
# =========================================

class CropInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@app.post("/recommend-crop")
def recommend_crop(data: CropInput):

    input_data = np.array([[
        data.nitrogen,
        data.phosphorus,
        data.potassium,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]])

    prediction = recommendation_model.predict(input_data)
    crop_name = label_encoder.inverse_transform(prediction)[0]

    return {
        "model": "Crop Recommendation",
        "recommended_crop": crop_name
    }
