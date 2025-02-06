from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io

# Load the trained model
try:
    model = keras.models.load_model("digit_recognition_cnn.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")

# Initialize FastAPI
app = FastAPI()

@app.get("/")
async def serve_frontend():
    return FileResponse(r"C:\Users\Pravin\Desktop\cnn_image\templates\index.html")

@app.post("/predict/")
async def predict_digit(file: UploadFile = File(...)):
    try:
        # Read image file from request
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28

        # Convert image to numpy array and preprocess
        image_array = np.array(image)
        image_array = 255 - image_array  # Invert colors (MNIST expects white on black)
        image_array = image_array / 255.0  # Normalize to 0-1
        image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for CNN

        # Predict
        prediction = model.predict(image_array)
        predicted_digit = int(np.argmax(prediction))  # Ensure int return type

        return {"predicted_digit": predicted_digit}
    
    except Exception as e:
        return {"error": str(e)}
