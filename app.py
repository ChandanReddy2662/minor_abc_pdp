from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Correct Google Drive file ID
file_id = "1tvZp8Ue4zMy_cmVEcwK8FOKs5ANFpDKX"
model_path = "plant_disease_model_inception.h5"

# ✅ Download model if it doesn't exist
if not os.path.exists(model_path):
    print("🔽 Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    print("✅ Model downloaded successfully!")

# ✅ Load the model
model = tf.keras.models.load_model(model_path)

# ✅ Correct class names (removed extra commas)
CLASS_NAMES = [
    'Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy', 'Potato___healthy',
    'Corn_(maize)___Northern_Leaf_Blight', 'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Strawberry___Leaf_scorch',
    'Peach___healthy', 'Coffee__Rust', 'Apple___Apple_scab', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Bacterial_spot', 'Coffee__red spider mite', 'Apple___Black_rot',
    'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot', 'Apple___Cedar_apple_rust',
    'Tomato___Target_Spot', 'Chili__whitefly', 'Pepper,_bell___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight', 'Chili__healthy',
    'Tomato___Tomato_mosaic_virus', 'Strawberry___healthy', 'Apple___healthy', 'Grape___Black_rot',
    'Chili__leaf spot', 'Potato___Early_blight', 'Cherry_(including_sour)___healthy',
    'Coffee__healthy', 'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)',
    'Tomato___Leaf_Mold', 'Chili__yellowish', 'Chili__leaf curl',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Pepper,_bell___Bacterial_spot',
    'Corn_(maize)___healthy'
]

# ✅ Function to preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).resize((139, 139))  # ✅ Ensure correct input shape
    image = np.array(image) / 255.0  # Normalize to [0,1]
    return np.expand_dims(image, axis=0)  # Add batch dimension

# ✅ Main route (upload & predict)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            image_array = preprocess_image(file_path)
            prediction = model.predict(image_array)
            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = round(100 * np.max(prediction), 2)

            return render_template("index.html", prediction=predicted_class, confidence=confidence, image=file_path)
    
    return render_template("index.html", prediction=None)

# ✅ Information page route
@app.route("/info")
def info():
    return render_template("info.html")

# ✅ Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
