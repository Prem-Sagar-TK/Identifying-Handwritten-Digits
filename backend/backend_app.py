import os
import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask App
app = Flask(__name__)
# Enable CORS for all routes to allow React frontend to connect
CORS(app)

# Device configuration (CUDA if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the CNN architecture (must match the trained model in Phase2.py)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load the trained CNN model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_cnn.pth")
model = CNN().to(device)

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        # Initialize model with random weights if loading fails (fallback)
        model.eval()
else:
    print(f"Warning: {MODEL_PATH} not found. Running with uninitialized weights.")
    model.eval()

def preprocess_image(image_bytes):
    """
    Preprocesses raw image bytes for MNIST CNN prediction:
    1. Converts image to grayscale ('L')
    2. Resizes to 28x28 pixels
    3. Automatically inverts colors if the background is light (black-on-white drawing)
    4. Normalizes pixel values to [0, 1]
    5. Converts to PyTorch Tensor shape (1, 1, 28, 28)
    """
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to grayscale
    image = image.convert("L")
    
    # Resize to 28x28 (using modern Resampling if available, else ANTIALIAS fallback)
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.ANTIALIAS
    image = image.resize((28, 28), resample_filter)
    
    # Convert to numpy array
    img_np = np.array(image, dtype=np.float32)
    
    # Heuristic for color inversion:
    # MNIST dataset expects white digits (255) on a black background (0).
    # If the average pixel value of the image is > 127.5, the background is light.
    # We must invert the image so the digit becomes white and background becomes black.
    if np.mean(img_np) > 127.5:
        img_np = 255.0 - img_np
        
    # Scale pixel values to [0, 1]
    img_np = img_np / 255.0
    
    # Convert to PyTorch tensor and add batch & channel dimensions: [1, 1, 28, 28]
    tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).to(device)
    return tensor, img_np

@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint."""
    return jsonify({
        "status": "healthy",
        "device": str(device),
        "model_loaded": os.path.exists(MODEL_PATH)
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Predict endpoint to handle image uploads and return prediction results."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided in request.files['image']"}), 400
        
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
        
    try:
        image_bytes = file.read()
        tensor, img_np = preprocess_image(image_bytes)
        
        # Run inference
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1).squeeze(0)
            
            # Extract prediction, confidence, and probabilities list
            prediction = torch.argmax(probabilities).item()
            confidence = round(probabilities[prediction].item() * 100, 2)
            probabilities_list = [round(p.item(), 5) for p in probabilities]
            
        # Convert preprocessed numpy array back to base64 for visualization
        # img_np has values in range [0, 1]
        prep_img = Image.fromarray((img_np * 255.0).astype(np.uint8))
        buffered = io.BytesIO()
        prep_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        preprocessed_image_url = f"data:image/png;base64,{img_base64}"
            
        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities_list,
            "preprocessed_image": preprocessed_image_url
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == "__main__":
    # Run the server on host 0.0.0.0 and port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
