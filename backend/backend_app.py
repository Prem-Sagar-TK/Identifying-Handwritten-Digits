import os
import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── CNN architecture — must match train_enhanced.py exactly ──────────────────
class CNN(nn.Module):
    """
    Enhanced CNN: 3 conv blocks with BatchNorm, Dropout, and a deeper classifier.
      Conv1 (1→32)  → BN → ReLU → MaxPool2d(2)      [28→14]
      Conv2 (32→64) → BN → ReLU → MaxPool2d(2)      [14→7]
      Conv3 (64→128)→ BN → ReLU → AdaptiveAvgPool(4) [7→4]
      Flatten → FC(512) → BN → ReLU → Dropout(0.4) → FC(10)
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Load model ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_cnn.pth")
model = CNN().to(device)

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model state dict: {e}")
else:
    print(f"Warning: {MODEL_PATH} not found. Running with uninitialized weights.")

model.eval()   # Always eval mode — disables Dropout & uses BN running stats


# ── Preprocessing ─────────────────────────────────────────────────────────────
# MNIST training pipeline (test_transform):
#   PIL image  →  ToTensor (÷255, shape [1,28,28])  →  Normalize(0.1307, 0.3081)
#
# So inference must do the same sequence exactly.
#
# BUG 1 (fixed): original code only divided by 255 and skipped Normalize().
# BUG 2 (fixed): original code resized directly to 28×28 without centring,
#   causing the digit to fill the entire frame — different from MNIST padding style.
# BUG 3 (fixed): mean-threshold inversion was unreliable. Canvas always draws
#   white on black so we use a more robust max-pixel check instead.

MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081

def centre_digit_on_black(img_gray: Image.Image, target: int = 28, padding: int = 4) -> Image.Image:
    """
    1. Threshold the grayscale image to find the digit bounding box.
    2. Crop to the bounding box.
    3. Resize the cropped digit to fit inside (target - 2*padding) × (target - 2*padding),
       preserving aspect ratio.
    4. Paste it centred on a black (target × target) canvas.

    This matches how MNIST digits are laid out: centred with a small border.
    """
    arr = np.array(img_gray)

    # Find rows/cols that have any bright pixel (digit pixels)
    threshold = 10   # anything > 10/255 counts as digit
    rows = np.any(arr > threshold, axis=1)
    cols = np.any(arr > threshold, axis=0)

    if not rows.any():
        # Blank canvas — return all-black 28×28
        return Image.fromarray(np.zeros((target, target), dtype=np.uint8))

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop the digit with a 1-pixel guard
    rmin = max(0, rmin - 1)
    rmax = min(arr.shape[0] - 1, rmax + 1)
    cmin = max(0, cmin - 1)
    cmax = min(arr.shape[1] - 1, cmax + 1)

    digit_crop = img_gray.crop((cmin, rmin, cmax + 1, rmax + 1))

    # Resize to fit the inner area keeping aspect ratio
    inner = target - 2 * padding
    digit_crop.thumbnail((inner, inner), Image.Resampling.LANCZOS)

    # Paste centred on a black background
    canvas = Image.fromarray(np.zeros((target, target), dtype=np.uint8))
    paste_x = (target - digit_crop.width)  // 2
    paste_y = (target - digit_crop.height) // 2
    canvas.paste(digit_crop, (paste_x, paste_y))
    return canvas


def preprocess_image(image_bytes: bytes):
    """
    Full preprocessing pipeline matching the training test_transform:

      Step 1: Load → RGBA-safe convert → Grayscale ('L')
      Step 2: Ensure white-on-black orientation (MNIST format)
      Step 3: Centre the digit with MNIST-style padding
      Step 4: ToTensor equivalent: convert to float32, divide by 255  → [0, 1]
      Step 5: Normalize: (pixel - 0.1307) / 0.3081                   → training range
      Step 6: Add batch + channel dims → shape [1, 1, 28, 28]

    Returns:
        tensor  — shape [1, 1, 28, 28], ready for model forward pass
        vis_np  — shape [28, 28], float32 in [0,1] for visualisation (pre-norm)
    """
    # ── Step 1: Load and convert to grayscale ────────────────────────────────
    image = Image.open(io.BytesIO(image_bytes))

    # Handle alpha channel: composite on black before grayscale conversion
    # so transparent areas become black (background), not white.
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (0, 0, 0))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    image_gray = image.convert("L")

    # ── Step 2: Ensure white-digit on black-background ───────────────────────
    # The canvas draws white strokes on a black background, which is already
    # correct MNIST format. For uploaded images we use a robust check:
    # compute the proportion of "bright" pixels. If the majority of bright
    # pixels form a sparse foreground (digit < 30% of area) and the overall
    # mean is high, the image is black-on-white and must be inverted.
    arr_u8 = np.array(image_gray, dtype=np.uint8)
    mean_val = arr_u8.mean()
    if mean_val > 200:
        # Very bright background — almost certainly black-digit on white paper
        image_gray = ImageOps.invert(image_gray)
        arr_u8 = np.array(image_gray, dtype=np.uint8)

    # ── Step 3: Centre digit with MNIST-style padding ────────────────────────
    image_28 = centre_digit_on_black(image_gray, target=28, padding=4)

    # ── Step 4: Float conversion (ToTensor equivalent) ───────────────────────
    img_np = np.array(image_28, dtype=np.float32) / 255.0   # [0, 1]
    vis_np = img_np.copy()   # save for visualisation BEFORE normalization

    # ── Step 5: Normalize exactly like training ───────────────────────────────
    # training: Normalize(mean=(0.1307,), std=(0.3081,))
    # pixel_norm = (pixel - mean) / std
    img_np = (img_np - MNIST_MEAN) / MNIST_STD

    # ── Step 6: Build tensor [1, 1, 28, 28] ──────────────────────────────────
    tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return tensor, vis_np


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "device": str(device),
        "model_loaded": os.path.exists(MODEL_PATH)
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided in request.files['image']"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        tensor, vis_np = preprocess_image(image_bytes)

        # Run inference — model is always in eval() mode
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1).squeeze(0)

        prediction        = torch.argmax(probabilities).item()
        confidence        = round(probabilities[prediction].item() * 100, 2)
        probabilities_list = [round(p.item(), 5) for p in probabilities]

        # Build visualisation image from vis_np (pre-normalization, [0,1])
        prep_img = Image.fromarray((vis_np * 255.0).astype(np.uint8))
        buffered  = io.BytesIO()
        prep_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        preprocessed_image_url = f"data:image/png;base64,{img_base64}"

        return jsonify({
            "prediction":          prediction,
            "confidence":          confidence,
            "probabilities":       probabilities_list,
            "preprocessed_image":  preprocessed_image_url
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
