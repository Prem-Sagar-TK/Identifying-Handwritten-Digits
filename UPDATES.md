# Project Updates & Changelog

A complete record of every change made to the **Handwritten Digit Recognizer** project.

---

## Problem Statement — Why the App Was Broken

### Observed Symptoms

After drawing a digit on the canvas and clicking **Predict**, the app returned wrong results consistently:

- **Output probabilities were nearly uniform** (each digit scored ~10%) regardless of what was drawn.
- **Confidence scores were very low** (typically 10–15%).
- The model occasionally returned a random digit with no correlation to what was drawn.
- Single-click taps/dots left **no visible mark** on the canvas.

### Root Cause Summary

The trained CNN model (99.64% accuracy on MNIST test data) was working correctly. The failures were entirely in the **inference pipeline** — the way drawn images were processed before being fed to the model:

1. **The model was trained with MNIST normalization** (`mean=0.1307, std=0.3081` applied after ÷255). The backend **never applied this normalization**, so the model received raw pixel values in `[0,1]` — a distribution it had never seen during training. This alone was enough to cause near-uniform probability outputs.

2. **MNIST digits are centred with padding**. The backend was directly resizing the 280×280 canvas to 28×28, which squashed and distorted the digit to fill the entire frame — a spatial layout completely different from the training data.

3. **The canvas inversion check was flawed**. The logic to detect and flip black-on-white images used a mean threshold of 127.5, but a white-on-black canvas drawing has a mean of 5–20 after downsampling (most pixels are black). This sometimes incorrectly inverted already-correct images.

4. **PNG exports from canvas may be RGBA**. Converting RGBA directly to grayscale without first compositing the alpha channel caused transparent pixels to be computed as mid-grey, corrupting the digit edges.

5. **React async state bug** in the canvas component caused single clicks to draw nothing, making it harder to draw precise digits like `1`.

All five issues are now fully resolved as documented below.

---

## Version 2.0 — Enhanced Model + Full Pipeline Bugfix

### Date: 2026-06-09

---

## 1. New Training Script — `training/train_enhanced.py`

A brand-new training script was created to replace the original `training/Phase2.py` baseline.

### What changed

| Feature | Phase2.py (Before) | train_enhanced.py (After) |
| :--- | :--- | :--- |
| **Architecture** | 2 Conv blocks, no BatchNorm, no Dropout | 3 Conv blocks + BatchNorm + Dropout(0.4) |
| **Pooling** | MaxPool2d on all blocks | AdaptiveAvgPool2d on final block |
| **Classifier** | Flatten → FC(128) → FC(10) | Flatten → FC(512) → BN → ReLU → Dropout → FC(10) |
| **Trainable Params** | ~421k | 1,148,362 |
| **Epochs** | 5 | 15 |
| **Optimizer** | SGD | Adam (weight decay = 1e-4) |
| **Learning Rate** | Fixed 0.001 | OneCycleLR (max_lr = 0.003) |
| **Loss Function** | CrossEntropyLoss | CrossEntropyLoss + Label Smoothing (0.1) |
| **Data Augmentation** | None | RandomRotation, RandomAffine (translate, scale, shear) |
| **Gradient Clipping** | None | Norm threshold = 1.0 |
| **Test Accuracy** | ~98.5% | **99.64%** |

### Data Augmentation Details

The following transforms are applied **dynamically per batch** during training to simulate real-world handwriting variation:

```python
transforms.RandomRotation(degrees=15)
transforms.RandomAffine(
    degrees=0,
    translate=(0.1, 0.1),   # ±10% horizontal/vertical shift
    scale=(0.85, 1.15),     # ±15% zoom
    shear=10,               # ±10° shear
)
transforms.ToTensor()
transforms.Normalize(mean=(0.1307,), std=(0.3081,))
```

No augmentation is applied to the **test set** (only normalize).

### Training Results

```
Epoch [ 1/15] Train: 92.37%  |  Test: 98.45%
Epoch [ 2/15] Train: 97.22%  |  Test: 98.89%
Epoch [ 3/15] Train: 97.79%  |  Test: 99.27%
Epoch [ 4/15] Train: 98.09%  |  Test: 99.29%
Epoch [ 5/15] Train: 98.33%  |  Test: 99.07%
Epoch [ 6/15] Train: 98.40%  |  Test: 99.45%
Epoch [ 7/15] Train: 98.42%  |  Test: 99.44%
Epoch [ 8/15] Train: 98.54%  |  Test: 99.49%
Epoch [ 9/15] Train: 98.64%  |  Test: 99.43%
Epoch [10/15] Train: 98.73%  |  Test: 99.43%
Epoch [11/15] Train: 98.86%  |  Test: 99.49%
Epoch [12/15] Train: 98.97%  |  Test: 99.57%
Epoch [13/15] Train: 99.12%  |  Test: 99.63%
Epoch [14/15] Train: 99.27%  |  Test: 99.61%
Epoch [15/15] Train: 99.28%  |  Test: 99.64%  ← Best
```

The trained model weights are stored at: `backend/mnist_cnn.pth` (~4.6 MB)

---

## 2. Backend Preprocessing Bugfix — `backend/backend_app.py`

The backend `preprocess_image()` function had **4 critical bugs** causing near-uniform prediction probabilities. All are now fixed.

---

### Bug 1 — Missing MNIST Normalization *(Critical)*

**File:** `backend/backend_app.py`

**Problem:** The training pipeline applies `Normalize(mean=0.1307, std=0.3081)` after dividing by 255, shifting pixel values from `[0, 1]` to approximately `[-0.42, 2.82]`. The backend only divided by 255 and **never applied this normalization**, so the model received pixel values in `[0, 1]` — a completely different input distribution from training. This was the primary cause of uniform output probabilities.

```python
# BEFORE (broken) — no normalization
img_np = img_np / 255.0
tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)

# AFTER (fixed) — matches training exactly
MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081
img_np = np.array(image_28, dtype=np.float32) / 255.0
img_np = (img_np - MNIST_MEAN) / MNIST_STD          # ← added
tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
```

---

### Bug 2 — No MNIST-Style Centring Before Resize *(Critical)*

**File:** `backend/backend_app.py`

**Problem:** MNIST training images always have the digit centred in a 28×28 frame with ~4px black padding on all sides. The backend used a raw `resize(280→28)` which stretched the digit to fill the **entire** 28×28 frame — a completely different spatial layout from training data.

```python
# BEFORE (broken) — raw resize, no centring
image = image.resize((28, 28), resample_filter)

# AFTER (fixed) — bounding-box crop, then paste centred with 4px padding
image_28 = centre_digit_on_black(image_gray, target=28, padding=4)
```

The new `centre_digit_on_black()` function:
1. Finds the bounding box of all bright (digit) pixels.
2. Crops tightly to the digit.
3. Resizes the crop to fit inside 20×20 (28 − 2×4 padding).
4. Pastes it centred on a blank 28×28 black canvas.

---

### Bug 3 — Unreliable Inversion Heuristic *(Bug)*

**File:** `backend/backend_app.py`

**Problem:** The original `if mean > 127.5` check was supposed to detect black-on-white images and invert them. However, a canvas drawing (white digit on black) downsampled from 280×280 to 28×28 has a very low mean (most pixels are black), so the mean was typically 5–20 — well below 127.5. The condition was sometimes wrongly triggered, inverting a correct image and producing a black-on-white input the model had never seen.

```python
# BEFORE (broken) — wrong threshold
if np.mean(img_np) > 127.5:
    img_np = 255.0 - img_np

# AFTER (fixed) — only invert for clearly paper-white backgrounds
if mean_val > 200:   # clearly white-background scanned/photographed image
    image_gray = ImageOps.invert(image_gray)
```

---

### Bug 4 — RGBA Alpha Channel Corruption *(Bug)*

**File:** `backend/backend_app.py`

**Problem:** `canvas.toBlob('image/png')` can produce RGBA PNGs. Converting RGBA directly to `'L'` (greyscale) via PIL computes luminosity from the RGB channels but the alpha values can affect the result, making transparent areas appear as mid-grey instead of black — corrupting the digit boundary.

```python
# BEFORE (broken) — RGBA converted directly to 'L'
image = image.convert("L")

# AFTER (fixed) — composite RGBA on black RGB first
if image.mode == "RGBA":
    background = Image.new("RGB", image.size, (0, 0, 0))
    background.paste(image, mask=image.split()[3])
    image = background
image_gray = image.convert("L")
```

---

## 3. Frontend Canvas Bugfix — `frontend/src/components/DrawingCanvas.jsx`

### Bug 5 — Single Click / Tap Draws No Dot *(Bug)*

**File:** `frontend/src/components/DrawingCanvas.jsx`

**Problem:** `startDrawing` called `draw(e)` to render the initial dot, but `setIsDrawing(true)` is a React state update and is **asynchronous**. When `draw(e)` ran, `isDrawing` was still `false`, so the `if (!isDrawing) return` guard immediately exited without drawing anything. Single taps or precise clicks left no mark on the canvas.

```jsx
// BEFORE (broken) — draw() exits immediately because isDrawing is still false
const startDrawing = (e) => {
  setIsDrawing(true);
  draw(e);   // ← isDrawing is still false here, draws nothing
};

// AFTER (fixed) — draw the initial dot directly in startDrawing
const startDrawing = (e) => {
  setIsDrawing(true);
  const ctx = canvas.getContext('2d');
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + 0.1, y + 0.1);   // tiny offset makes lineTo register a dot
  ctx.stroke();
};
```

---

## 4. Upgraded CNN Architecture — `backend/backend_app.py`

The `CNN` class in the backend was updated to match the new `train_enhanced.py` architecture exactly. `load_state_dict()` requires the class definition to be an exact structural match.

| Layer | Old (Phase2) | New (Enhanced) |
| :--- | :--- | :--- |
| Conv Block 1 | Conv→ReLU→MaxPool | Conv→**BN**→ReLU→MaxPool |
| Conv Block 2 | Conv→ReLU→MaxPool | Conv→**BN**→ReLU→MaxPool |
| Conv Block 3 | *(none)* | Conv(64→128)→**BN**→ReLU→**AdaptiveAvgPool(4×4)** |
| Classifier | FC(3136→128)→ReLU→FC(10) | FC(2048→**512**)→**BN**→ReLU→**Dropout(0.4)**→FC(10) |

---

## 5. Files Changed Summary

| File | Change Type | Description |
| :--- | :--- | :--- |
| `training/train_enhanced.py` | **New** | Full enhanced training script |
| `training/README.md` | **New** | Training pipeline documentation |
| `backend/mnist_cnn.pth` | **Replaced** | New model weights (99.64% accuracy) |
| `backend/backend_app.py` | **Rewritten** | Fixed 4 preprocessing bugs + CNN architecture match |
| `frontend/src/components/DrawingCanvas.jsx` | **Fixed** | Single-click dot drawing bug |
| `UPDATES.md` | **New** | This file |

---

## How to Re-train

```powershell
# From the project root
python training\train_enhanced.py
```

The script saves the new model to `backend/mnist_cnn.pth` automatically.

## How to Run the App

```powershell
# From the project root — starts both Flask backend and Vite frontend
python Phase3.py
```

- Backend: `http://localhost:5000`
- Frontend: `http://localhost:5173`
