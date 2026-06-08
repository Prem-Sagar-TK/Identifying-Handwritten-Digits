# Handwritten Digit Identifier (MNIST)

A complete full-stack web application that allows users to draw or upload images of handwritten digits and identifies them in real-time. The system uses a deep Convolutional Neural Network (CNN) trained in PyTorch on the MNIST dataset, with a Flask backend server and a responsive, warm-linen themed React/Vite frontend client.

---

## 🎨 Warm Linen Design Theme

The interface features a custom minimalist design designed to feel premium and editorial:
- **Background**: `#F7F4EF` (Light linen cream)
- **Primary Text & Canvas**: `#231F20` (Dark espresso black)
- **Accent Lines**: `#4A3E3D` (Deep matte brown)
- **Secondary Items**: `#8A827A` (Taupe gray)
- **Interactive Elements & Buttons**: `#EFEBE4` backgrounds with `#5C534C` text.

---

## 📁 Repository Structure

```text
├── backend/
│   ├── backend_app.py     # Flask Server (Inference API)
│   ├── requirements.txt   # Python Dependencies
│   └── mnist_cnn.pth      # Trained model state dict (PyTorch CNN)
├── frontend/
│   ├── src/               # React client components & UI styling
│   ├── package.json       # Node dependency sheet
│   └── index.html
├── training/
│   ├── main.py            # Phase 1: Simple PyTorch model training (MLP)
│   └── Phase2.py          # Phase 2: CNN Training pipeline with 99.1% accuracy
├── Phase3.py              # Orchestrator script to run backend & frontend together
└── data/                  # Standard folder for model artifacts / training data
```

---

## 🚀 Getting Started

### Prerequisites

- **Python**: version 3.8 or above
- **Node.js**: version 18 or above (with `npm`)

### The Quick Way (Recommended)

To run both the frontend and the backend simultaneously using a single terminal window, use the unified coordinator script:

```powershell
python Phase3.py
```

This script will spin up both servers, print out their respective outputs with prefix labels (`[BACKEND]` / `[FRONTEND]`), and cleanly stop both of them if you press `Ctrl+C`.

---

## 🔧 Manual Step-by-Step Setup

If you prefer to run the client and the server manually in separate terminal windows, follow these instructions:

### 1. Flask Backend Setup

1. Navigate to the `backend` folder:
   ```powershell
   cd backend
   ```
2. Install the Python requirements:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run the development server:
   ```powershell
   python backend_app.py
   ```
   *The backend will boot on `http://localhost:5000` with the health endpoint visible at `/health`.*

### 2. Frontend React Client Setup

1. Navigate to the `frontend` folder:
   ```powershell
   cd frontend
   ```
2. Install dependencies:
   ```powershell
   npm install
   ```
3. Start the dev server:
   ```powershell
   npm run dev
   ```
   *The dev server will boot on `http://localhost:5173`.*

---

## 🧠 Model Pipeline & Accuracy

- **Architecture**: Convolutional Neural Network (CNN) built in PyTorch (refer to training scripts in `training/`).
- **Layers**: Two convolution-relu-maxpool blocks followed by linear classification layers.
- **Accuracy**: $\approx 99.1\%$ on the test split of the MNIST handwritten digit database.
- **Preprocessing**: Input images drawn or uploaded are automatically converted to 8-bit grayscale, downsampled/upsampled to 28x28 pixels, auto-inverted if a light background color is used (to match the MNIST white-on-black format), normalized to the range `[0.0, 1.0]`, and converted to tensors.
