# Digit Recognizer Training Pipeline

This folder contains the training scripts, data configuration, and logs for the Convolutional Neural Network (CNN) used to identify handwritten digits.

---

## 📁 Files
- [train_enhanced.py](file:///d:/GitProjects/ML/Identifying%20Handwritten%20Digits/training/train_enhanced.py): The active training script containing our upgraded data augmentation, improved 3-block CNN architecture, and advanced training loop (Adam, OneCycleLR, Label Smoothing).
- [Phase2.py](file:///d:/GitProjects/ML/Identifying%20Handwritten%20Digits/training/Phase2.py): Baseline CNN training pipeline.
- [main.py](file:///d:/GitProjects/ML/Identifying%20Handwritten%20Digits/training/main.py): Baseline Logistic Regression model training script.
- **data/**: Directory containing downloaded raw MNIST training data.

---

## 📊 Dataset & Augmentation

The model is trained on the standard **MNIST database** of handwritten digits. 

### Data Specs:
- **Training set**: 60,000 images ($28 \times 28$ pixels, single-channel grayscale)
- **Test set**: 10,000 images ($28 \times 28$ pixels, single-channel grayscale)
- **Normalization**: Pixel values are normalized to a mean of `0.1307` and standard deviation of `0.3081` to match PyTorch/MNIST training standards.

### Virtual Data Augmentation:
To expand training variety and prevent overfitting, online data augmentation is applied dynamically to the training loader:
1. **Random Rotation**: Up to $\pm 15^\circ$
2. **Random Affine Shifts**: Translation up to $\pm 10\%$ in both $X$ and $Y$ directions
3. **Random Scaling**: Scales between $85\%$ and $115\%$ of original size
4. **Random Shear**: Shear angles up to $10^\circ$

---

## 🧠 Model Architecture

We use a deeper, upgraded Convolutional Neural Network (CNN) with **1,148,362 trainable parameters**:

| Layer Type | Input Shape | Output Shape | Details |
| :--- | :--- | :--- | :--- |
| **Input** | $1 \times 28 \times 28$ | $1 \times 28 \times 28$ | Grayscale normalized tensor |
| **Conv Block 1** | $1 \times 28 \times 28$ | $32 \times 14 \times 14$ | Conv2D ($3\times3$, pad=1) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool2D ($2\times2$) |
| **Conv Block 2** | $32 \times 14 \times 14$ | $64 \times 7 \times 7$ | Conv2D ($3\times3$, pad=1) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool2D ($2\times2$) |
| **Conv Block 3** | $64 \times 7 \times 7$ | $128 \times 4 \times 4$ | Conv2D ($3\times3$, pad=1) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ AdaptiveAvgPool ($4\times4$) |
| **Classifier Head**| $2048$ | $512$ | Flatten $\rightarrow$ Linear ($2048 \rightarrow 512$) $\rightarrow$ BatchNorm1D $\rightarrow$ ReLU $\rightarrow$ Dropout ($40\%$) |
| **Output Head** | $512$ | $10$ | Linear ($512 \rightarrow 10$) logits |

---

## ⚙️ Hyperparameters & Training Loop

- **Batch Size**: 128
- **Learning Rate Scheduler**: OneCycleLR (Max learning rate: `0.003`, starting warm-up phase at `30%` of steps)
- **Weight Decay**: $10^{-4}$ (L2 Regularization)
- **Loss Function**: CrossEntropyLoss with **Label Smoothing (0.1)** to prevent the model from becoming overly confident.
- **Gradient Clipping**: Norm threshold set to `1.0` to ensure training stability.
- **Device**: Automatic GPU/CUDA acceleration if available, otherwise falls back to CPU.

---

## 🚀 How to Run Training

To run the training script from the root workspace directory, run:

```bash
python training/train_enhanced.py
```

This will:
1. Load and initialize the MNIST dataset.
2. Train the CNN model for 15 epochs.
3. Save the best state dictionary weights directly to [backend/mnist_cnn.pth](file:///d:/GitProjects/ML/Identifying%20Handwritten%20Digits/backend/mnist_cnn.pth).

---

## 📈 Final Model Accuracy Metrics

The training completed with the following epoch outcomes:

```text
======================================================================
Training for 15 epochs  |  batch=128  |  max_lr=0.003
======================================================================
Epoch [ 1/15] Train Loss: 0.7552 Acc: 92.37% | Test Loss: 0.5929 Acc: 98.45% <<best>>
Epoch [ 2/15] Train Loss: 0.6264 Acc: 97.22% | Test Loss: 0.5703 Acc: 98.89% <<best>>
Epoch [ 3/15] Train Loss: 0.6044 Acc: 97.79% | Test Loss: 0.5481 Acc: 99.27% <<best>>
Epoch [ 4/15] Train Loss: 0.5931 Acc: 98.09% | Test Loss: 0.5420 Acc: 99.29% <<best>>
Epoch [ 5/15] Train Loss: 0.5827 Acc: 98.33% | Test Loss: 0.5489 Acc: 99.07%
Epoch [ 6/15] Train Loss: 0.5783 Acc: 98.40% | Test Loss: 0.5325 Acc: 99.45% <<best>>
Epoch [ 7/15] Train Loss: 0.5746 Acc: 98.42% | Test Loss: 0.5321 Acc: 99.44%
Epoch [ 8/15] Train Loss: 0.5695 Acc: 98.54% | Test Loss: 0.5278 Acc: 99.49% <<best>>
Epoch [ 9/15] Train Loss: 0.5664 Acc: 98.64% | Test Loss: 0.5317 Acc: 99.43%
Epoch [10/15] Train Loss: 0.5612 Acc: 98.73% | Test Loss: 0.5298 Acc: 99.43%
Epoch [11/15] Train Loss: 0.5563 Acc: 98.86% | Test Loss: 0.5237 Acc: 99.49%
Epoch [12/15] Train Loss: 0.5504 Acc: 98.97% | Test Loss: 0.5200 Acc: 99.57% <<best>>
Epoch [13/15] Train Loss: 0.5455 Acc: 99.12% | Test Loss: 0.5175 Acc: 99.63% <<best>>
Epoch [14/15] Train Loss: 0.5400 Acc: 99.27% | Test Loss: 0.5164 Acc: 99.61%
Epoch [15/15] Train Loss: 0.5388 Acc: 99.28% | Test Loss: 0.5157 Acc: **99.64%** <<best>>

[OK] Model saved to: backend/mnist_cnn.pth
  Best test accuracy: 99.64%
```
