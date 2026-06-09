================================================================================
DIGIT RECOGNIZER TRAINING PIPELINE
================================================================================

This folder contains the training scripts, data configuration, and logs for
the Convolutional Neural Network (CNN) used to identify handwritten digits.

================================================================================
FILES
================================================================================

  train_enhanced.py   - Active training script. Contains upgraded data
                        augmentation, improved 3-block CNN architecture, and
                        advanced training loop (Adam, OneCycleLR, Label
                        Smoothing).

  Phase2.py           - Baseline CNN training pipeline (original script).

  main.py             - Baseline Logistic Regression model training script.

  data/               - Directory containing downloaded raw MNIST training data.

================================================================================
DATASET & AUGMENTATION
================================================================================

The model is trained on the standard MNIST database of handwritten digits.

Data Specs:
  - Training set : 60,000 images (28 x 28 pixels, single-channel grayscale)
  - Test set     : 10,000 images (28 x 28 pixels, single-channel grayscale)
  - Normalization: Pixel values normalized to mean=0.1307, std=0.3081
                   to match PyTorch/MNIST training standards.

Virtual Data Augmentation:
To expand training variety and prevent overfitting, online data augmentation
is applied dynamically to the training loader:

  1. Random Rotation   - Up to +/- 15 degrees
  2. Random Affine     - Translation up to +/- 10% in X and Y directions
  3. Random Scaling    - Scale between 85% and 115% of original size
  4. Random Shear      - Shear angles up to 10 degrees

No augmentation is applied to the test set (only normalization).

================================================================================
MODEL ARCHITECTURE
================================================================================

Enhanced CNN with 1,148,362 trainable parameters.

  Layer            | Input Shape    | Output Shape   | Details
  -----------------|----------------|----------------|---------------------------
  Input            | 1 x 28 x 28   | 1 x 28 x 28   | Grayscale normalized tensor
  Conv Block 1     | 1 x 28 x 28   | 32 x 14 x 14  | Conv2D(3x3, pad=1)
                   |                |                | -> BatchNorm -> ReLU
                   |                |                | -> MaxPool2D(2x2)
  Conv Block 2     | 32 x 14 x 14  | 64 x 7 x 7    | Conv2D(3x3, pad=1)
                   |                |                | -> BatchNorm -> ReLU
                   |                |                | -> MaxPool2D(2x2)
  Conv Block 3     | 64 x 7 x 7    | 128 x 4 x 4   | Conv2D(3x3, pad=1)
                   |                |                | -> BatchNorm -> ReLU
                   |                |                | -> AdaptiveAvgPool(4x4)
  Classifier Head  | 2048           | 512            | Flatten -> Linear(2048->512)
                   |                |                | -> BatchNorm1D -> ReLU
                   |                |                | -> Dropout(40%)
  Output Head      | 512            | 10             | Linear(512->10) logits

================================================================================
HYPERPARAMETERS & TRAINING LOOP
================================================================================

  Batch Size         : 128
  Epochs             : 15
  Optimizer          : Adam
  Weight Decay       : 1e-4  (L2 Regularization)
  Learning Rate      : OneCycleLR scheduler
                       Max LR  = 0.003
                       Warm-up = first 30% of total steps
  Loss Function      : CrossEntropyLoss with Label Smoothing (0.1)
  Gradient Clipping  : Norm threshold = 1.0
  Device             : CUDA (GPU) if available, else CPU

================================================================================
HOW TO RUN TRAINING
================================================================================

Run from the project root directory:

    python training/train_enhanced.py

This will:
  1. Load and initialize the MNIST dataset from training/data/
  2. Train the CNN model for 15 epochs with augmentation
  3. Save the final model weights to:  backend/mnist_cnn.pth

================================================================================
FINAL MODEL ACCURACY METRICS
================================================================================

Training completed with the following epoch outcomes:

  ======================================================================
  Training for 15 epochs  |  batch=128  |  max_lr=0.003
  ======================================================================
  Epoch [ 1/15] Train Loss: 0.7552  Acc: 92.37% | Test Loss: 0.5929  Acc: 98.45% <<best>>
  Epoch [ 2/15] Train Loss: 0.6264  Acc: 97.22% | Test Loss: 0.5703  Acc: 98.89% <<best>>
  Epoch [ 3/15] Train Loss: 0.6044  Acc: 97.79% | Test Loss: 0.5481  Acc: 99.27% <<best>>
  Epoch [ 4/15] Train Loss: 0.5931  Acc: 98.09% | Test Loss: 0.5420  Acc: 99.29% <<best>>
  Epoch [ 5/15] Train Loss: 0.5827  Acc: 98.33% | Test Loss: 0.5489  Acc: 99.07%
  Epoch [ 6/15] Train Loss: 0.5783  Acc: 98.40% | Test Loss: 0.5325  Acc: 99.45% <<best>>
  Epoch [ 7/15] Train Loss: 0.5746  Acc: 98.42% | Test Loss: 0.5321  Acc: 99.44%
  Epoch [ 8/15] Train Loss: 0.5695  Acc: 98.54% | Test Loss: 0.5278  Acc: 99.49% <<best>>
  Epoch [ 9/15] Train Loss: 0.5664  Acc: 98.64% | Test Loss: 0.5317  Acc: 99.43%
  Epoch [10/15] Train Loss: 0.5612  Acc: 98.73% | Test Loss: 0.5298  Acc: 99.43%
  Epoch [11/15] Train Loss: 0.5563  Acc: 98.86% | Test Loss: 0.5237  Acc: 99.49%
  Epoch [12/15] Train Loss: 0.5504  Acc: 98.97% | Test Loss: 0.5200  Acc: 99.57% <<best>>
  Epoch [13/15] Train Loss: 0.5455  Acc: 99.12% | Test Loss: 0.5175  Acc: 99.63% <<best>>
  Epoch [14/15] Train Loss: 0.5400  Acc: 99.27% | Test Loss: 0.5164  Acc: 99.61%
  Epoch [15/15] Train Loss: 0.5388  Acc: 99.28% | Test Loss: 0.5157  Acc: 99.64% <<best>>

  [OK] Model saved to: backend/mnist_cnn.pth
       Best test accuracy: 99.64%

================================================================================
