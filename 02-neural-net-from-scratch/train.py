"""
Project 2 — Neural Network from Scratch
========================================

A 2-layer fully-connected neural network for MNIST digit classification,
implemented in pure NumPy. No PyTorch, no autograd — every gradient is
derived and computed by hand.

Architecture:   784 (input pixels) → 128 (hidden, ReLU) → 10 (output, softmax)
Loss:           cross-entropy
Optimizer:      mini-batch gradient descent (batch size 64, lr 0.1)
"""

import numpy as np
from sklearn.datasets import fetch_openml


# ----------------------------------------------------------------------------
# Model functions
# ----------------------------------------------------------------------------

def relu(z):
    """ReLU activation — element-wise max(0, z). Negative values become 0."""
    return np.maximum(0, z)


def softmax(z):
    """
    Convert logits to probabilities (each row sums to 1).
    Subtracting the max before exp prevents numerical overflow with large logits.
    """
    z_shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


def forward(X, W1, b1, W2, b2):
    """
    Forward pass through the 2-layer network.
    Returns (Z1, A1, Z2, Y_hat) — Z1 and A1 are cached for backprop.
    """
    Z1 = X @ W1 + b1     # linear layer 1
    A1 = relu(Z1)        # nonlinearity
    Z2 = A1 @ W2 + b2    # linear layer 2
    Y_hat = softmax(Z2)  # logits → probabilities
    return Z1, A1, Z2, Y_hat


def one_hot(y, num_classes=10):
    """Convert integer labels to one-hot vectors. e.g. 5 → [0,0,0,0,0,1,0,0,0,0]."""
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out


def cross_entropy(Y_hat, Y_true_onehot):
    """
    Cross-entropy loss. Clips Y_hat to avoid log(0) blowing up to -inf.
    Returns the mean loss across the batch.
    """
    Y_hat = np.clip(Y_hat, 1e-7, 1 - 1e-7)
    return -np.sum(Y_true_onehot * np.log(Y_hat)) / len(Y_hat)


# ----------------------------------------------------------------------------
# Load MNIST and split
# ----------------------------------------------------------------------------

print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Pixels come in as 0-255 ints. Normalize to 0-1 floats.
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int)

# MNIST is conventionally split as first 60K train, last 10K test.
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


# ----------------------------------------------------------------------------
# Weight initialization
# ----------------------------------------------------------------------------

# He initialization — sqrt(2 / fan_in) keeps activation variance roughly
# constant through ReLU layers. Plain random or *0.01 makes activations vanish.
np.random.seed(42)
W1 = np.random.randn(784, 128) * np.sqrt(2 / 784)
b1 = np.zeros(128)
W2 = np.random.randn(128, 10) * np.sqrt(2 / 128)
b2 = np.zeros(10)


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------

batch_size = 64
learning_rate = 0.1
epochs = 10

losses = []
accuracies = []

print(f"Training for {epochs} epochs (batch size {batch_size}, lr {learning_rate})...")

for epoch in range(epochs):
    # Shuffle each epoch so the model doesn't learn the order of the data.
    perm = np.random.permutation(len(X_train))
    X_shuf = X_train[perm]
    y_shuf = y_train[perm]

    epoch_loss = 0.0
    epoch_correct = 0

    # Mini-batch loop — 60,000 / 64 ≈ 937 updates per epoch.
    for i in range(0, len(X_train), batch_size):
        X_batch = X_shuf[i : i + batch_size]
        y_batch = y_shuf[i : i + batch_size]
        Y_onehot = one_hot(y_batch)

        # Forward pass
        Z1, A1, Z2, Y_hat = forward(X_batch, W1, b1, W2, b2)

        # Track loss and accuracy on the batch.
        loss = cross_entropy(Y_hat, Y_onehot)
        epoch_loss += loss * len(X_batch)
        epoch_correct += (Y_hat.argmax(axis=1) == y_batch).sum()

        # Backward pass — the 7 gradient lines derived in milestone 3.
        dZ2 = Y_hat - Y_onehot                 # softmax + cross-entropy → clean Y_hat - Y
        dW2 = A1.T @ dZ2 / len(X_batch)
        db2 = dZ2.mean(axis=0)
        dA1 = dZ2 @ W2.T                       # pass gradient back through linear layer
        dZ1 = dA1 * (Z1 > 0)                   # backprop through ReLU
        dW1 = X_batch.T @ dZ1 / len(X_batch)
        db1 = dZ1.mean(axis=0)

        # Gradient descent — step opposite the gradient.
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    # End-of-epoch report
    avg_loss = epoch_loss / len(X_train)
    train_acc = epoch_correct / len(X_train)
    losses.append(avg_loss)
    accuracies.append(train_acc)

    print(f"Epoch {epoch + 1:2d}  loss={avg_loss:.4f}  train_acc={train_acc:.4f}")
