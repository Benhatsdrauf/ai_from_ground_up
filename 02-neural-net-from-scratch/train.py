import numpy as np
from sklearn.datasets import fetch_openml

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    # Step 1: numerical stability — subtract the max along the last axis.
    # axis=-1 means "the last axis" (works for both 1D and 2D).
    # keepdims=True keeps the shape so broadcasting works in the next line.
    z_shifted = z - z.max(axis=-1, keepdims=True)

    # Step 2: exponentiate every element. NumPy: np.exp(...)
    exp_z = np.exp(z_shifted)

    # Step 3: divide by the sum along the last axis (use keepdims=True again).
    return exp_z / exp_z.sum(axis=-1, keepdims=True) 

def forward(X, W1, b1, W2, b2):
    # Layer 1: linear, then ReLU
    Z1 = X @ W1 + b1     # shape: (batch, 128)
    A1 = relu(Z1)        # shape: (batch, 128)

    # Layer 2: linear, then softmax
    Z2 = A1 @ W2 + b2    # shape: (batch, 10)
    Y_hat = softmax(Z2)  # shape: (batch, 10) — probabilities

    return Z1, A1, Z2, Y_hat

def one_hot(y, num_classes=10):
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out

def cross_entropy(Y_hat, Y_true_onehot):
    # Clip to avoid log(0)
    Y_hat = np.clip(Y_hat, 1e-7, 1 - 1e-7) # Clip to avoid log(0) and log(1)

    # YOUR CODE HERE — return the cross-entropy loss as a single number
    return -np.sum(Y_true_onehot * np.log(Y_hat)) / len(Y_hat) # Average over the batch 

mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Pixel values come in as 0–255 ints. Normalize to 0–1 floats.
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int)

# Train/test split — MNIST is conventionally split as first 60K / last 10K.
# (No shuffle needed; the dataset is already shuffled.)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Initialize random weights — small values so initial logits aren't huge
np.random.seed(42)
W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros(128)
W2 = np.random.randn(128, 10) * 0.01
b2 = np.zeros(10)


batch = X_train[:64]
y_batch = y_train[:64]

_, _, _, Y_hat = forward(batch, W1, b1, W2, b2)
Y_onehot = one_hot(y_batch)

loss = cross_entropy(Y_hat, Y_onehot)
print(f"Initial loss on a random model: {loss:.4f}")
