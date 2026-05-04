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

Z1, A1, _, Y_hat = forward(batch, W1, b1, W2, b2)
Y_onehot = one_hot(y_batch)

# Backprop step 1 — gradient at the output layer
dZ2 = Y_hat - Y_onehot

print("dZ2 shape:", dZ2.shape, "(should be (64, 10))")
print("dZ2 first row:", dZ2[0])
print("y_batch[0] (correct class for image 0):", y_batch[0])

batch_size = len(batch)   # 64

# Backprop step 2 — gradients for the second layer
dW2 = A1.T @ dZ2 / batch_size
db2 = dZ2.mean(axis=0)

print("dW2 shape:", dW2.shape, "(should match W2: (128, 10))")
print("db2 shape:", db2.shape, "(should match b2: (10,))")
print("\ndb2 sample:", db2)

dA1 = dZ2 @ W2.T

print("dA1 shape:", dA1.shape, "(should match A1: (64, 128))")
print("dA1 sample (first 5 values of first row):", dA1[0][:5])


# Backprop step 4 — through ReLU
dZ1 = dA1 * (Z1 > 0)

print("dZ1 shape:", dZ1.shape, "(should match Z1: (64, 128))")
print("Fraction of dZ1 elements that are zero:", (dZ1 == 0).mean())

# Backprop step 5 — gradients for the first layer
dW1 = batch.T @ dZ1 / batch_size
db1 = dZ1.mean(axis=0)

print("dW1 shape:", dW1.shape, "(should match W1: (784, 128))")
print("db1 shape:", db1.shape, "(should match b1: (128,))")

def numerical_grad(param, idx, X_b, Y_onehot_b, W1, b1, W2, b2, eps=1e-5):
    """
    Compute numerical gradient at param[idx] by finite differences.
    Mutates param temporarily, then restores it.
    """
    original = param[idx]

    param[idx] = original + eps
    _, _, _, Yh_plus = forward(X_b, W1, b1, W2, b2)
    loss_plus = cross_entropy(Yh_plus, Y_onehot_b)

    param[idx] = original - eps
    _, _, _, Yh_minus = forward(X_b, W1, b1, W2, b2)
    loss_minus = cross_entropy(Yh_minus, Y_onehot_b)

    param[idx] = original   # restore — important!
    return (loss_plus - loss_minus) / (2 * eps)


print("\n=== GRADIENT CHECK ===")
np.random.seed(0)

print("\ndW2 checks:")
for _ in range(3):
    i, j = np.random.randint(W2.shape[0]), np.random.randint(W2.shape[1])
    num = numerical_grad(W2, (i, j), batch, Y_onehot, W1, b1, W2, b2)
    ana = dW2[i, j]
    print(f"  W2[{i:3d},{j:2d}]:  analytical={ana:+.8f}  numerical={num:+.8f}  diff={abs(ana-num):.2e}")

print("\ndW1 checks:")
for _ in range(3):
    i, j = np.random.randint(W1.shape[0]), np.random.randint(W1.shape[1])
    num = numerical_grad(W1, (i, j), batch, Y_onehot, W1, b1, W2, b2)
    ana = dW1[i, j]
    print(f"  W1[{i:3d},{j:2d}]:  analytical={ana:+.8f}  numerical={num:+.8f}  diff={abs(ana-num):.2e}")

print("\ndW1 re-check (center pixels — should be non-zero):")
# Pixel positions in the middle of the image (rows 8-20 of 28×28)
center_pixels = [350, 380, 410, 440]   # spread across the middle
for i in center_pixels:
    j = np.random.randint(W1.shape[1])
    num = numerical_grad(W1, (i, j), batch, Y_onehot, W1, b1, W2, b2)
    ana = dW1[i, j]
    print(f"  W1[{i:3d},{j:2d}]:  analytical={ana:+.8f}  numerical={num:+.8f}  diff={abs(ana-num):.2e}")

print("\ndb2 checks:")
for _ in range(3):
    j = np.random.randint(b2.shape[0])
    num = numerical_grad(b2, j, batch, Y_onehot, W1, b1, W2, b2)
    ana = db2[j]
    print(f"  b2[{j}]:  analytical={ana:+.8f}  numerical={num:+.8f}  diff={abs(ana-num):.2e}")

print("\ndb1 checks:")
for _ in range(3):
    j = np.random.randint(b1.shape[0])
    num = numerical_grad(b1, j, batch, Y_onehot, W1, b1, W2, b2)
    ana = db1[j]
    print(f"  b1[{j}]:  analytical={ana:+.8f}  numerical={num:+.8f}  diff={abs(ana-num):.2e}")