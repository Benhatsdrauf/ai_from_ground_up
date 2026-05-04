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
import matplotlib.pyplot as plt


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

# ----------------------------------------------------------------------------
# Evaluate on the test set
# ----------------------------------------------------------------------------

# Forward pass on the entire test set (10,000 images at once — no batching needed for inference)
_, _, _, Y_hat_test = forward(X_test, W1, b1, W2, b2)
predictions = Y_hat_test.argmax(axis=1)

test_accuracy = (predictions == y_test).mean()
print(f"\nTest accuracy: {test_accuracy:.4f}  ({(predictions == y_test).sum()} / {len(y_test)} correct)")

# Find indices where the model was wrong
wrong = np.where(predictions != y_test)[0]
print(f"Total misclassified: {len(wrong)} ({len(wrong) / len(y_test):.2%} error rate)")

# Plot the first 10 misclassified images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for ax, idx in zip(axes.flat, wrong[:10]):
    ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    ax.set_title(f"True: {y_test[idx]}\nPred: {predictions[idx]}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("misclassified.png", dpi=120)
plt.show()

# ----------------------------------------------------------------------------
# Visualize what hidden layer 1 learned
# ----------------------------------------------------------------------------

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, j in zip(axes.flat, range(16)):
    weights = W1[:, j].reshape(28, 28)  # weight pattern for hidden neuron j
    ax.imshow(weights, cmap="seismic")  # red = positive, blue = negative, white ≈ 0
    ax.set_title(f"Hidden #{j}")
    ax.axis("off")

plt.suptitle("First-layer weight patterns — what each hidden neuron is 'looking for'")
plt.tight_layout()
plt.savefig("weights_layer1.png", dpi=120)
plt.show()

# ----------------------------------------------------------------------------
# Visualize what hidden layer 1 learned (improved)
# ----------------------------------------------------------------------------

# Pick the 16 neurons with the largest weight magnitude — the most "specialized" ones.
# (Neurons whose weights are all near zero learned little; skip them.)
weight_magnitudes = np.abs(W1).sum(axis=0)
top_neurons = np.argsort(weight_magnitudes)[-16:]   # indices of top 16

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, j in zip(axes.flat, top_neurons):
    weights = W1[:, j].reshape(28, 28)
    # Symmetric scale around 0 so red/blue have equal weight
    abs_max = np.abs(weights).max()
    ax.imshow(weights, cmap="seismic", vmin=-abs_max, vmax=abs_max)
    ax.set_title(f"Hidden #{j}")
    ax.axis("off")

plt.suptitle("Top 16 most-active hidden neurons (W1) — what each is 'looking for'")
plt.tight_layout()
plt.savefig("weights_layer1.png", dpi=120)
plt.show()

# ----------------------------------------------------------------------------
# Visualize the forward pass for one specific image
# ----------------------------------------------------------------------------

idx = 0  # try changing this to pick different test images
img = X_test[idx]
true_label = y_test[idx]

# Run forward pass and capture intermediates
Z1_one, A1_one, Z2_one, Y_hat_one = forward(img.reshape(1, -1), W1, b1, W2, b2)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

# 1) The input image
axes[0].imshow(img.reshape(28, 28), cmap="gray")
axes[0].set_title(f"Input image\n(true label: {true_label})")
axes[0].axis("off")

# 2) Hidden activations after ReLU — 128 numbers
axes[1].bar(range(128), A1_one[0])
axes[1].set_title("Hidden layer A1\n(after ReLU)")
axes[1].set_xlabel("Hidden neuron #")
axes[1].set_ylabel("Activation")

# 3) Output logits BEFORE softmax — 10 numbers
axes[2].bar(range(10), Z2_one[0])
axes[2].set_title("Output logits Z2\n(before softmax)")
axes[2].set_xlabel("Class")
axes[2].set_ylabel("Logit value")

# 4) Final probabilities AFTER softmax — 10 numbers summing to 1
axes[3].bar(range(10), Y_hat_one[0])
axes[3].set_title(f"Probabilities Y_hat\n(predicted: {Y_hat_one[0].argmax()})")
axes[3].set_xlabel("Class")
axes[3].set_ylabel("Probability")
axes[3].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("forward_pass.png", dpi=120)
plt.show()

# ----------------------------------------------------------------------------
# Schematic architecture diagram (decorative)
# ----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 7))
ax.axis("off")
ax.set_xlim(-0.5, 4)
ax.set_ylim(-1.3, 1.3)

# We can't show all 784/128/10 nodes — show simplified counts.
layer_x = [0, 1.8, 3.6]
layer_show_n = [7, 8, 10]   # how many circles to draw per layer
layer_actual = [784, 128, 10]
layer_names = ["Input\n(784 pixels)", "Hidden\n(128 ReLU)", "Output\n(10 softmax)"]

positions = []
for x, n_show in zip(layer_x, layer_show_n):
    ys = np.linspace(-1, 1, n_show)
    for y in ys:
        ax.plot(x, y, 'o', markersize=18, color='lightblue', zorder=2)
    positions.append([(x, y) for y in ys])

# Connections between layers (simplified — only the visible nodes)
for li in range(2):
    for (x1, y1) in positions[li]:
        for (x2, y2) in positions[li + 1]:
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.15, linewidth=0.5, zorder=1)

# Layer labels
for x, label in zip(layer_x, layer_names):
    ax.text(x, -1.2, label, ha='center', va='top', fontsize=11)

ax.set_title("Network architecture (schematic — actual: 784 → 128 → 10)")
plt.tight_layout()
plt.savefig("architecture.png", dpi=120)
plt.show()