import numpy as np
from sklearn.datasets import fetch_california_housing

def predict(X, w, b):
    #Predict the price of a house given its features.
    return X @ w + b    

def mse_loss(predicted_prices, actual_prices):
    #Calculate the mean squared error between predicted and actual prices.
    errors        = predicted_prices - actual_prices     # how wrong, with sign
    squared       = errors ** 2          # all positive, big errors blown up
    loss          = squared.mean()       # one number summarising "how bad on average"
    return loss
    # could also be written as a one liner 
    # return np.mean((predicted_prices - actual_prices) ** 2)


def calculate_gradient(X, predicted_prices, actual_prices):
    #Calculate the gradient of the loss with respect to w and b.
    N = len(X)
    errors = predicted_prices - actual_prices
    w_gradient = (2/N) * np.dot(X.T, errors)  # dL/dw
    b_gradient = (2/N) * errors.sum()            # dL/db
    return w_gradient, b_gradient


# Load the California housing dataset
dataset = fetch_california_housing()
X = dataset.data        # shape (20640, 8) — 20,640 houses, 8 features each
y = dataset.target      # shape (20640,) — median house value, in units of $100,000

# Look at what we have
print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nFeature names:", dataset.feature_names)
print("\nFirst 3 houses (raw values):")
print(X[:3])
print("\nFirst 3 target prices:", y[:3])

# Normalize each feature: subtract column mean, divide by column std
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Verify: every feature should have mean ≈ 0 and std ≈ 1
print("\n--- after normalization ---")
print("Per-feature means:", X.mean(axis=0))
print("Per-feature stds: ", X.std(axis=0))

# Train/test split: 80/20
n_samples = X.shape[0]
np.random.seed(42)                      # reproducible split
indices = np.random.permutation(n_samples)
split_at = int(0.8 * n_samples)

train_idx = indices[:split_at]
test_idx  = indices[split_at:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("Train:", X_train.shape, y_train.shape)
print("Test: ", X_test.shape,  y_test.shape)


 # Initialize: 8 weights now, not 1
n_features = X_train.shape[1]   # = 8
w = np.zeros(n_features)
b = 0.0
learning_rate = 0.01            # normal lr now! Normalization made this possible.

losses = []

for epoch in range(1000):
    y_pred = predict(X_train, w, b)
    loss   = mse_loss(y_pred, y_train)
    grad_w, grad_b = calculate_gradient(X_train, y_pred, y_train)

    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b
    losses.append(loss)

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}  train_loss={loss:.4f}")

# Final evaluation on the held-out test set
test_predictions = predict(X_test, w, b)
test_loss = mse_loss(test_predictions, y_test)

print(f"\nFinal train loss: {losses[-1]:.4f}")
print(f"Test loss:        {test_loss:.4f}")
print(f"\nLearned weights (one per feature):")
for name, weight in zip(dataset.feature_names, w):
    print(f"  {name:12s} {weight:+.4f}")
print(f"  bias         {b:+.4f}")
