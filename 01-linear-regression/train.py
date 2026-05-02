import numpy as np

def predict(sizes, w, b):
    #Predict the price of a house given its size.
    return w * sizes + b    

def mse_loss(predicted_prices, actual_prices):
    #Calculate the mean squared error between predicted and actual prices.
    errors        = predicted_prices - actual_prices     # how wrong, with sign
    squared       = errors ** 2          # all positive, big errors blown up
    loss          = squared.mean()       # one number summarising "how bad on average"
    return loss
    # could also be written as a one liner 
    # return np.mean((predicted_prices - actual_prices) ** 2)


def calculate_gradient(sizes, predicted_prices, actual_prices):
    #Calculate the gradient of the loss with respect to w and b.
    N = len(sizes)
    errors = predicted_prices - actual_prices
    w_gradient = (2/N) * np.dot(errors, sizes)  # dL/dw
    b_gradient = (2/N) * errors.sum()            # dL/db
    return w_gradient, b_gradient

# Secret rule the model has to discover
TRUE_W = 200.0
TRUE_B = 50.0

np.random.seed(0)

# CHANGED: 100 houses now (more data = easier to see the trend)
N = 100
sizes  = np.random.uniform(20, 200, size=N)

# NEW: add random noise, so prices wobble around the line
# np.random.randn(N) gives N samples from a standard normal (mean 0, std 1).
# We multiply by 500 so the noise is "± a few hundred €" worth of wiggle.
noise  = np.random.randn(N) * 500
prices = TRUE_W * sizes + TRUE_B + noise


# Initialize
w = 0.0
b = 0.0
learning_rate = 1e-5

# Storage for the loss at every epoch (we'll plot this in Milestone 6)
losses = []

# Train for 100,000 iterations ("epochs")
for epoch in range(100_000):
    # Forward pass: predict and measure how wrong we are
    y_pred = predict(sizes, w, b)
    loss = mse_loss(y_pred, prices)

    # Backward pass: figure out which way to nudge w and b
    grad_w, grad_b = calculate_gradient(sizes, y_pred, prices)

    # Apply the update rule
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    # Remember the loss so we can plot it later
    losses.append(loss)

    # Print every 1000 epochs so the terminal isn't flooded
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d}  loss={loss:.2f}  w={w:.2f}  b={b:.2f}")

print(f"\nFinal: w = {w:.2f}  (true: 200.0)")
print(f"       b = {b:.2f}  (true: 50.0)")
print(f"Final loss = {loss:.2f}")