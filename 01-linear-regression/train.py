import numpy as np
import matplotlib.pyplot as plt   # NEW: for plotting

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

# Show the first 5 so you can see the noise
for size, price in zip(sizes[:5], prices[:5]):
    print(f"size = {size:6.1f} m²   ->   price = {price:7.1f}")

# NEW: plot the houses as a scatter plot
plt.scatter(sizes, prices, alpha=0.6)
plt.xlabel("Size (m²)")
plt.ylabel("Price")
plt.title("Fake houses — noisy data around the true line")
plt.show()

# Try the function with deliberately WRONG weights first.
# The model hasn't "learned" anything yet, so predictions should be bad.
guess_w = 100.0
guess_b = 0.0
predicted_prices = predict(sizes, guess_w, guess_b)

print("\n--- predictions with bad weights (w=100, b=0) ---")
for size, pred, actual in zip(sizes[:5], predicted_prices[:5], prices[:5]):
    print(f"size={size:6.1f}   predicted={pred:8.1f}   actual={actual:8.1f}")

# Now try with the TRUE weights — predictions should be very close to actual.
predicted_prices = predict(sizes, TRUE_W, TRUE_B)

print("\n--- predictions with true weights (w=200, b=50) ---")
for size, pred, actual in zip(sizes[:5], predicted_prices[:5], prices[:5]):
    print(f"size={size:6.1f}   predicted={pred:8.1f}   actual={actual:8.1f}")

# Test 1: predicting the actuals exactly should give loss = 0
print("loss when predictions = actuals:", mse_loss(prices, prices))

# Test 2: with bad weights (w=100, b=0), loss should be a big number
bad_predictions = predict(sizes, guess_w, guess_b)
print("loss with bad weights:", mse_loss(bad_predictions, prices))