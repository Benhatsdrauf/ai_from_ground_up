import numpy as np

# We secretly decide the "true" rule that connects size to price.
# The model won't know these — its job is to figure them out.
TRUE_W = 200.0   # each m² adds 200 € to the price
TRUE_B = 50.0    # baseline (in thousands, just for nicer numbers)

# Make 5 fake houses with random sizes between 20 and 200 m².
np.random.seed(0)                          # so you and I get the same random numbers
sizes  = np.random.uniform(20, 200, size=5)
prices = TRUE_W * sizes + TRUE_B           # apply the secret rule

# Print them so you can see what we made
for size, price in zip(sizes, prices):
    print(f"size = {size:6.1f} m²   ->   price = {price:7.1f}")