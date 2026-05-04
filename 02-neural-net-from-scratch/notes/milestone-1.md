# Milestone 1 — Forward pass

## What I built

I implemented the forward pass of a 2-layer neural network — three functions: relu, softmax, and forward that chains them together. Given a batch of MNIST images, it outputs a probability distribution over the 10 digit classes (which doesn't mean anything yet because weights are still random)

## New concepts I learned

- **ReLU (and why neural nets need it)**: ReLU serves as a "gatekeeper" within a neural network, deciding which information should be passed to the next layer. It applies a simple non-linear transformation: If the input is negative, it converts it to zero. If the input is positive, it allows it to pass through unchanged. Without a non-linearity between layers, stacking two linear layers just gives you another linear layer — the network would collapse back to plain linear regression no matter how deep it got. ReLU's non-linear "kink" at zero is what lets the network fit curves and complex patterns.

  ```python
  def relu(z):
      return np.maximum(0, z)
  ```

- **Softmax (and the numerical stability trick)**: Softmax takes the network's 10 raw output scores ("logits") and turns them into 10 probabilities that sum to exactly 1 — one per digit class. The biggest logit gets the most probability. That's how we get from "raw model output" to "the model thinks it's a 3 with 87% confidence". The numerical stability trick helps us because values over ~709 ( np.exp(709) is the largest finite value for float64 ) would be turned into infinity so if we have multiple values higher than the threshold we would end wit infinity / infinity at some point

  ```python
  def softmax(z):
      # subtract max BEFORE exp so big numbers don't blow up to infinity
      z_shifted = z - z.max(axis=-1, keepdims=True)
      exp_z = np.exp(z_shifted)
      return exp_z / exp_z.sum(axis=-1, keepdims=True)
  ```

- **The 2-layer forward pass architecture**: We get the inputs that go trought linear regression => then we go to the hidden layer where we run one ReLu path => that gets passed into the second layer / run of a linear regression => the we get into the softmax state that returns the classification result

  ```python
  def forward(X, W1, b1, W2, b2):
      Z1 = X @ W1 + b1     # linear layer 1 (input → hidden)
      A1 = relu(Z1)        # nonlinearity
      Z2 = A1 @ W2 + b2    # linear layer 2 (hidden → output)
      Y_hat = softmax(Z2)  # logits → probabilities
      return Z1, A1, Z2, Y_hat
  ```

- **Random weight initialization (and why scale by 0.01)**: We scale them down so the logits stay small. If logits started huge, softmax would saturate immediately — one class would get ~100% probability before training even begins. That makes the gradients near-zero, so training would be stuck. Small init keeps the model neutral so it has room to learn

## What clicked / surprised me

- The huge-logits softmax test giving the same answer as the small-logits test — that was a real "aha" moment showing why the stability trick exists.
- Realizing that a 2-layer neural net is literally just two linear regressions stacked with a ReLU in between.

## What I got stuck on

I understood the formulas before I understood why we need them — especially softmax. It took seeing the stability test pass with [1000, 1001, 1002] for the trick to click.
