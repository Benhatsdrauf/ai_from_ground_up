# Milestone 2 — Loss function (`cross_entropy`)

## What I built

I implemented cross-entropy loss — the function that turns the model's 10-probability output and the true digit label into a single 'how wrong is this?' number. With random weights, it gives ~2.30 (the uniform-guess baseline).

## New concepts I learned

- **One-hot encoding (and why we need it)**: In Project one we used MSE-Loss wich compared the prediction ( 1 number ) wiht the truth ( 1 number ), for our new functionality we cant compare it like that because our output is an array of values ( vectores ) with a value form 0 to 1 => 1 in place 2 would mean this number is the number 1. One-hot encoding takes the true value lets say 5 and converts it to the spae we need => [0,0,0,0,0,1,0,0,0,0] => wich would mean it has 100% possability that the number is a 5 wich now allows us to compare the models prediction with the real value.

  ```python
  def one_hot(y, num_classes=10):
      out = np.zeros((len(y), num_classes))
      out[np.arange(len(y)), y] = 1
      return out
  ```

- **Cross-entropy (and why not just MSE again)**: Why we cant use MSE again is explained in the top bullet point. Cross-entropy is our way to tell the model how wrong it was. We compare the real value with the return value from our One-hot encoding and check if the prediction put the highest possibility to the right number. we use -log() => Cross-entropy looks at the probability the model gave the correct class. High probability on the correct class → low loss. Low probability on the correct class → high loss.

  ```python
  def cross_entropy(Y_hat, Y_true_onehot):
      Y_hat = np.clip(Y_hat, 1e-7, 1 - 1e-7)
      return -np.sum(Y_true_onehot * np.log(Y_hat)) / len(Y_hat)
  ```

- **The natural log (`log`) function**:
  - What does log do? (Inverse of e^x, "what power of e gives me x?") it returns me what power of e gives me x => it returns the exponent
  - What does log do specifically to numbers between 0 and 1? possabilities live between 0 and 1 so it is the perfect fucntion to return small loss on big possabilities and big losses on small possabilities
  - Why is the minus sign in -log(p) necessary? because natural log of fractions returns negative numbers and we want to flip them to be positive
  - One concrete example: what does -log(0.1) give you, and why is that the value you saw? -log(0.1) returns the "loss" if the possability is just at 10%

  | p (probability of correct class) | -log(p) (loss) |
  | -------------------------------- | -------------- |
  | 1.0 (perfect)                    | 0              |
  | 0.99 (confident-correct)         | 0.01           |
  | 0.50 (unsure)                    | 0.69           |
  | 0.10 (uniform guess)             | 2.30           |
  | 0.01 (confident-wrong)           | 4.60           |
  | 0.001 (very confident-wrong)     | 6.91           |

- **The clip trick (avoiding `log(0)`)**: Just guarantees no probability is exactly 0 or 1. otherwise it could return -inf or nan wich would break our logic.
  - What problem does clip solve? = log(0) = -inf would break training (loss becomes nan). The upper clip near 1 is a symmetry safety net — log(1) = 0 is actually fine, but clipping costs nothing and protects against floating-point edge cases
  - What does np.clip do mechanically? we limit the values / keep them in a save space so values that are 0 would be keept just above 0 and values at exactly 1 will be kept just under 1
  - Why 1e-7 and 1 - 1e-7 specifically? because 1e-7 is slightly above 0 and 1 - 1e-7 is slightly under 1` so we clip away from both unsafe edges.

## What clicked / surprised me

- The 2.30 = -log(0.1) connection — the loss number wasn't arbitrary, it came directly from the math.
- The fact that one-hot just reshapes the truth so it has the same shape as the prediction
- log does opposite things for "small" and "big" probabilities — that asymmetry is what makes it work as a loss.

## What I got stuck on

I implemented the functions but didn't actually understand what we were doing or why until we walked through a concrete example with a single image.
Cross-entropy felt abstract until I saw 'the answer is 5, the model gave 0.10 to position 5, so loss = -log(0.10) ≈ 2.3' written out step by step.
