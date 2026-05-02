# Milestone 4 — Training loop

## What I built

The training loop => finally the model starts to come to live and starts learning on his own

```python
w = 0.0
b = 0.0
learning_rate = 1e-5
losses = []

for epoch in range(10000):
    y_pred = predict(sizes, w, b)
    loss = mse_loss(y_pred, prices)
    grad_w, grad_b = compute_gradients(sizes, y_pred, prices)

    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b
    losses.append(loss)
```

## New concepts I learned

- **The training loop / "epochs"**: The loop the model uses to train hisself aka starts to tune the nobs on its own. the epoch just means how many training iterations we do => in a real project we probably. would tune this to get the best accuracy in the shortest time so we train long enaught for it to get high accuracy but only as long as needed
- **The update rule (`w = w - lr * grad_w`)**: this is where the magic happens each iteration we tune the nobs by the learning rate, over time the ai / model will tune itself to get as close to the real value as possible without us having to adjust anything manually
- **Learning rate (and why ours had to be tiny)**: It has to be tiny because our raw prices and sizes are large numbers, so the gradients come out huge (~5 million in the first iteration). With a normal learning rate like 0.01 we'd skip way past the right answer.
- **What "convergence" looks like in the printout**:
  - convergence = loss + weights stop changing meaningfully => the model settled in on his answers and further training wouldn't change it by much / would not be worth the compute
  - In my test w converged fast (~1000 epochs), but b didn't converge in 10,000 => it kept slowly creeping upward and only reached 50 around epoch 90,000.
  - The loss plateau at ~253,000 was the noise floor — the model couldn't get below it, so it converged there

  ```
  Epoch     0  loss=555,811,286   w=55.26   b=0.42
  Epoch  1000  loss=    253,279   w=201.07  b=2.18
  Epoch  9000  loss=    252,964   w=201.03  b=7.19
  ```

## What clicked / surprised me

after having all steps brocken down like we did seeing all of them work together in the end in a loop makes it not feel as magical anymore as it did before

Surprised that when I ran 100,000 epochs, b actually went past 50 and stopped at 54. I learned that the model isn't trying to recover the "true" w and b — it's finding the best fit on the noisy data we gave it. With finite samples, those two answers are slightly different.

## What I got stuck on

honestly nothig this was smooth sayling
