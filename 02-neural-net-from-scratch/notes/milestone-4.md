# Milestone 4 — Training loop

## What I built

Impplemented the full training loop for our neural network.
Using mini batches of 64 images for training to get a good mix of speed and fine tuning

```python
for epoch in range(epochs):
    # Shuffle so the model doesn't learn the data order
    perm = np.random.permutation(len(X_train))
    X_shuf, y_shuf = X_train[perm], y_train[perm]

    for i in range(0, len(X_train), batch_size):
        X_batch = X_shuf[i : i + batch_size]
        y_batch = y_shuf[i : i + batch_size]

        # Forward → loss → backward → update
        Z1, A1, Z2, Y_hat = forward(X_batch, W1, b1, W2, b2)
        # ... backward pass (the 7 gradient lines from milestone 3) ...
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
```

## New concepts I learned

- **He initialization (and why `* 0.01` wasn't enough)**: sqrt(2/fan_in) keeps the activations from vanishing through the ReLU layers. With 0.01 scaling, after one or two layers the values are basically all zero and nothing learns.

  ```python
  W1 = np.random.randn(784, 128) * np.sqrt(2 / 784)   # ≈ 0.05
  W2 = np.random.randn(128, 10)  * np.sqrt(2 / 128)   # ≈ 0.125
  ```
- **Mini-batches (and why not full-batch or single-example)**: We use mini-batches for a good middle ground of speed and training capacity.
  if we would go through all images at once so a batch of 60k images we would get a really accurate gradient but we would only get one "nob tune" per epoch so we would have to train the model for longer.
  if we would do 1 image at a time we would get the most self tunign but the runtime for one epoch would take way longer ( just imagine having 1mill images instad of 60k )
- **Why we shuffle each epoch**: We schuffel each epoch so the model doesnt just learn the order of the input
- **Epoch vs mini-batch vs weight update (the count confusion)**:
  Lets think about it like reading a book.
  Data = The Book
  Epoch = Reading the whole Book
  mini-batch = number of pages you read in one sitting
  weight updates = the notes you take after reading
  So in one Epoch ( reading the whole Book ) we read the book in our mini-batches ( 64 pages ) and after that we took our notes ( adjusted weights ) after we read the whole book in batches of 64 pages until we are done => one Epoch done. Now we do that for 9 more times.
- **Tracking accuracy alongside loss**: We track both because loss can technically go up or down without accuracy chanign.
  Loss is what the models uses to see how well it did
  The accuracy is mostly for peoply to have an understanding of how accurate the model is

  ```python
  # argmax picks the class with the highest predicted probability
  predictions = Y_hat.argmax(axis=1)
  accuracy = (predictions == y_batch).mean()   # fraction correct
  ```

## What clicked / surprised me

Its still really surprising to see how much the model learns in its first epoch ( in our case over 900 weight adjustments ) and how little it changes in the later epochs.

## What I got stuck on

Not much this time all the major functionality was already in , i just had to create the training loop and the mini-batches.
