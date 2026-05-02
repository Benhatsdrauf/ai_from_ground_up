# Milestone 5 — Real housing data + normalization

## What I built

I usd real housing data and added normalization

## New concepts I learned

- **Real data (California Housing dataset)**: Used real data from the sklearn datasets of Californain Housing to test more detailed data with more then one feature
- **Why normalization matters**: The test data now has 8 features instead of one and all of them have different sizes and we would run in the same problem as we had before with our w and b in our own test data. normalizing the data helps so we can use a bigger learning rate and that all features can keep up with being weighted not like before where b would creep really slowly behind w when w changed very quickly
- **The formulas for normalization (subtract mean, divide by std)**: i dont have to say much here im not trying to learn the formulas but how to use them if i need deeper understanind i can always look them up and also NumPy abstracts alot of that responsability away from me so i can focus on the code and how to implement it rather then how do i calculate it on my own

  ```python
  # subtract mean, divide by std — per feature (axis=0 = "collapse rows, keep columns")
  X_mean = X.mean(axis=0)
  X_std  = X.std(axis=0)
  X = (X - X_mean) / X_std
  ```

- **Generalizing predict / compute_gradients to multiple features (`X @ w`, `X.T @ errors`)**:
  - What stays the same? (mse_loss didn't change.) First calculating the loss stayed the same.
  - What does X @ w do that w \* sizes couldn't? x @ w allows multiplying 2D arrays meanwile w \_ size only worked on 1D arrays so thats the adjustment needed or else the code would not work.
  - Why X.T (transpose) in the gradient formula? Because we need 8 gradients (one per feature). X.T @ errors has shape (8, 20640) @ (20640,) = (8,). Without the transpose, the shapes wouldn't match.
  - @ is the matrix-version of np.dot.

  ```python
  def predict(X, w, b):
      return X @ w + b

  def compute_gradients(X, y_pred, y_true):
      n = len(y_true)
      errors = y_pred - y_true
      grad_w = (2 / n) * X.T @ errors
      grad_b = (2 / n) * errors.sum()
      return grad_w, grad_b
  ```

- **Train/test split (and why)**: so we an use a part of the data set for training and then we can test the model on a part of the data set it has never seen before to see how well it trained and how close it got with its predictions to the real values

  ```python
  # Shuffle the data, then take 80% for training, 20% for test
  np.random.seed(42)
  indices = np.random.permutation(len(X))
  split_at = int(0.8 * len(X))

  X_train, X_test = X[indices[:split_at]], X[indices[split_at:]]
  y_train, y_test = y[indices[:split_at]], y[indices[split_at:]]
  ```

- **What the learned weights tell you about the data**: the wights tell us which of the 8 features have the most impact on house prices

  ```
  MedInc       +0.8503    ← biggest — income matters most
  HouseAge     +0.1463
  AveRooms     -0.2579
  AveBedrms    +0.2580
  Population   +0.0018    ← essentially zero
  AveOccup     -0.0458
  Latitude     -0.6761    ← location matters (coast vs inland, north vs south)
  Longitude    -0.6455
  bias         +2.0671
  ```

## What clicked / surprised me

It was really cool to see that canging form our initial tes data with one weight an d a bias how relatively easy it was to adjust for 8 features and only having to do minor code changes

## What I got stuck on

the matrix calculations aka using the "@" sign and nowing what it does
also the axis = 0 was a little confising in the beginning
