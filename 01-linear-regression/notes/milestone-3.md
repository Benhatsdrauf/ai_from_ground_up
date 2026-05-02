# Milestone 3 — Gradients (`compute_gradients`)

## What I built

We built the logic for calculating gradients aka the info the model needs to self adjust itself in the future

```python
def compute_gradients(sizes, y_pred, y_true):
    n = len(y_true)
    errors = y_pred - y_true
    grad_w = (2 / n) * (errors * sizes).sum()
    grad_b = (2 / n) * errors.sum()
    return grad_w, grad_b
```

## New concepts I learned

- **What a gradient is (intuition)**: The gradient is a way to calculate in wich direction our nobs ( weight and bias ) should be adjusted. Its kinda like trying to go down a hill if your blind folded. you take a step forward and can feel it going down o you know you are headed the right direction if you can feel it going uphill again, you know you are on the wrong path and have to go the other way. So the gradients sign tells the model if we need to adjust our nobs up or down
- **Numerical vs analytical gradient**: Numerical ones would get us to the same end goal but they are slower. I did try it as a simple example and we had to calculate it once with w = 100 and b = 0 and once with w = 101 and b = 0 to see the difference and to know wich direction to adjust. For this usecase it would be to bad but if we had a real llm with millions of weights this would take forever. Gradients on the other hand show you the same result / info we want but we dont need to calculate it twice to know if our ajdustment got us closer or further away form where we want to be

  ```
  Numerical (bumping w by 1):  -2,763,969
  Analytical (formula):        -2,777,708
  ```

  Same answer to within ~0.5% — but the analytical formula is instant.

- **The gradient formulas** (`grad_w` and `grad_b`):
  - `grad_w = (2/n) * sum(sizes * error)` — error of each house, weighted by its size (because big houses pull harder on w), summed and scaled.
  - `grad_b = (2/n) * sum(error)` — same idea but no size weighting (b affects every house equally).
  - `2/n` is just a scaling factor: the `2` comes from squaring the error in MSE, the `1/n` comes from taking the mean. Same loss → same factor → so it's identical in both formulas.
- **Why `sizes` appears in `grad_w` but not in `grad_b`**: This is becasue w affects differently sized houses differently ... ( sorry english is not my first language :D ) what i mean is w's inpect changes with the houses size so if the house is 10m2 => 10 _ w or if its 100m2 => 100 _ w. b on the other side is just simply added the same way to each house and does not change depending on the size of the house 10m2 + b or 100m2 + b its simply just added to the house.
- **`np.dot(a, b)` vs `(a * b).sum()`**: Basically they to the same but np.dot() is a NumPy built in function that is optimized for our use case and under the hood is implemented in C i think. They return the same result but np.dot() is alot faster (10-100x)

  ```python
  np.dot(errors, sizes)      # optimized C/BLAS under the hood
  (errors * sizes).sum()     # same result, slower at scale
  ```

## What clicked / surprised me

It surprised me that there is a formular to calculate it as "simple" and efficent like this.
My fist thought was to go the numerial way to calculating a baseline then adjust w and b recalulating and seeing how it changed.
I also learned about pythons Tuple unpacking wich reminds me of Object destructuring from Javascript :D

## What I got stuck on

This one went really well. I only had a hickup on the return 2 values form one function in the beginning but that was over quickly :D
