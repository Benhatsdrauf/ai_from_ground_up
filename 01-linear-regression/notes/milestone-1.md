# Milestone 1 — Forward pass (`predict`)

## What I built

I build a simple predict function that uses accepts the size of a house and the the weights and bias to predict the cost of the houses

```python
def predict(sizes, w, b):
    return w * sizes + b
```

## New concepts I learned

- **Weights and bias (`w`, `b`)**: I realised that these are the nobs the "AI/Model" will have to tune in the future to self tune to get better at predictions. Im still not 100% certain i understand it fully because this is also a really simple example with one one input ( size of the house ) one weight ( price per m2) and one bias ( offset ? what a house with 0m2 would cost)
- **A linear model (`y = w * x + b`)**: Our simple Model that calculates the price of the house / or better said predicts it => its calculating the weight times the size plus the bias to predict prices. by itself not really helpfull because its just a calculation right now. But later it will become more powerfull wen our Model/Ai will get more intelligent and can tune the weight and bias itself to better predict prices
- **NumPy broadcasting**: is a really cool feature that i didnt know about before. withou NumPy i would have to rewrite the fucntion as a foor loop to calculate each entry in our house list but with NumPy arrays broadcasting all of that is handled on its own and i can concentrate on the logic of the model instead of worrying about implementing the foorloop myself

  ```python
  # `sizes` is a NumPy array of 100 house sizes.
  # `w * sizes` multiplies every element by w in one step — no for-loop needed.
  return w * sizes + b
  ```

- **Synthetic data (and why we use it before real data)**: We do this so we can easily debug the logic. This way we now what the data aka the price should be like and so its easyer to compare the output of the funciton to the synthetic data because if i would user real data right away it would be harder for me to spot an error, especilly now while im still learning how everything works

  ```python
  # I pick the "true" weights, generate fake houses with them, plus a bit of noise.
  # If my model later recovers numbers close to TRUE_W and TRUE_B, I know it works.
  TRUE_W = 200.0
  TRUE_B = 50.0
  prices = TRUE_W * sizes + TRUE_B + noise
  ```

## What clicked / surprised me

in the beginning it was really overwalming and i didnt understand a thing. But after wrapping my head arround it didnt feel to scary anymore. What helped me was when i realised the predictions with w=100 came out exactly half — that was the moment I realized w really is just the slope

```
--- predictions with bad weights (w=100, b=0) ---
size= 118.8   predicted= 11878.6   actual= 23224.7
size= 148.7   predicted= 14873.4   actual= 30247.2
size= 128.5   predicted= 12849.7   actual= 25982.3
```

Every prediction is roughly half of the actual — because `w=100` is half of the true `w=200`.

## What I got stuck on

Honestly in the beginning with everything haha.
I felt really lost and felt like i what am i doing here this is way above my head. I never used NumPy or anything before. I have worked in python before but only for some really simple scripts so this is also like a crash course in python at the same time

i didnt understand the formulas at first and that made me rething and made me overthing so badly i didnt even know what to start with in code at all.
