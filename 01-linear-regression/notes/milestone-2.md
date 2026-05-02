# Milestone 2 — Loss function (`mse_loss`)

## What I built

I build the Loss function that calculates how "wrong" or "off" our predictions where

```python
def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()
```

## New concepts I learned

- **Loss / measuring "how wrong"**: We take the predicted prices subtrackt the real prices to get a value of how "off" or "wrong" the preditions where
- **Mean Squared Error (MSE)**: As the name already tells we take the mean squared error. its only three simple steps:
  1. we calculate the error so we do predicition - real
  2. we square the value ( why is explained in the next bullet point )
  3. we take the mean from all entrys and return a single number as result

  I think it is usefull for the model once it can adjust the weight and the bias hisself to take this and compare if the change it just made was bringin us closer or farther away from our goal of predictions beeing as close to real as possible.
- **Why we square the errors**: This is the logic, we square the results before we take the mean because if the model would have errors that are off by the same amount but one is positive and the other one was negative they would cancel out and show that the model has 0 loss wich would mean it works great but in reallity its completely off
- **NumPy element-wise operations** (subtraction, squaring, `.mean()`): Again NumPy makes our life easier. we can simpliy subtract one array form another without having to do the looping ourselfs, numpy does this for us same with the squaring of the values. we also utalize its mean function wich we could also do on our own by just adding all of the values and then divide by the amount of entries but NumPy go us covered

## What clicked / surprised me

This felt alot easyer to understand then the last milestone, my head slowly adapts to the right way of thinking. This was really straight forward in terms of implementation and also understanding how we calculate the prices

## What I got stuck on

Not much to be onest i just didnt know about NumPys ability to help me so much so in th beginning i wanted to build the squaring funciton and the subtraction loop myself
