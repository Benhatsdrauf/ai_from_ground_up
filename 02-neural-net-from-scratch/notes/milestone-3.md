# Milestone 3 — Backpropagation

## What I built

I implemented a working backdrop for a 2-layer neural network written in pure NumPy. Derived all gradients by hand and verefied them.

## New concepts I learned

- **Backpropagation (the big idea)**: Its the chain rule applied layer by layer, walking backwards throught the network.
  The forward pass produces a prediction and a loss. The backwards pass checks "how much did each weight contribute to that loss by multiplying the local gradiants toghether while we move from the outbuck back towards the input.
  We reuse the values form the forward pass so we dont have to recompute them again on the backdrop,

- **The miracle gradient: `dZ2 = Y_hat - Y_onehot`**: When you pair softmax with cross-entropy loss, the gradient at the output layer collapses to "predicted probabilities minus the true one-hot label." All the messy softmax derivatives cancel out with the log in cross-entropy. So if the network predicts 0.7 for the right class and the truth is 1.0, the gradient there is -0.3 — a clean, intuitive "you were 0.3 too low." This isn't a coincidence; softmax + cross-entropy are designed to pair like this.

- **Backprop through a linear layer (`dA1 = dZ2 @ W2.T`)**: For a layer `Z2 = A1 @ W2 + b2`, the gradient flowing back to `A1` is the gradient at `Z2` multiplied by `W2` transposed. Intuitively: each hidden neuron's "blame" is the weighted sum of the blame of every output it feeds into, weighted by _how much_ it fed into them (which is exactly `W2`). The transpose just lines up the shapes so the matmul works.

- **Backprop through ReLU (`dZ1 = dA1 * (Z1 > 0)`)**: ReLU's derivative is 1 where the input was positive and 0 where it was negative — it's literally a gate. So backprop through ReLU just zeros out the gradient for any neuron that was "off" during the forward pass. In my run, a big chunk of `dZ1` elements were zero, which makes sense: those neurons didn't fire, so they can't be blamed for the loss.

- **Gradient checking (and what dW1 = 0 taught me)**: Finite differences (`(loss(w+ε) - loss(w-ε)) / 2ε`) give a numerical estimate of any gradient, which you can compare against your analytical one. If they match to ~1e-7, your math is right. My first sanity check on `dW1` showed near-zero gradients for random pixels — which initially looked like a bug, but it's actually correct: most MNIST pixels (corners, edges) are black across the entire batch, so those input weights genuinely have zero gradient. Re-checking with center pixels gave non-zero values that matched analytically. Lesson: a "zero gradient" isn't always wrong — sometimes the input is just zero.

## What clicked / surprised me

How elegant the math is once you write it out. I expected backprop to be a pile of nasty calculus, but in code it's five lines of matrix multiplications and one elementwise mask. The shapes basically tell you what to do — if you know `dW2` has to match `W2`'s shape `(128, 10)`, and you have `A1` `(64, 128)` and `dZ2` `(64, 10)` lying around, there's really only one way to combine them: `A1.T @ dZ2`. Dimensional analysis is most of the work.

The other surprise was how much the gradient check builds confidence. Before running it I wasn't sure if my derivations were right; afterwards, seeing analytical and numerical agree to 8 decimal places, I knew the whole pipeline was correct.

## What I got stuck on

The dead-pixel zero-gradient thing genuinely threw me — I spent a while convinced I'd messed up the chain rule before realizing the input itself was zero. Also took me a minute to internalize _why_ `dZ2 = Y_hat - Y_onehot` works. I knew the formula, but I had to actually write out the softmax + cross-entropy derivative on paper to see the cancellation happen. Once I did, it stopped feeling like magic.
