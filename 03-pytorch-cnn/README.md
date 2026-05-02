# Project 3: PyTorch + Convolutional Neural Networks

**Stack:** PyTorch + torchvision + matplotlib
**Time:** ~1 week
**Goal:** Learn modern deep learning tooling now that you know what it abstracts. Build a CNN that classifies CIFAR-10 images.

After Project 2, you've earned the right to use PyTorch. You'll appreciate `loss.backward()` because you've done it the hard way. This project introduces three big ideas: (1) PyTorch's autograd, (2) convolutions, and (3) GPU acceleration.

---

## Part 1: Theory

### Why PyTorch?

PyTorch is a library that gives you:

1. **Tensor operations** — like NumPy arrays, but they can live on a GPU.
2. **Autograd** — automatic computation of gradients. No more deriving backprop by hand.
3. **`nn.Module`** — a clean way to organize models with learnable parameters.
4. **DataLoaders** — efficient batching, shuffling, and loading of data.

PyTorch is what virtually all modern AI research is built on, including the LLMs you'll work with later. Other frameworks exist (JAX, TensorFlow) but PyTorch is where you should be.

### The autograd magic

In Project 2, you computed gradients by hand. PyTorch does this for you by **building a computation graph** as you do operations:

```python
import torch
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()      # PyTorch traces back through the graph
print(x.grad)     # tensor([7.])  — that's dy/dx evaluated at x=2
```

Internally, PyTorch is doing exactly what you did by hand: chain rule, layer by layer, in reverse. But now it's automatic. Magical, but you know how the magic works.

### Convolutions: why CNNs for images

For MNIST in Project 2, you flattened a 28×28 image into a 784-dim vector and fed it to a fully-connected layer. This works for tiny images, but throws away spatial structure. For real images, this is wasteful.

**A convolution** is a small filter (e.g., 3×3) that slides over the image, computing dot products at each position. Each filter learns to detect a particular pattern (edge, texture, color blob). A CNN stacks many filters per layer, then stacks layers to detect increasingly complex patterns:

- Layer 1 filters: edges, color blobs
- Layer 2 filters: textures, simple shapes
- Layer 3 filters: parts of objects (wheels, eyes)
- Final layers: whole objects (cars, cats)

This hierarchy mirrors how our visual cortex appears to work, and it's *vastly* more parameter-efficient than fully-connected networks for images.

### Key components of a CNN

- **`Conv2d`** — applies learned filters. Key params: `in_channels`, `out_channels` (= number of filters), `kernel_size` (usually 3).
- **Pooling** (`MaxPool2d`) — downsamples by taking the max in each small region. Reduces resolution, increases the receptive field of later filters.
- **Activation** (ReLU, same as before) — applied after each conv.
- **Flatten + Linear layers at the end** — to produce class scores.

A simple CNN looks like:

```
Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Linear → ReLU → Linear → output
```

### CIFAR-10

60,000 32×32 color images in 10 classes (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck). Harder than MNIST (~95% is good for a basic CNN; humans do ~94%, ResNet does ~95%+).

### GPU acceleration

CNNs are slow on CPU. Use Google Colab's free GPU:

1. Go to https://colab.research.google.com
2. New notebook
3. Runtime → Change runtime type → GPU (T4)

In code:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyCNN().to(device)
inputs = inputs.to(device)
labels = labels.to(device)
```

Anything that's a tensor or model can be moved with `.to(device)`. Forgetting to move things is a common bug — you'll get an error about tensors on different devices.

### Recommended Watch / Read

- **PyTorch's "Learn the Basics" official tutorial** — 1 hour, gives you the lay of the land
- **Stanford CS231n notes — Convolutional Neural Networks** (Google: "cs231n cnn"): excellent, deeper than you need but very clear
- **3Blue1Brown — "But what is a convolution?"** (YouTube): visual intuition for what convolution actually does

---

## Part 2: Project Setup

### Local install (CPU only — slow but works)

```bash
mkdir pytorch-cnn
cd pytorch-cnn
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib
```

### Google Colab (recommended for GPU)

Just open a new notebook. PyTorch is pre-installed.

### Loading CIFAR-10

```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # to [-1, 1]
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
                                            shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64,
                                           shuffle=False, num_workers=2)
```

---

## Part 3: Implementation Milestones

### Milestone 1: Rebuild Project 2 in PyTorch

Before adding convolutions, rebuild your MNIST classifier from Project 2 using PyTorch's `nn.Module`. This is your "Hello World" for PyTorch.

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

Note: PyTorch's `CrossEntropyLoss` expects raw logits (no softmax in the model). It applies softmax internally.

**Checkpoint:** Get >97% on MNIST. Same as before, but in 50 lines of PyTorch instead of 200 lines of NumPy.

### Milestone 2: First CNN on MNIST

Replace the MLP with a small CNN:

```python
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

**Checkpoint:** Get >99% on MNIST. CNNs crush MLPs at images.

### Milestone 3: Move to CIFAR-10

Adapt the model: input has 3 channels (RGB) instead of 1, image size is 32×32. Reshape input dimensions and `Linear` layer sizes accordingly.

**Checkpoint:** Get to 70%+ test accuracy on CIFAR-10. This is much harder than MNIST.

### Milestone 4: Improve

Get to ~80% test accuracy by adding any of these (try one at a time, observe the effect):

- More conv layers (3 conv blocks instead of 2)
- **Batch normalization** between conv and ReLU
- **Data augmentation** — `RandomHorizontalFlip`, `RandomCrop` in your training transforms
- **Dropout** before the final linear layer

**Checkpoint:** Note which change helped and by how much. Building intuition for what works.

### Milestone 5: Use a pretrained model

Load **ResNet-18** pretrained on ImageNet, and fine-tune it for CIFAR-10:

```python
from torchvision import models
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)   # replace final layer for 10 classes
```

**Checkpoint:** Get to 90%+ on CIFAR-10 with very little training. This is your first taste of **transfer learning** — using a model trained on one task as a starting point for another. Crucial concept for Project 5.

### Milestone 6 (Bonus): Inspect what the model learned

Visualize the first conv layer's filters as 3×3 RGB images. They often look like color edges, oriented gradients, etc. — feature detectors learned from data.

---

## Part 4: Common Bugs

### "RuntimeError: expected scalar type Float but found Double"

PyTorch is strict about dtypes. Make sure your tensors are `float32`, not `float64` or `int`. `tensor.float()` converts.

### "RuntimeError: tensors on different devices"

You moved the model to GPU but not the data, or vice versa. Move both with `.to(device)`.

### "My loss is NaN"

- Learning rate too high. Try `0.001` with Adam.
- For MSE specifically, check for NaN in your data.
- For cross-entropy: check that targets are class *indices* (not one-hot).

### "Training accuracy is fine, test accuracy is bad"

Classic overfitting. Add dropout, augmentation, or use a pretrained model.

### "Training is very slow"

Make sure you're on GPU (`print(next(model.parameters()).device)` should say `cuda`). On Colab, check Runtime type. On a local machine without a CUDA GPU, it'll always be slow — switch to Colab.

### "DataLoader is the bottleneck"

Set `num_workers=2` or `4` in DataLoader. With `num_workers=0`, data loading is single-threaded and starves the GPU.

---

## Tasks for Claude Code (When Stuck)

### "Help me read this PyTorch error"

> I'm getting this error in PyTorch: [paste error]. My code is: [paste code]. Walk me through what's happening — don't just give me the fix, explain the underlying concept.

### "Explain why my CNN architecture has these dimensions"

> Walk me through the shape of the tensor at every step in this CNN's forward pass: [paste]. Input is 32x32x3 with batch size 64. Compute the output shape after each layer step by step.

### "What is batch normalization actually doing?"

> Explain batch normalization intuitively. What is it normalizing, when, and why does it help training? When should I use it vs not?

### "Compare optimizers"

> I'm using SGD with learning rate 0.01 but my training is unstable. Compare SGD, SGD with momentum, and Adam in the context of CIFAR-10 training. Which should I try first and why?

### "Walk through a transfer learning script"

> I want to fine-tune ResNet-18 on CIFAR-10. Walk me through the conceptual steps, then point me to which layers I should freeze vs train. Don't write the code — I'll write it.

---

## What You Should Walk Away Knowing

- What does `loss.backward()` actually do? (Hint: walks the autograd graph, computes gradients via chain rule)
- What's a convolution? Why is it good for images?
- What's the difference between `.to(device)`, `.cuda()`, and `.cpu()`?
- What is transfer learning, and why is it so powerful?
- Why does `model.train()` vs `model.eval()` matter? (Dropout, batch norm behave differently)

---

## Stretch Goals

- **Implement a custom Dataset class** — load your own images instead of using a built-in dataset.
- **Implement learning rate scheduling** — `CosineAnnealingLR` or `ReduceLROnPlateau`.
- **Train on a real-world dataset** — Tiny ImageNet, or scrape your own ~1000 image classifier (cats vs dogs from Kaggle is the classic).
- **Read a paper** — pick "ResNet" (He et al., 2015). Read it. Implement skip connections in your CNN. This is your transition to "I can read papers" territory.

---

**Previous:** [Project 2 — Neural Net from Scratch](./02-neural-net-from-scratch.md)
**Next:** [Project 4 — Tiny GPT from Scratch](./04-tiny-gpt.md)
