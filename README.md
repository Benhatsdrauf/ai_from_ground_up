# AI / LLM From the Ground Up — A Project Ladder

A learn-by-building path from "I know a little Python" to "I understand how LLMs actually work." Each project is self-contained and earns you real, foundational understanding rather than just gluing APIs together.

## The Ladder

| # | Project | Stack | Core Concept | Status |
|---|---------|-------|--------------|--------|
| 1 | [Linear Regression from Scratch](./01-linear-regression/) | Python + NumPy | Gradient descent, loss functions | **Complete** — test MSE 0.56 on California Housing |
| 2 | [Neural Net from Scratch](./02-neural-net-from-scratch/) | Python + NumPy | Backpropagation, MLPs | **Complete** — 97.7% test accuracy on MNIST |
| 3 | [PyTorch + CNN](./03-pytorch-cnn/) | PyTorch + torchvision | Modern tooling, convolutions | In progress |
| 4 | [Tiny GPT from Scratch](./04-tiny-gpt/) | PyTorch | Transformers, self-attention | Not started |
| 5 | [Local Inference & Fine-tuning](./05-local-inference-finetuning/) | transformers + peft | Quantization, LoRA | Not started |

## How to Use These Files With Claude Code

Each project file is structured so Claude Code can effectively help you. Each one contains:

- **Theory section** — read this first, no code yet
- **Implementation milestones** — broken into small, testable chunks
- **`Tasks for Claude Code`** blocks — copy-paste prompts for when you get stuck
- **Checkpoints** — "your code should now do X" gates so you know if you're on track
- **Common bugs** — the mistakes everyone makes, and how to spot them

### Recommended Workflow

1. **Read the theory section yourself first.** Don't let Claude Code explain the math — you need to wrestle with it. The understanding comes from the struggle.
2. **Try implementing each milestone yourself.** Set a timer (30–60 min). If you're stuck after that, then ask for help.
3. **When asking Claude Code for help, share what you tried.** Paste your broken code with a clear question, not "implement this for me." The goal is learning, not shipping.
4. **Always run the checkpoint tests** at the end of each milestone before moving on. Skipping these is how you end up confused 3 projects later.

### Anti-pattern to Avoid

> "Claude Code, build me Project 4."

This will produce working code and zero learning. The whole point is that *you* build it — Claude Code is a tutor, not a contractor. Use it like Stack Overflow with infinite patience: when you're stuck on a specific bug or concept, not as a substitute for thinking.

## Prerequisites

- Comfortable with basic Python (functions, classes, loops, list/dict)
- High school math: matrix multiplication, basic calculus (derivatives — what they *mean*, not heavy techniques)
- A Python environment (3.10+). Anaconda or `uv` both fine.
- Optional but useful: Google Colab account for free GPU access (needed for Projects 3 and 4)

If your calculus is rusty, that's okay. You'll pick up what you need from 3Blue1Brown's videos as you go. You do not need to be a math major.

## Companion Resources (Don't Read All Upfront)

Pull these in as referenced from individual projects:

- **3Blue1Brown — "Neural Networks" series** (YouTube): Best visual intuition for Projects 1–2
- **Andrej Karpathy — "Neural Networks: Zero to Hero"** (YouTube): Pairs perfectly with Projects 2 and 4. The single best resource on this list.
- **PyTorch official tutorials**: For Project 3
- **"Attention Is All You Need"** (Vaswani et al., 2017): The transformer paper. Read this *after* Project 4 — it'll be way more readable.
- **Hugging Face course** (free, online): For Project 5

## A Note on Discipline

The ladder is designed so each project unlocks the next. Project 4 is the payoff — building a working transformer — but it's only meaningful if Projects 1 and 2 are solid. People who skip ahead end up "knowing about" transformers without actually understanding them.

Do them in order. Take your time. Build the foundations.

Good luck.
