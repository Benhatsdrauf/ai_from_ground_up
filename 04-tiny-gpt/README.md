# Project 4: Tiny GPT from Scratch

**Stack:** PyTorch
**Time:** 2–3 weeks
**Goal:** Build a small character-level transformer that generates Shakespeare-like text. Understand exactly how an LLM works.

This is the payoff project of the entire ladder. After this, you will *understand* GPT-4, Llama, Claude — not in a hand-wavy "it's a transformer" way, but in a "I have built one of these myself" way. The difference between your toy model and GPT-4 is scale and data, not fundamentally different mechanisms.

---

## Part 1: Theory

### What is a language model?

A language model predicts the next token given previous tokens.

```
Given: "The cat sat on the"
Predict probabilities over: ["mat", "chair", "floor", "rug", ...]
```

That's it. That's the whole thing. Train a model on enough text to predict the next token well, and you get a system that can write coherent paragraphs, answer questions, write code, debate philosophy. Surprisingly, this simple objective produces all the emergent capabilities of modern LLMs at sufficient scale.

### Tokens

A "token" is a unit of text. Three options:
- **Character-level** — each character is a token. Vocab size ~100. Simple, what you'll start with.
- **Word-level** — each word is a token. Vocab size huge, struggles with rare words.
- **Subword (BPE)** — what real LLMs use. Common words are one token, rare words split into pieces. Vocab size ~50,000.

You'll start char-level and optionally upgrade to BPE later.

### The transformer architecture

A transformer processes a sequence of tokens. The key innovation is **self-attention**, which lets each token "look at" every other token in the sequence and decide which ones are relevant.

#### Self-attention, intuitively

For each token in the sequence, we compute three vectors from it:

- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I represent?"
- **Value (V):** "What information do I carry?"

Each token's query is compared (via dot product) against every token's key. High dot product = high attention weight. Then we use those attention weights to compute a weighted sum of all tokens' values.

The result: each token's new representation is a context-aware mixture of information from across the sequence.

In math:

```
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

Where `d_k` is the dimension of keys (the `sqrt` is for numerical stability — without it, dot products grow too large for softmax to handle).

#### Why "multi-head" attention?

Instead of one attention computation, we run several in parallel ("heads"), each with its own Q, K, V projections. Different heads learn to attend to different things — one might track syntax, another semantic relations, another long-range dependencies. Their outputs are concatenated and projected back.

#### Causal masking

For language modeling, a token at position `i` should only attend to tokens at positions ≤ `i`. (You shouldn't be able to "see the future" when predicting the next token.) We enforce this by setting attention scores for future positions to `-infinity` before the softmax. After softmax, those positions get probability ~0.

#### Positional encoding

Self-attention has no inherent notion of order — without help, "dog bites man" and "man bites dog" look identical. We add a **positional embedding** to each token's vector to encode its position. Original GPT uses learned position embeddings. (Modern variants like RoPE do this differently, but learned is fine for our purposes.)

#### Putting it together: a transformer block

```
Input
  ↓
[LayerNorm → MultiHeadAttention] → residual add
  ↓
[LayerNorm → FeedForward (2-layer MLP)] → residual add
  ↓
Output
```

This block is repeated N times. Each block refines the representation of every token using context from all other (preceding, due to causal mask) tokens.

#### The whole GPT

```
Tokens
  ↓ [token embedding + position embedding]
Vectors
  ↓ [N transformer blocks]
Refined vectors
  ↓ [final LayerNorm + Linear projection to vocab size]
Logits over vocabulary
  ↓ [softmax during inference]
Next token probabilities
```

### Training

For each position in the input, predict the *next* token. The target for position `i` is the token at position `i+1`. Standard cross-entropy loss, just like Project 2 — we just have a much bigger output vocabulary and a richer architecture in front of it.

### Inference (generation)

To generate text:
1. Feed in a starting context.
2. Get probabilities over next token from the model.
3. Sample one token (greedy = argmax, or random sampling with temperature).
4. Append it to the context.
5. Repeat.

This is autoregressive generation. Same loop GPT-4 runs.

### Scale and the bitter lesson

Your tiny GPT will have ~1–10 million parameters. GPT-3 has 175 billion. Llama 3 70B has 70 billion. The architecture is essentially the same. What you gain by scaling is more parameters, more data, more compute — and remarkably, capabilities emerge at scale (zero-shot reasoning, few-shot learning, instruction following) that don't exist at small scale. This is one of the most important phenomena in modern AI, and it's still poorly understood.

### Recommended Watch — Critical

- **Andrej Karpathy — "Let's build GPT: from scratch, in code, spelled out"** (YouTube, ~2 hours).
- **This is essentially the curriculum for Project 4.** Watch it. Do everything he does. Then extend it.
- His associated repo, `nanoGPT`, is the cleanest reference implementation of GPT in existence. Bookmark it.
- Optional follow-up: Karpathy's "Let's build the GPT Tokenizer" video for understanding BPE.

---

## Part 2: Project Setup

### Use Colab with GPU.

You'll be training a real (small) language model. CPU training is impractical.

```bash
# In a Colab cell
!pip install torch tiktoken
```

(`tiktoken` is OpenAI's BPE tokenizer — useful for milestone 6.)

### Dataset

Start with **Tiny Shakespeare** — a 1MB text file of all of Shakespeare's plays. Karpathy's tutorial uses it.

```python
import requests
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
```

Other options once you have the basics:
- Your own writing
- Project Gutenberg books
- Code (a small subset of GitHub)

---

## Part 3: Implementation Milestones

### Milestone 1: Tokenization & data prep

Char-level: build a vocabulary of unique characters, mappings between chars and ints, and `encode`/`decode` functions.

```python
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)
```

Split into train/val (90/10). Convert to tensors. Write a `get_batch(split)` function that samples random chunks of length `block_size` (e.g., 256 chars) and returns `(x, y)` where `y` is `x` shifted by one position.

**Checkpoint:** `decode(encode("hello"))` returns `"hello"`. `get_batch("train")` returns two tensors of shape `(batch_size, block_size)` and they differ by one position.

### Milestone 2: Bigram baseline

Before transformers, build the simplest possible language model: a single embedding table.

```python
class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.embed(idx)  # (B, T, vocab_size)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
```

This model only looks at the *previous* character. Train it. Generate from it. The output will be garbage (vaguely plausible characters but no structure).

**Checkpoint:** Loss should drop from `log(vocab_size) ≈ 4.17` to about `2.5`. Generated text: random but with vaguely Shakespeare-like character frequencies.

### Milestone 3: Single-head self-attention

Add one attention head. Build it manually with linear layers for Q, K, V. Implement causal masking with a triangular matrix:

```python
mask = torch.tril(torch.ones(block_size, block_size))
# In attention: scores = scores.masked_fill(mask == 0, float('-inf'))
```

**Checkpoint:** Loss should drop further (~2.3). Generated text: still mostly garbage but shows some local structure.

### Milestone 4: Multi-head attention + transformer block

Wrap multi-head attention into a class. Add a feed-forward layer. Add residual connections and layer norm. This is one full transformer block.

```python
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_head, n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
```

**Checkpoint:** Loss drops to ~1.8. Generated text starts to look Shakespeare-y in shape — line breaks, character names followed by colons, even occasional readable phrases.

### Milestone 5: Stack blocks → full GPT

Stack 4–6 transformer blocks. Add token + position embeddings at the input. Add final LayerNorm and projection to vocab. Train with AdamW, learning rate ~3e-4, for ~5000 steps.

Suggested hyperparameters:
- `block_size = 256`
- `n_embd = 384`
- `n_head = 6`
- `n_layer = 6`
- `dropout = 0.2`
- `batch_size = 64`
- ~10M parameters total

On a Colab T4 GPU, this trains in ~15–30 minutes.

**Checkpoint:** Validation loss ~1.5. Generated text reads like surreal Shakespeare — characters speak in turn, words are mostly real English, syntax is mostly correct, but content is dreamlike. **You have built a working language model.** Pause and reflect on this.

### Milestone 6 (Bonus): Upgrade to BPE

Replace char-level tokenization with `tiktoken`'s GPT-2 BPE encoder. Retrain on a larger dataset (Project Gutenberg, your own data). This is what real LLMs use.

### Milestone 7 (Bonus): Sampling improvements

Implement temperature, top-k, and top-p (nucleus) sampling. Compare quality of generations:

- **Temperature** — divide logits by `T` before softmax. `T < 1` = more confident/repetitive. `T > 1` = more random.
- **Top-k** — only sample from the top `k` tokens.
- **Top-p** — sample from smallest set whose cumulative probability ≥ `p`.

These three knobs control all modern LLM generation, including the ones you've used.

---

## Part 4: Common Bugs

### "Loss is stuck at log(vocab_size)"

Model is outputting uniform predictions. Check:
- Embedding sizes match
- You're actually using the model output (not bypassing it)
- Loss function expects logits, not probabilities

### "Loss is NaN"

- Learning rate too high. Try `1e-4`.
- Numerical instability in attention — verify the `/sqrt(d_k)` scaling.
- Forgot causal mask, model attending to future tokens.

### "Generated text is repetitive ('the the the the')"

- Temperature too low (greedy decoding gets stuck).
- Increase temperature to ~1.0 or use top-k sampling.

### "Training is super slow"

- Make sure you're on GPU.
- Increase `batch_size` if memory allows.
- Reduce `block_size` if needed.

### "Out of memory"

- Reduce `batch_size`, `block_size`, or `n_embd`.
- If on Colab, runtime might already have stale memory — restart runtime.

### "My attention masking is wrong"

Print the attention matrix for one head and one example. The upper triangle (above the diagonal) should be 0 or near 0 after softmax. The lower triangle and diagonal should have attention weights summing to 1 along each row.

---

## Tasks for Claude Code (When Stuck)

### "Walk me through self-attention with a concrete example"

> I'm implementing self-attention. Walk me through what happens with a 4-token sequence and embedding dim 8. Show me the shape of Q, K, V, the attention scores, the mask, and the output. Use concrete made-up numbers, not abstract notation.

### "Explain layer norm vs batch norm"

> Why does GPT use LayerNorm instead of BatchNorm? What's the difference, and why does it matter for transformers specifically?

### "I don't understand causal masking"

> Explain causal masking in self-attention with a concrete example. Why is it needed? What goes wrong without it during training? What goes wrong without it during inference?

### "Compare residual connections to no-residual"

> What does the residual connection in `x = x + self.attn(self.ln1(x))` actually do? What problem does it solve? What happens to gradient flow without it?

### "Help me debug attention"

> My transformer's loss is plateauing at ~2.3 and generated text doesn't improve beyond random characters. Here's my attention implementation: [paste]. Help me find the bug.

### "Why scale by sqrt(d_k)?"

> Why do we divide attention scores by `sqrt(d_k)`? What goes wrong without it? Walk me through the math.

---

## What You Should Walk Away Knowing

- What is a token, and what's the difference between char-level, word-level, and BPE?
- Explain self-attention to a friend in 2 minutes, no math.
- Why is causal masking necessary for language modeling?
- What's the role of position embeddings, given that attention is order-invariant?
- What does a transformer block look like? (LN → Attn → residual → LN → FFN → residual)
- How does sampling temperature work?
- What is the difference between training and inference for a language model?

You should now be able to read the original "Attention Is All You Need" paper (Vaswani et al., 2017) and largely understand it. Try it.

---

## Stretch Goals — Each Worth Doing

- **Read "Attention Is All You Need"** and "Language Models are Few-Shot Learners" (GPT-3 paper). Both are readable now.
- **Train on your own dataset.** Your blog posts, your code, your favorite author. Watch the model start to mimic.
- **Implement a more efficient attention** — flash attention, multi-query attention. (These are real research topics, you're at the frontier.)
- **Build a chat interface for your model.** A simple Gradio app where you type and the model responds. The output won't be useful (your model is too small) but the loop is real.
- **Compare your model's behavior to a real LLM** with the same prompt. Where does scale help? Where doesn't it?

---

## Reflection

When your model first generates text that looks even vaguely like Shakespeare — not just gibberish, but with character names, colons, dialogue structure, mostly-real words — pause. You did this. You built a thing that learns language structure from raw data. The same pattern, scaled up 10,000x with internet-scale data and a billion-dollar compute budget, is GPT-4.

The architecture you just built is the architecture of every major LLM. The difference is scale and engineering, not fundamental ideas.

That's worth sitting with.

---

**Previous:** [Project 3 — PyTorch + CNN](./03-pytorch-cnn.md)
**Next:** [Project 5 — Local Inference & Fine-tuning](./05-local-inference-finetuning.md)
