# Project 5: Local Inference & Fine-tuning

**Stack:** `transformers`, `peft`, `bitsandbytes`, optionally `llama.cpp` Python bindings
**Time:** 1–2 weeks
**Goal:** Run real open-source LLMs locally, understand quantization, and fine-tune one with LoRA on your own data.

This project closes the loop between everything you've learned and the production reality of working with LLMs today. You'll go from "I built a tiny GPT" to "I can run, customize, and deploy real open models."

This project is especially relevant if you're shipping apps with on-device LLM inference — you'll deeply understand what's happening under the hood when you load a quantized model.

---

## Part 1: Theory

### The open-model ecosystem

Modern LLMs come in several flavors:

- **Closed APIs** — GPT-4, Claude, Gemini. You send tokens in, get tokens out. No weights for you.
- **Open weights** — Llama 3, Qwen, Mistral, Gemma. Full weight files released. You can run them yourself, fine-tune, modify.
- **Truly open** — OLMo, Pythia. Open weights *and* open training data and code.

For local work, you want open-weights models. For learning, smaller is better — start with 0.5B–3B parameter models. They run on consumer hardware and train fast.

Recommended starter models (as of recent times — check Hugging Face for the current best):
- **Qwen 2.5 0.5B / 1.5B** — small, capable, multilingual
- **Llama 3.2 1B / 3B** — Meta's small models
- **Phi-3 mini** — Microsoft's small model
- **TinyLlama 1.1B** — designed for on-device

### The Hugging Face stack

Hugging Face has become the GitHub of ML models. Three key libraries:

- **`transformers`** — load and run any model. Standardized interface across architectures.
- **`datasets`** — load any dataset. Memory-efficient, streaming-capable.
- **`peft`** — Parameter-Efficient Fine-Tuning. Includes LoRA.

Plus the **Hugging Face Hub** — `pip install huggingface_hub`, log in, push and pull models.

### Quantization: how to fit a 7B model on your laptop

A model's weights are stored as numbers. By default, they're 32-bit floats (`fp32`) — 4 bytes each. A 7-billion parameter model in fp32 is 28 GB. Won't fit on most consumer hardware.

**Quantization** reduces precision:

- **fp16 / bf16** — 16-bit floats. 2 bytes per param. Half the size, almost no quality loss. Standard for inference.
- **int8** — 8-bit integers. 1 byte per param. Some quality loss, much faster on CPU.
- **int4 (Q4)** — 4-bit integers. 0.5 bytes per param. Noticeable but acceptable quality loss for many tasks. **What llama.cpp's GGUF Q4 uses.**

A 7B model at Q4 is ~3.5 GB. Runs on a laptop. Runs on your phone (with `llama.cpp` / `llama.rn`).

#### How does Q4 actually work?

Conceptually: for a group of weights (say, 32 weights), find the min and max. Map them to 16 evenly-spaced "buckets" (4 bits = 16 values). Store the bucket index per weight, plus a single scale factor and zero point per group.

There's information loss — many original values map to the same bucket — but for inference, surprisingly, it works well. Modern formats (GGUF Q4_K_M, GPTQ, AWQ) use clever group sizes and other tricks to preserve more accuracy.

This is one of the most practically important ideas in deploying LLMs. Worth understanding.

### Fine-tuning: full vs LoRA

**Full fine-tuning** updates every weight in the model. For a 7B model, you need to store 7B × 4 bytes (gradients) + 7B × 8 bytes (Adam optimizer states) = ~84 GB just for training overhead. Impractical without serious GPUs.

**LoRA (Low-Rank Adaptation)** freezes the original weights and adds tiny trainable "adapter" matrices alongside specific layers. Math:

```
W_finetuned = W_original + B @ A     (where A and B are small)
```

If `W` is `(4096, 4096)`, `A` might be `(4096, 8)` and `B` `(8, 4096)`. So instead of training 16M parameters, you train ~64K. **Roughly 0.1% of original parameters.**

Tradeoffs:
- LoRA works astonishingly well — often nearly matching full fine-tuning quality.
- Trains in minutes/hours instead of days.
- Adapters are tiny (~10MB), so you can have many specialized adapters per base model.
- Limitation: can't dramatically change the model's underlying knowledge — better for style, format, narrow tasks.

LoRA is how most fine-tuning happens in practice now. Combined with Q4 quantization (called **QLoRA**), you can fine-tune a 7B model on a single consumer GPU.

### Inference engines

For deployment, raw `transformers` is often slow. Options:

- **`llama.cpp`** — C++ inference, super fast on CPU, GPU support, best-in-class for Mac. GGUF format. (You're already using `llama.rn`, which is built on this.)
- **`vLLM`** — fast batched server inference. For backend deployment.
- **`ollama`** — a wrapper around llama.cpp with a simpler interface. Good for local experimentation.
- **`MLX`** — Apple's ML framework. Best on Apple Silicon.

For this project, focus on `transformers` (to learn the concepts) and `llama.cpp` (for production-style inference).

### Recommended Watch / Read

- **Hugging Face NLP course** (free, online, ~10 hours): comprehensive overview of the stack
- **"Tim Dettmers — QLoRA" paper** (2023): the foundational LoRA + quantization paper, very readable
- **Sebastian Raschka's blog** (magazine.sebastianraschka.com): excellent hands-on LLM articles, especially on LoRA

---

## Part 2: Project Setup

### Local install

```bash
mkdir local-inference
cd local-inference
python -m venv .venv
source .venv/bin/activate

pip install transformers datasets accelerate peft bitsandbytes torch
pip install jupyterlab    # nice for experimentation
```

### Hugging Face login

```bash
huggingface-cli login   # paste a token from https://huggingface.co/settings/tokens
```

Some models (Llama family) require accepting a license on the Hub before downloading.

### Hardware notes

- For inference: any CPU works for ≤3B models with quantization. GPU helps but not required.
- For LoRA training: NVIDIA GPU with ≥8GB VRAM, or use Colab. CPU LoRA training is extremely slow.
- Mac with Apple Silicon: use MLX for best performance.

---

## Part 3: Implementation Milestones

### Milestone 1: Run a model with `transformers`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"   # or any small model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Explain backpropagation in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Checkpoint:** You see a coherent response. You've just run a real LLM locally with ~10 lines of code.

### Milestone 2: Inspect the model

Look inside. The architecture should be familiar after Project 4.

```python
print(model)
# Should print something like: layers, attention modules, MLPs — same components you built.
```

Examine:
- Number of parameters: `sum(p.numel() for p in model.parameters())`
- Architecture: how many layers? embedding dim? attention heads?
- Tokenizer: tokenize a sentence and inspect the tokens. How does BPE split unfamiliar words?

**Checkpoint:** You can articulate the size and shape of this model.

### Milestone 3: Quantization

Load the same model in 4-bit:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

Compare:
- Memory footprint (use `torch.cuda.memory_allocated()` if on GPU)
- Inference speed (time a generation)
- Output quality (run the same prompt — is it noticeably worse?)

**Checkpoint:** You see ~4x memory reduction. You can articulate the quality/size tradeoff.

### Milestone 4: Use llama.cpp / GGUF

Download a GGUF version of the same model (or a similar one) from Hugging Face. Try `bartowski/Qwen2.5-0.5B-Instruct-GGUF` or similar. Run it with `llama-cpp-python`:

```bash
pip install llama-cpp-python
```

```python
from llama_cpp import Llama
llm = Llama(model_path="path/to/model.gguf", n_ctx=2048)
output = llm("Q: What is the capital of France? A:", max_tokens=20)
print(output)
```

**Checkpoint:** Compare with the `transformers` version — usually llama.cpp is significantly faster on CPU.

### Milestone 5: Build a small dataset for fine-tuning

Pick a narrow, focused task. Examples:

- Convert customer support emails to structured JSON
- Translate plain English questions into SQL for a specific schema
- Generate plant care advice in a specific format and tone (relevant to your interests!)
- Mimic a specific writing style (your blog, your favorite author with permission)

Aim for **100–1000 examples**. More if you have it, but LoRA does well with surprisingly little data. Format as instruction–response pairs:

```python
{
    "instruction": "Explain how to water a snake plant.",
    "response": "Snake plants prefer infrequent watering — wait until the soil is fully dry, then water thoroughly. Overwatering is the #1 killer of snake plants."
}
```

Save as JSON or use Hugging Face `datasets` to push to the Hub.

**Checkpoint:** You have a clean dataset of 100+ examples in a consistent format.

### Milestone 6: Fine-tune with LoRA

Use the `peft` library and `trl`'s `SFTTrainer`:

```python
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

lora_config = LoraConfig(
    r=8,                          # rank — how big the adapter
    lora_alpha=16,                # scaling factor
    target_modules=["q_proj", "v_proj"],   # which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Should show ~0.1-1% of total params are trainable

# Train with SFTTrainer (read the docs — interface evolves)
```

**Checkpoint:** Training loss decreases over a few hundred steps. Generations on held-out prompts now reflect your fine-tuning data's style/format.

### Milestone 7: Evaluate

Critical — fine-tuning often *seems* to work but the model has degraded on general tasks. Test:

1. **In-domain examples** — does the model now produce the desired format/style?
2. **Out-of-domain examples** — has the model retained general capability?
3. **Reasoning** — pick a few standard tasks (basic arithmetic, common-sense questions). Compare base vs fine-tuned. Significant regression is a red flag.

**Checkpoint:** You have a clear-eyed view of what fine-tuning did and didn't accomplish. Often, the answer is "it changed format but not knowledge" — which is exactly LoRA's strength.

### Milestone 8 (Bonus): Convert and ship

Convert your fine-tuned model to GGUF for use with llama.cpp / llama.rn / Ollama. There are scripts in the llama.cpp repo for this. Now you've gone from raw open weights → fine-tuned for your task → quantized → deployable on-device. This is the full production loop.

This is exactly the workflow your InnerVoice / GreenThumb apps use. You now understand it end-to-end.

---

## Part 4: Common Bugs

### "CUDA out of memory"

- Use 4-bit quantization
- Reduce `batch_size` to 1 (use gradient accumulation if you need bigger effective batch)
- Reduce sequence length
- Use a smaller model

### "Loss is NaN during fine-tuning"

- Switch from `fp16` to `bf16` if your GPU supports it (Ampere or newer)
- Reduce learning rate (try `1e-4` or `2e-5`)
- Check that your data doesn't have weird unicode / empty rows

### "Fine-tuning seems to work but model now answers everything in the fine-tuning style"

This is **catastrophic forgetting / over-fitting on style**. Solutions:

- Lower LoRA rank (`r=4` instead of `8`)
- Mix fine-tuning data with general instruction data
- Train for fewer steps
- Use `lora_dropout`

### "Model generates nonsense after fine-tuning"

- Check that you're applying the LoRA adapter at inference time (not just the base model)
- Check tokenizer matches between training and inference
- Check that your dataset format matches the model's expected chat template (use `tokenizer.apply_chat_template`)

### "Model is super slow on CPU"

Use llama.cpp (GGUF format) instead of `transformers`. Often 5–10x faster.

---

## Tasks for Claude Code (When Stuck)

### "Help me design a fine-tuning dataset"

> I want to fine-tune a small model to [task]. Help me design a dataset structure. Suggest: how many examples I'll likely need, what format the instruction/response should follow, what variations I should include for robustness, common pitfalls in dataset design.

### "Explain QLoRA in detail"

> Walk me through QLoRA. What's the combination of techniques (4-bit quantization + LoRA + paged optimizers)? What does each piece contribute? Why does training in 4-bit work even though we generally need fp16/fp32 for gradients?

### "Debug my fine-tuning"

> My fine-tuned model gives worse responses on out-of-domain tasks than the base model. Here's my LoRA config: [paste]. Here's a sample of my training data: [paste]. Help me diagnose what's wrong and what to change.

### "Compare quantization formats"

> Compare GGUF (Q4_K_M, Q5_K_M, Q8), GPTQ, AWQ, and bitsandbytes 4-bit. When should I use each? What are the quality/speed tradeoffs? Which is best for: server deployment, mobile (llama.rn), Mac inference, fine-tuning?

### "Help me write a chat template"

> My fine-tuned model is producing tokens like `<|im_start|>` mid-response. Explain chat templates, why they exist, and how to use `tokenizer.apply_chat_template` correctly during both training and inference.

### "When does LoRA fail?"

> When is LoRA not enough and full fine-tuning is needed? Give me concrete examples of tasks where LoRA's low-rank assumption breaks down.

---

## What You Should Walk Away Knowing

- How to run an open-source LLM locally with `transformers`
- What quantization is, and the tradeoffs between different formats
- How LoRA works and why it's so much cheaper than full fine-tuning
- The end-to-end workflow: download → fine-tune → quantize → deploy
- How to evaluate a fine-tuned model honestly
- When fine-tuning is the right tool vs prompt engineering vs RAG

---

## Stretch Goals

- **Set up RAG** (Retrieval-Augmented Generation) with `llama-index` or `langchain`. Connect your model to a knowledge base.
- **Build a small evaluation harness** with multiple prompts and rubrics. Run it on every model variant.
- **Try DPO (Direct Preference Optimization)** instead of plain SFT. This is how production models do alignment now.
- **Deploy to your phone** — convert to GGUF, integrate via llama.rn, run inference in a React Native app. (You already have the stack for this.)
- **Read the Llama 3 paper** — Meta's deep dive into how a real frontier model is built. Now readable.

---

## Where to Go From Here

Beyond this ladder, the next steps depend on your interests:

- **Research / papers** — pick a topic (efficient inference, alignment, multi-modality, agents). Read papers in that area for a few weeks. The bar to "actually understanding what's published" is much lower now.
- **Production engineering** — vLLM, batching, KV-cache, speculative decoding, model serving at scale.
- **Multi-modal** — vision-language models (LLaVA, you've already worked with this in GreenThumb), audio models (Whisper).
- **Agents** — function calling, tool use, planning, multi-step workflows.
- **Training from scratch** — pretraining your own (small) model. Expensive but illuminating. Karpathy's `nanoGPT` repo shows how.

The foundations you've built across these 5 projects will make any of these accessible.

---

**Previous:** [Project 4 — Tiny GPT from Scratch](./04-tiny-gpt.md)

You've reached the end of the ladder. Congrats. Now go build something.
