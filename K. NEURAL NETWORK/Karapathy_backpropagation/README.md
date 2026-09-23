# Neural Networks & Large Language Models: From Scratch

An end-to-end, first-principles implementation of deep learning fundamentals and autoregressive Transformer architectures, following Andrej Karpathy's *Neural Networks: Zero to Hero* series.

This repository documents a complete progression through the mechanics of deep learning: starting with scalar autograd engines and manual backpropagation, advancing through feedforward language models and normalization dynamics, developing custom Byte-Pair Encoding (BPE) tokenizers, and culminating in full decoder-only Transformers and a bit-for-bit replication of OpenAI's GPT-2 (124M).

---

## Repository Map

```
Karapathy_backpropagation/
├── makemore series from basic/
│   ├── 01_basic_of_BP.ipynb          # Micrograd scalar autograd engine & MLP from scratch
│   ├── 01_micrograd_exercises.ipynb  # Analytic/numerical derivatives & softmax NLL verification
│   ├── 02_makemore.ipynb             # Bigram language model: count-matrix vs. single-layer neural net
│   ├── 03_makemore_mlp.ipynb         # Bengio et al. (2003) character-level MLP with embedding lookups
│   ├── 04_makemore_mlp2.ipynb        # Activations, Kaiming init, BatchNorm & diagnostic plotting
│   ├── 05_makemore_backprop.ipynb    # Exact manual tensor backpropagation through MLP & BatchNorm
│   ├── 06_wavenet.ipynb              # Hierarchical dilated WaveNet architecture with modular layers
│   └── names.txt                     # Training corpus (32,033 names)
│
├── mini-GPT/
│   ├── bigram.py                     # Minimal baseline bigram model on Tiny Shakespeare
│   ├── GPT.py                        # Full decoder-only Transformer with multi-head self-attention
│   └── Mini-GPT.ipynb                # Step-by-step mathematical derivation of attention mechanisms
│
├── GPT-Tokenizer/
│   ├── tokenizer.ipynb               # BPE mechanics, Unicode UTF-8 handling & SentencePiece exploration
│   ├── encoder.json                  # GPT-2 vocabulary definition
│   ├── vocab.bpe                     # GPT-2 merge table
│   ├── solidgoldmagikarp.md          # Analysis of tokenizer anomalies & vector clustering
│   ├── Exercise/
│   │   └── exercise.md               # Step-by-step tokenizer development roadmap
│   └── sol_of_exercise/
│       └── minbpe-master/            # Complete modular tokenizer package (Basic, Regex, GPT4)
│           ├── minbpe/
│           │   ├── base.py           # Abstract Tokenizer base class & merge serialization
│           │   ├── basic.py          # Basic byte-level BPE tokenizer
│           │   ├── regex.py          # Regex-based chunked BPE with special token handling
│           │   └── gpt4.py           # GPT-4 tokenizer matching tiktoken with byte un-shuffling
│           ├── train.py              # BPE training script
│           └── tests/                # Test suites & Taylor Swift evaluation corpus
│
├── GPT-2(124M)/
│   ├── My GPT-2 (duplicated by me from scratch )/
│   │   ├── train_gpt2.py             # From-scratch PyTorch GPT-2 (124M) & HuggingFace weight loader
│   │   └── play.ipynb                # Weight inspection & manual top-k autoregressive sampling
│   └── GPT-2(released by openai)/    # Reference OpenAI release for architectural validation
│
└── Papers_&_Notes/
    ├── NOTES/                        # Comprehensive study notes & architectural walkthroughs
    │   ├── 01.karpathy_makemore_master_notes.pdf
    │   ├── 02.Karpathy_GPT_Code_Study_Note-1.pdf
    │   ├── 03.0.nanogpt_notes_upto_forward_loss.pdf
    │   ├── 03.1.nanogpt_revision_notes_full.pdf
    │   └── 03.2.nanogpt_train_gpt2_master_revision.pdf
    └── reseaerch papers/             # Foundational literature referenced across the work
        ├── 01.GPT-2(language_models_are_unsupervised_multitask_learners).pdf
        ├── 02.GPT-3 (Language Models are Few-Shot Learners).pdf
        ├── 03.Attention Is All You Need.pdf
        └── 04.GAUSSIAN ERROR LINEAR UNITS (GELUS).pdf
```

---

## Architectural Progression & Implementation Details

### 1. Autograd & Computational Graphs (Micrograd)
* **File:** `makemore series from basic/01_basic_of_BP.ipynb`, `01_micrograd_exercises.ipynb`
* **Core Concepts:** Directed acyclic graphs (DAG), reverse-mode automatic differentiation, gradient accumulation for variable reuse, numerical vs. analytical differentiation.
* **Key Components:**
  - Implemented the `Value` scalar wrapper class tracking data, gradients, children nodes (`_prev`), and operation labels (`_op`).
  - Implemented forward and reverse autodiff methods for arithmetic and non-linear operations: `__add__`, `__mul__`, `__pow__`, `__truediv__`, `__sub__`, `__neg__`, `tanh`, `exp`.
  - Solved gradient overwrite bugs across branching graphs by enforcing multivariate chain-rule gradient accumulation (`+=`).
  - Implemented automatic reverse topological sorting (`build_topo`) to order backward passes through arbitrary graph topologies.
  - Built computation graph visualizer using `graphviz` to inspect node data, operations, and backpropagated gradients.
  - Constructed `Neuron`, `Layer`, and `MLP` abstractions with parameter management and trained on custom classification targets using Mean Squared Error (MSE) and SGD.
  - Extended scalar autodiff to derive Softmax activation and Negative Log-Likelihood (NLL) classification loss, verified against PyTorch tensors.

---

### 2. Statistical Modeling to Neural Language Models
* **File:** `makemore series from basic/02_makemore.ipynb`
* **Core Concepts:** N-gram statistical language modeling, maximum likelihood estimation (MLE), negative log-likelihood loss, one-hot input representations, cross-entropy minimization.
* **Key Components:**
  - Built a statistical character-level bigram frequency model using a $27 \times 27$ count matrix $N$ on 32,033 names, incorporating start/stop tokens (`.`).
  - Derived probability distribution matrices using row-wise normalization ($P_{i,j} = \frac{N_{i,j}}{\sum_k N_{i,k}}$) with model smoothing (Laplace $+1$).
  - Evaluated sequence likelihood via average negative log-likelihood (NLL): $\mathcal{L} = -\frac{1}{N}\sum \log P(x_t \mid x_{t-1})$.
  - Re-implemented bigram modeling as a single-layer neural network:
    - One-hot encoded input characters into $1 \times 27$ vectors.
    - Linear projection to log-counts (logits) via weight matrix $W \in \mathbb{R}^{27 \times 27}$.
    - Softmax normalization: $\text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$.
    - Backpropagation with PyTorch autodiff and parameter updates via SGD.

---

### 3. Bengio Neural Probabilistic Language Model
* **File:** `makemore series from basic/03_makemore_mlp.ipynb`
* **Core Concepts:** Continuous character embeddings, distributed representations (Bengio et al., 2003), context window conditioning, learning rate search schedules.
* **Key Components:**
  - Replaced sparse one-hot encodings with an embedding matrix $C \in \mathbb{R}^{27 \times D}$ ($D \in \{2, 10\}$), projecting characters into continuous representation space.
  - Structured sliding context windows (`block_size = 3`) concatenated via zero-copy tensor reshaping (`.view(-1, block_size * n_embd)`).
  - Two-layer feedforward architecture:
    $$h = \tanh(C(x) W_1 + b_1)$$
    $$\text{logits} = h W_2 + b_2$$
  - Rigorous data splitting into Training (80%), Validation/Dev (10%), and Test (10%) splits.
  - Implemented systematic learning rate tuning via exponential sweeps ($10^{\text{linspace}(-3, 0, 1000)}$) to locate stable step rates.
  - Visualized 2D learned embeddings ($C \in \mathbb{R}^{27 \times 2}$) demonstrating clustering of vowels vs. consonants.

---

### 4. Deep Network Dynamics & Batch Normalization
* **File:** `makemore series from basic/04_makemore_mlp2.ipynb`
* **Core Concepts:** Activation distribution diagnostics, dead neuron prevention, Kaiming (He) weight initialization, Batch Normalization theory and momentum tracking.
* **Key Components:**
  - Diagnosed vanishing gradients and tanh saturation caused by uncalibrated random weight initialization.
  - Applied Kaiming / He normal scaling: $W \sim \mathcal{N}\left(0, \frac{\text{gain}}{\sqrt{\text{fan\_in}}}\right)$ with $\text{gain} = \frac{5}{3}$ for $\tanh$.
  - Scaled final output layer weights down ($W_2 \times 0.01$) to eliminate high initial entropy loss (the "hockey stick" curve).
  - Built custom `BatchNorm1d` from scratch:
    - Standardized pre-activation batches: $\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$.
    - Learned scale ($\gamma$) and shift ($\beta$) parameters: $y = \gamma \hat{x} + \beta$.
    - Maintained exponential moving average running statistics (`running_mean`, `running_var`) for deterministic inference.
  - Structured modular PyTorch-style layers (`Linear`, `BatchNorm1d`, `Tanh`) with explicit parameter registries.
  - Implemented diagnostic telemetry: plotting activation distributions, layer gradient histograms, and $\log_{10}$ update-to-data ratios ($\frac{\eta \cdot \sigma(\nabla W)}{\sigma(W)}$) across deep networks.

---

### 5. Exact Manual Backpropagation
* **File:** `makemore series from basic/05_makemore_backprop.ipynb`
* **Core Concepts:** Matrix calculus, analytical gradients through multi-stage composite graphs, fused operator derivations, numerical bit-level gradient verification.
* **Key Components:**
  - Reconstructed an entire MLP with Batch Normalization and Cross-Entropy loss without relying on PyTorch's `loss.backward()`.
  - Derived and validated exact analytical gradients for every intermediate tensor step-by-step:
    - **Cross-Entropy + Softmax:**
      $$\frac{\partial \mathcal{L}}{\partial z_i} = \frac{1}{N} (p_i - y_i)$$
    - **Output Layer Weights & Bias:**
      $$\frac{\partial \mathcal{L}}{\partial W_2} = h^T \frac{\partial \mathcal{L}}{\partial \text{logits}}, \quad \frac{\partial \mathcal{L}}{\partial b_2} = \sum \frac{\partial \mathcal{L}}{\partial \text{logits}}$$
    - **Hidden Non-Linearity:**
      $$\frac{\partial \mathcal{L}}{\partial h_{\text{preact}}} = (1 - h^2) \odot \frac{\partial \mathcal{L}}{\partial h}$$
    - **Batch Normalization (Atomic & Fused Formulation):**
      Derived step-by-step backpropagation through mean, variance (with Bessel's correction $\frac{1}{N-1}$), normalization, and affine scaling, as well as the condensed fused equation:
      $$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma}{N \sqrt{\sigma^2 + \epsilon}} \left[ N \frac{\partial \mathcal{L}}{\partial y_i} - \sum_j \frac{\partial \mathcal{L}}{\partial y_j} - \frac{N}{N-1} \hat{x}_i \sum_j \left(\frac{\partial \mathcal{L}}{\partial y_j} \hat{x}_j\right) \right]$$
    - **Embedding Lookup Gradients:**
      Accumulated sparse parameter updates $\frac{\partial \mathcal{L}}{\partial C}$ matching input token occurrences.
  - Verified every gradient tensor against PyTorch autograd using a custom numerical comparison utility (`cmp()`), achieving exact bit matches and differences $\le 10^{-8}$.
  - Trained the network entirely inside a `torch.no_grad()` context using purely handcrafted gradient math.

---

### 6. Hierarchical Architecture (WaveNet)
* **File:** `makemore series from basic/06_wavenet.ipynb`
* **Core Concepts:** Dilated convolutions, tree-structured receptive field expansion, hierarchical feature aggregation, multi-dimensional tensor batch normalization.
* **Key Components:**
  - Expanded context length from 3 to 8 characters using progressive binary grouping (WaveNet style).
  - Developed a modular architecture composed of `Embedding`, `FlattenConsecutive(n)`, `Linear`, `BatchNorm1d` (supporting 2D/3D tensor slicing), `Tanh`, and `Sequential` container.
  - Replaced monolithic linear input flattening with progressive hierarchical downsampling:
    $$\mathbb{R}^{B \times 8 \times 24} \xrightarrow{\text{FlattenConsecutive}(2)} \mathbb{R}^{B \times 4 \times 48} \xrightarrow{\text{Linear}} \mathbb{R}^{B \times 4 \times 128} \dots \xrightarrow{} \mathbb{R}^{B \times 27}$$
  - Reduced parameter count while significantly scaling context horizon, achieving lower validation loss ($\sim 1.99$) and producing more coherent character-level generations.

---

### 7. Self-Attention & Decoder-Only Transformer (mini-GPT)
* **File:** `mini-GPT/bigram.py`, `mini-GPT/GPT.py`, `mini-GPT/Mini-GPT.ipynb`
* **Core Concepts:** Scaled dot-product attention, multi-head causal self-attention, pre-layer normalization, residual stream highway, autoregressive generation.
* **Key Components:**
  - Traced the mathematical derivation of causal pooling from primitive nested loops to matrix multiplication with normalized lower-triangular masks (`torch.tril`), masked softmax, and key-query affinity projections:
    $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
  - Implemented `Head` module: Single causal self-attention head with linear $Q, K, V$ projections and causal lower-triangular buffer.
  - Implemented `MultiHeadAttention`: Parallel head evaluation with concatenated output projection ($W_O$) and dropout.
  - Implemented `FeedFoward`: Position-wise two-layer MLP with expansion factor 4 and ReLU non-linearity.
  - Implemented `Block`: Pre-LayerNorm Transformer block featuring residual connections:
    $$x \leftarrow x + \text{MultiHeadAttention}(\text{LayerNorm}_1(x))$$
    $$x \leftarrow x + \text{FeedForward}(\text{LayerNorm}_2(x))$$
  - Implemented `GPTLanguageModel`: Full autoregressive architecture combining learned token embeddings (`wte`), learned positional embeddings (`wpe`), sequential `Block` layers, final `LayerNorm`, and language model classification head.
  - Trained on the Tiny Shakespeare dataset with AdamW ($lr = 3\times 10^{-4}$), context window $T=256$, embedding dimension $C=384$, $6$ heads, and $6$ layers, supporting dynamic autoregressive sampling.

---

### 8. Byte-Pair Encoding (BPE) & Tokenizer Engineering
* **File:** `GPT-Tokenizer/tokenizer.ipynb`, `GPT-Tokenizer/sol_of_exercise/minbpe-master/`
* **Core Concepts:** Unicode standards (code points, UTF-8 variable-length byte representations, UTF-16 surrogates, grapheme clusters), BPE pair-merge statistics, regex-based boundary splitting, special token handling.
* **Key Components:**
  - Implemented BPE training algorithm from scratch:
    - Byte-level tokenization over raw UTF-8 streams ($0\dots255$).
    - Consecutive pair frequency counting (`get_stats`).
    - Iterative replacement of top-frequency pairs with newly minted token IDs (`merge`).
    - Bidirectional vocabulary and merge dictionary generation.
  - Built three production-grade tokenizer variants in `minbpe`:
    - `BasicTokenizer`: Byte-level BPE encoder/decoder without pattern constraints.
    - `RegexTokenizer`: Chunked BPE respecting GPT-2/GPT-4 regex patterns (`GPT4_SPLIT_PATTERN`), preventing merges across disparate categories (letters, numbers, punctuation, multi-spaces), and managing configurable special tokens (`<|endoftext|>`, etc.).
    - `GPT4Tokenizer`: Exact replication of OpenAI's GPT-4 tokenizer (`cl100k_base`):
      - Recovered raw merge hierarchies from tiktoken mergeable ranks.
      - Reconstructed the historical 256-byte shuffle permutation mapping.
      - Bit-for-bit identical encoding and decoding matching official `tiktoken` outputs.
  - Experimented with Google's `SentencePiece` (`spm.SentencePieceTrainer`) using Llama-2-compatible specifications (character coverage, byte fallback, and BPE rules).

---

### 9. Complete GPT-2 (124M) Replication
* **File:** `GPT-2(124M)/My GPT-2 (duplicated by me from scratch )/train_gpt2.py`, `play.ipynb`
* **Core Concepts:** OpenAI GPT-2 architecture reproduction, HuggingFace weight translation, residual scaling initialization, high-throughput dataloading.
* **Key Components:**
  - Implemented the full GPT-2 (124M) architecture matching OpenAI's official release:
    - `GPTConfig` dataclass: `block_size = 1024`, `vocab_size = 50257`, `n_layer = 12`, `n_head = 12`, `n_embd = 768`.
    - `CausalSelfAttention`: Combined $Q, K, V$ linear projection ($768 \to 2304$), multi-head reshaping, causal masking, and output projection.
    - `MLP`: Hidden expansion ($768 \to 3072$) with Gaussian Error Linear Units (GELU, tanh approximation) and projection back to residual stream.
    - `Block`: Transformer block with pre-LayerNorm residual routing.
    - **Weight Tying:** Shared parameters between token embedding table and linear output projection head (`transformer.wte.weight = lm_head.weight`), reducing model footprint by $\sim 38.5\text{M}$ parameters.
    - **Residual Scaling Init:** Standard deviation scaled by $\frac{1}{\sqrt{2 \times n_{\text{layer}}}}$ for projection layers adding into residual streams.
  - **Pretrained Weight Loader (`from_pretrained`):**
    - Parsed HuggingFace `GPT2LMHeadModel` checkpoints (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`).
    - Programmatically transposed Conv1D/Linear weight matrices and aligned parameter keys.
  - **High-Performance Data Pipeline (`DataLoaderLite`):**
    - Multi-process memory-mapped shard streaming for massive pretraining corpora (e.g. `edu_fineweb10B`).
  - **Generation & Inference (`play.ipynb`):**
    - Implemented manual top-$k$ sampling ($k=50$) with temperature scaling and multinomial sampling for open-ended text generation.

---

## Key From-Scratch Implementations Summary

| Component | File / Module | Implementation Highlights |
| :--- | :--- | :--- |
| **Scalar Autograd Engine** | `makemore series from basic/01_basic_of_BP.ipynb` | Custom `Value` class, forward/backward operators, DAG traversal, topological sort. |
| **Manual Analytical Backprop** | `makemore series from basic/05_makemore_backprop.ipynb` | Complete derivation of $\frac{\partial \mathcal{L}}{\partial W}$, $\frac{\partial \mathcal{L}}{\partial b}$, $\frac{\partial \mathcal{L}}{\partial C}$, and fused BatchNorm backward without autograd. |
| **Custom BatchNorm1d** | `makemore series from basic/04_makemore_mlp2.ipynb` | Batch mean/variance computation, scale/shift affine parameters, running EMA buffers. |
| **Hierarchical WaveNet** | `makemore series from basic/06_wavenet.ipynb` | `FlattenConsecutive` multi-dimensional tensor folding with dilated tree aggregation. |
| **Causal Multi-Head Attention** | `mini-GPT/GPT.py` | Query-Key-Value projections, scale factor $\frac{1}{\sqrt{d_k}}$, causal masking, parallel head execution. |
| **BPE Tokenization Suite** | `GPT-Tokenizer/sol_of_exercise/minbpe-master/` | `BasicTokenizer`, `RegexTokenizer`, and `GPT4Tokenizer` matching OpenAI `cl100k_base` exactly. |
| **GPT-2 (124M) Engine** | `GPT-2(124M)/.../train_gpt2.py` | Full PyTorch replication, GELU activation, weight tying, HuggingFace weight importer, `DataLoaderLite`. |

---

## Technical Stack & Dependencies

* **Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch (`torch`, `torch.nn`, `torch.nn.functional`)
* **Tokenization & NLP:** `tiktoken`, `sentencepiece`, `regex`, `transformers`
* **Data Processing & Scientific Computing:** `numpy`, `math`, `dataclasses`
* **Visualization & Diagnostics:** `matplotlib`, `graphviz`, `tensorboard`

---

## How to Run

### 1. Running the Interactive Notebooks
Open any notebook in your Jupyter environment or IDE:
```bash
# Launch Jupyter Lab / Notebook
jupyter lab

# Recommended execution order:
# 1. makemore series from basic/01_basic_of_BP.ipynb
# 2. makemore series from basic/02_makemore.ipynb
# 3. makemore series from basic/03_makemore_mlp.ipynb
# 4. makemore series from basic/04_makemore_mlp2.ipynb
# 5. makemore series from basic/05_makemore_backprop.ipynb
# 6. makemore series from basic/06_wavenet.ipynb
# 7. mini-GPT/Mini-GPT.ipynb
# 8. GPT-Tokenizer/tokenizer.ipynb
# 9. GPT-2(124M)/My GPT-2 (duplicated by me from scratch )/play.ipynb
```

### 2. Training the Character-Level Transformer (mini-GPT)
To train the self-attention GPT model on the Shakespeare corpus:
```bash
cd "mini-GPT"
python GPT.py
```

### 3. Testing the BPE Tokenizer Engine (minbpe)
To run tests verifying BPE encoding, decoding, and regex splitting:
```bash
cd "GPT-Tokenizer/sol_of_exercise/minbpe-master"
python -m pytest tests/test_tokenizer.py
```

---

## Conclusion & Learning Outcomes

This folder represents a thorough, ground-up mastery of the foundational stack supporting modern Large Language Models:
1. **Mathematical Underpinnings:** Derived and verified manual tensor gradients, loss surfaces, and normalization dynamics.
2. **Modular Architecture Design:** Built clean, reusable neural network primitives bridging raw tensor algebra with modern PyTorch abstractions.
3. **Tokenization Systems:** Mastered the tokenization layer, resolving byte encoding quirks, regex boundary enforcement, and subword merge optimization.
4. **Production Transformer Replication:** Constructed a faithful replica of GPT-2 (124M) capable of loading official weights and executing efficient inference.
