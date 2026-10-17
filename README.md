# Character-Level Language Model Using PyTorch

A character-level language model built using **PyTorch** and a Transformer-based neural network architecture. The model learns patterns from a text dataset and generates new text one character at a time.

The project implements character-level tokenization, embeddings, causal multi-head self-attention, feed-forward networks, training and evaluation loops, and checkpoint-based training resumption.

## Features

- **Character-level tokenization:** Builds a vocabulary from the input text and maps each unique character to an integer.
- **Transformer architecture:** Uses multi-head self-attention and feed-forward neural networks to learn contextual relationships.
- **Positional embeddings:** Provides information about character positions in a sequence.
- **Causal attention masking:** Prevents the model from attending to future characters.
- **Training and evaluation:** Splits the dataset into training and evaluation sets to monitor model performance.
- **Checkpoint saving:** Saves model weights, optimizer state, training step, and random number generator state.
- **Training resumption:** Loads an existing checkpoint and resumes training.
- **Text generation:** Generates new text by predicting and sampling the next character repeatedly.

## Architecture

The model follows a decoder-only Transformer architecture.

```text
Input Text
    |
Character Vocabulary
    |
Character-to-Integer Encoding
    |
Token Embeddings + Positional Embeddings
    |
Transformer Block × 4
    |
    +-- Layer Normalization
    +-- Multi-Head Causal Self-Attention
    +-- Residual Connection
    +-- Layer Normalization
    +-- Feed-Forward Network
    +-- Residual Connection
    |
Final Layer Normalization
    |
Linear Language Modeling Head
    |
Next-Character Probability Distribution
    |
Sampling
    |
Generated Text
```

Each Transformer block contains two sublayers: causal multi-head self-attention and a feed-forward network. Both use pre-layer normalization and residual connections.

### Model Components

| Component | Configuration |
|---|---|
| Embedding dimension | 256 |
| Attention heads | 4 |
| Transformer blocks | 4 |
| Feed-forward hidden dimension | 1024 |
| Maximum context length | 64 characters |
| Dropout | 0.2 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Training steps | 100,000 |
| Evaluation interval | 1,000 steps |
| Evaluation iterations | 100 |

Cross-entropy loss measures how well the model predicts the next character in a sequence.

## Project Structure

```text
NLP_from_Texts/
├── Test.PY          # Model, training loop and text generation
├── input.txt        # Training corpus (provided by the user)
├── checkpoint.pt    # Generated training checkpoint
└── README.md        # Project documentation
```

## Requirements

- Python 3.10 or newer
- PyTorch
- A CPU-compatible Python environment

The current implementation explicitly uses the CPU, so a GPU is not required.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CrispyKSP/NLP_from_Texts.git
cd NLP_from_Texts
```

### 2. Install dependencies

```bash
pip install torch
```

### 3. Add the dataset

Place your training corpus in a plain-text file named `input.txt` in the project directory.

The model automatically builds its character vocabulary from the contents of this file.

### 4. Run the model

```bash
python Test.PY
```

The program begins training, periodically evaluates the model, saves checkpoints, and generates text after training completes.

## How It Works

### 1. Data Preprocessing

The program reads `input.txt`, extracts all unique characters, sorts them into a vocabulary, and creates two mappings:

- `StoI`: Converts characters into integer token IDs.
- `ItoS`: Converts integer token IDs back into characters.

The encoded dataset is split into 90% training data and 10% evaluation data.

### 2. Batch Generation

Random batches of sequences are sampled from the training or evaluation split. Each input sequence contains 64 characters, while its target sequence contains the same characters shifted one position forward.

This trains the model to predict the next character at every position.

### 3. Transformer Training

Token and positional embeddings are combined and passed through four Transformer blocks. Each block contains causal self-attention, a feed-forward network, residual connections, and layer normalization.

The model is optimized using AdamW with a learning rate of 0.0001 and cross-entropy loss.

### 4. Evaluation

Every 1,000 training steps, the program estimates the average training and evaluation losses using 100 batches for each split.

Lower evaluation loss generally indicates better next-character prediction on the evaluation dataset, although it does not guarantee coherent generated text.

### 5. Checkpointing

The program saves a checkpoint to `checkpoint.pt` containing:

- Model parameters
- Optimizer state
- Training step
- PyTorch CPU random number generator state

If a checkpoint exists when the program starts, the saved states are loaded and training resumes.

**Checkpoint limitation:** Checkpoints are saved periodically, rather than after every training step. If training ends between checkpoint intervals, the latest progress may not be saved. The saved step is also reused as the starting step when resuming, so the corresponding step can be repeated.

### 6. Text Generation

After training, the model generates up to 1,500 new characters by sampling from the predicted probability distribution at each step.

The generated token IDs are decoded back into text and printed to the terminal.

## Configuration

The primary hyperparameters are defined near the beginning of `Test.PY`.

```python
batch_size = 16
block_size = 64
eval_interval = 1000
eval_iters = 100
epoch = 100000

n_embed = 256
n_head = 4
n_layer = 4
dropout = 0.2
```

The variable `epoch` represents the total number of training steps, not the number of complete passes through the dataset.

You can experiment with these values to study their effects on training time, model capacity, loss, and generated text quality.

The random seed is set using `torch.manual_seed(1337)`. This supports reproducibility, although identical results are not guaranteed across all environments and execution conditions.

## Limitations

- The model learns only from the supplied text corpus.
- Character-level tokenization can require more generation steps than word- or subword-level tokenization.
- The context window is limited to 64 characters.
- Training for 100,000 steps on a CPU can take a substantial amount of time.
- Generated text may contain repetition, grammatical errors, or incoherent sequences.
- The current implementation does not include a command-line interface, automated dataset downloading, or a dedicated inference mode.

## Future Improvements

- Add configurable command-line arguments for model hyperparameters.
- Support GPU acceleration and mixed-precision training.
- Implement a dedicated inference mode that loads a checkpoint without retraining.
- Add generation controls such as temperature and top-k sampling.
- Track training curves and compare evaluation loss across experiments.
- Support larger datasets and alternative tokenization strategies.
- Add unit tests for encoding, decoding, attention masking, and checkpoint loading.
- Save a final checkpoint when training completes.

## Technologies Used

- **Python** — Core programming language
- **PyTorch** — Neural network implementation, automatic differentiation, optimization, and tensor operations
- **Transformer architecture** — Contextual sequence modeling and autoregressive text generation

---

**Author:** CrispyKSP

**Status:** Experimental / In Development
