# Character-Level Language Model Using PyTorch

A character-level language model built from scratch using **PyTorch** and a Transformer-based neural network architecture. The model learns patterns from a text dataset and generates new text one character at a time.

The project implements tokenization, character embeddings, causal multi-head self-attention, feed-forward networks, training and evaluation loops, and checkpoint-based training resumption.

## Features

- **Character-level tokenization:** Builds a vocabulary from the input text and maps each unique character to an integer.
- **Transformer architecture:** Uses multi-head self-attention and feed-forward neural networks to learn contextual relationships.
- **Positional embeddings:** Gives the model information about character positions in a sequence.
- **Causal attention masking:** Prevents the model from attending to future characters during training.
- **Training and validation:** Splits the dataset into training and evaluation sets to monitor model performance.
- **Checkpoint saving:** Saves model weights, optimizer state, training step, and random number generator state.
- **Training resumption:** Automatically loads an existing checkpoint and resumes training.
- **Text generation:** Generates new text by repeatedly predicting and sampling the next character.

## Architecture

The model follows a decoder-only Transformer design.

```text
Input Text
    |
Character Vocabulary
    |
Character-to-Integer Encoding
    |
Token Embeddings + Positional Embeddings
    |
Transformer Blocks × 4
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

The model uses residual connections and layer normalization within each Transformer block. Cross-entropy loss measures the difference between predicted next-character distributions and the actual next characters.

## Project Structure

```text
project/
├── main.py          # Model, training loop and text generation
├── input.txt        # Training corpus
├── checkpoint.pt    # Saved training checkpoint (generated)
└── README.md        # Project documentation
```

## Requirements

- Python 3.10 or newer
- PyTorch
- A CPU-compatible Python environment

The current implementation explicitly uses the CPU, so a GPU is not required.

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace the URL with your repository's actual URL.

**2. Install PyTorch**

```bash
pip install torch
```

**3. Add the dataset**

Place your training corpus in a file named `input.txt` in the project directory.

The dataset should be a plain-text file. The model constructs its vocabulary from the unique characters present in this file.

**4. Run the model**

```bash
python main.py
```

## How It Works

### 1. Data preprocessing

The program reads `input.txt`, extracts the unique characters, sorts them into a vocabulary, and creates two mappings:

- `StoI`: Converts characters into integer token IDs.
- `ItoS`: Converts integer token IDs back into characters.

The encoded dataset is split into 90% training data and 10% evaluation data.

### 2. Batch generation

Random batches of sequences are sampled from the training or evaluation split. Each input sequence contains 64 characters, while its target sequence contains the same characters shifted one position to the left.

This trains the model to predict the next character at every position.

### 3. Transformer training

The model processes token and positional embeddings through four Transformer blocks. Each block contains causal self-attention, a feed-forward network, residual connections, and layer normalization.

The model is optimized using AdamW and cross-entropy loss.

### 4. Evaluation

Every 1,000 training steps, the program estimates the average training and evaluation losses using 100 batches for each split.

Lower loss generally indicates better next-character prediction on the evaluated data, although evaluation loss alone does not guarantee coherent generated text.

### 5. Checkpointing

The program saves a checkpoint to `checkpoint.pt`, containing:

- Model parameters
- Optimizer state
- Training step
- PyTorch CPU random number generator state

If a checkpoint exists when the program starts, it loads the saved states and resumes training.

**Note:** Checkpoints can become large. They are generated artifacts and generally should not be committed to Git unless intentionally distributing a trained model.

### 6. Text generation

After training, the model generates up to 1,500 new characters by sampling from the predicted probability distribution at each step.

The generated token IDs are decoded back into text and printed to the terminal.

## Configuration

The primary hyperparameters are defined near the beginning of `main.py`.

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

You can experiment with these values to study their effects on training time, model capacity, loss, and generated text quality.

The `torch.manual_seed(1337)` setting is kept fixed for reproducibility experiments. Identical results are not guaranteed across different environments or execution conditions.

## Limitations

- The model learns only from the supplied text corpus.
- Character-level tokenization can require more generation steps than word- or subword-level tokenization.
- The context window is limited to 64 characters.
- Training for 100,000 steps on a CPU can take a substantial amount of time.
- Generated text may contain repetition, grammatical errors, or incoherent sequences.
- The current implementation does not include a command-line interface, automated dataset download, or a separate inference mode.

## Future Improvements

- Add configurable command-line arguments for model hyperparameters.
- Support GPU acceleration and mixed-precision training.
- Implement a dedicated inference mode that loads a checkpoint without retraining.
- Add generation controls such as temperature and top-k sampling.
- Track training curves and compare validation loss across experiments.
- Support larger datasets and alternative tokenization strategies.
- Add unit tests for encoding, decoding, attention masking, and checkpoint loading.

## Technologies Used

- **Python** — Core programming language
- **PyTorch** — Neural network implementation, automatic differentiation, optimization, and tensor operations
- **Transformer architecture** — Contextual sequence modeling and autoregressive text generation


---

**Author:** CrispyKSP

**Status:** Experimental / In Development
