# Annotated Transformer (single-script version)

A single-file, runnable-as-a-script version of
[The Annotated Transformer](https://github.com/harvardnlp/annotated-transformer)
by Harvard NLP, which implements the model from
[Attention Is All You Need](https://arxiv.org/abs/1706.03762) and trains it on
German → English translation (Multi30k).

All credit for the original implementation and explanations goes to the
original authors. This repository is a learning copy with the changes listed
below.

## Changes from the original

- **Plain Python script**: runs with `python transformer.py`, no Jupyter needed.
- **matplotlib instead of Altair** for all figures (masks, positional encoding,
  learning-rate schedule, label smoothing, attention maps), so plots open in a
  normal window.
- **Local Multi30k loading**: reads the dataset from `./data` instead of
  downloading it through `torchtext.datasets`.
- **Fixes for newer PyTorch**:
  - `LabelSmoothing` no longer returns NaN when the input contains `log(0)`
    (`0 * -inf`).
  - `vocab.pt` is loaded with `torch.load(..., weights_only=False)`, needed
    since PyTorch 2.6 changed the default.
- **AdamW** instead of Adam, with weight decay 0.01 (no decay on biases and
  LayerNorm parameters).
- **GELU** instead of ReLU in the position-wise feed-forward layers.

## Setup

Python **3.8 – 3.12** is required (torchtext has no release for Python 3.13+).

```bash
# 1. (GPU) install the CUDA build of torch 2.3.0 first
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 2. install the rest
pip install -r requirements.txt

# 3. spaCy tokenizers (the script also tries to download them automatically)
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Data

Download the Multi30k Task 1 raw files (e.g. from
[multi30k/dataset](https://github.com/multi30k/dataset/tree/master/data/task1/raw)),
unpack them and arrange them as:

```
data/
└── datasets/
    └── Multi30k/
        ├── train.de   train.en   (29000 lines each)
        ├── valid.de   valid.en   (1014 lines each)
        └── test.de    test.en    (1000 lines each)
```

The German and English file of each split must have the same number of lines;
the script raises an error otherwise.

## Usage

```bash
python transformer.py
```

The script runs the examples in order, showing each figure in a window (close
it to continue), then builds the vocabulary (`vocab.pt`), trains the model and
finally visualizes the attention of the trained model.

- Set `RUN_EXAMPLES = False` at the top of the script to skip the examples.
- Checkpoints are saved as `multi30k_model_XX.pt` after every epoch and
  `multi30k_model_final.pt` at the end. If `multi30k_model_final.pt` already
  exists, training is skipped.
- With the default config (batch size 32, 8 epochs), training takes roughly
  3 hours on an RTX 3070 Laptop GPU.

Delete `vocab.pt` and the `multi30k_model_*.pt` files after changing the
model, the data or the Python environment, so they are rebuilt.

## License

MIT, same as the original project. See [LICENSE](LICENSE).
