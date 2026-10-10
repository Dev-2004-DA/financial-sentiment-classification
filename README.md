# Financial Text Sentiment Classification
### A Comparative Study of Recurrent and Attention-Based Deep Learning Architectures

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Task](https://img.shields.io/badge/task-3--class%20sentiment-informational)
![Status](https://img.shields.io/badge/status-completed-brightgreen)

This project classifies financial news headlines and tweets as **negative, neutral or positive**, and uses that task to compare four architectures under identical preprocessing, embeddings and evaluation: **Simple RNN, LSTM, GRU and a Transformer encoder-only model**. The goal is not just the best score, but an evidence-based understanding of *why* gating and self-attention help, including their training cost.

---

## Highlights

- **The Transformer encoder (Word2Vec) won on every metric**: 81% test accuracy, **0.75 macro F1**, **0.897 macro AUC**.
- It trained **~6-10x faster per epoch** than LSTM/GRU (~13-14 s vs ~90-200 s) and completed hyperparameter search in ~7 minutes vs ~45-50 minutes for the recurrent models.
- The macro-F1 gain comes mainly from the **minority negative class** (F1 0.69 vs 0.55-0.56 for LSTM/GRU), so the improvement is not just from the dominant neutral class.
- **Class-imbalance-aware evaluation:** macro precision, recall, F1 and AUC on a held-out test set, because accuracy alone hides weak minority-class performance.
- **Embedding ablation:** Word2Vec beat FastText on the Transformer (macro F1 0.75 vs 0.68).
- Engineering issues are documented rather than hidden: a cuDNN masking bug, an over-regularised GRU run (a negative result), and transient Transformer training instability.

---

## Dataset

Two financial text sources were combined: a **financial news headlines** dataset and a **financial tweets** dataset. Labels were mapped consistently (0 = negative, 1 = neutral, 2 = positive), concatenated and shuffled with a fixed seed.

| Item | Value |
|------|-------|
| Combined size | 14,195 rows (14,191 after dropping 4 null-text rows) |
| Classes | negative / neutral / positive (neutral is dominant) |
| Split | 60% train / 20% validation / 20% test (fixed seed, done **before** tokenization to prevent leakage) |
| Test set | 2,838 samples (441 negative, 1,815 neutral, 582 positive) |

> Dataset sources: `<add links to the headlines and tweets datasets>`

**Cleaning and preprocessing:** dropped a stray index column, fixed mojibake (corrupted currency symbols) by stripping non-ASCII fragments, lowercased text, removed URLs and @mentions, replaced tickers with `<TICKER>` and hashtags with `<HASHTAG>`, converted emojis with `emoji.demojize()`, and normalised quotes, ellipses, brackets and whitespace.

---

## Tokenization and Embeddings

- **Tokenizer:** Keras `Tokenizer` with an `<OOV>` token, **fit only on the training split** and reused on validation and test.
- **Sequence length:** `MAX_LEN = 40` (99th percentile of token lengths), with post-padding and post-truncation.
- **Pretrained embeddings (300-d):** Word2Vec (Google News) as the primary embedding for all four models, and FastText (Wiki + News) as a secondary comparison on the best architecture. Out-of-vocabulary tokens use random-normal initialisation.

---

## Models

| Model | Architecture | Notes |
|-------|--------------|-------|
| **Simple RNN** | 3 stacked Bidirectional SimpleRNN layers (256, 128, 64) | Performance floor; overfits from epoch 2-3 regardless of dropout or size changes |
| **LSTM** | 3 stacked Bidirectional LSTM layers with LayerNormalization; final units reduced and regularised | Needs `use_cudnn=False` (see below), hence slow training |
| **GRU** | 2 GRU layers (64, 32), no dropout | Best validation loss among recurrent models |
| **Transformer encoder** | Word2Vec token embedding + `PositionEmbedding`, 2 `TransformerEncoder` blocks (6 heads, intermediate dim 2048, dropout 0.2), `GlobalAveragePooling1D`, `Dense(3, softmax)` | 7.76M parameters; built with `keras_hub`; the embedding layer is constructed manually to **preserve the pretrained Word2Vec weights** used by the other models |

Hyperparameters were tuned with **KerasTuner RandomSearch**.

---

## Results

### Cross-model comparison (held-out test set)

| Model | Val Acc | Test Acc | Macro F1 | Macro AUC | Time / Epoch |
|-------|---------|----------|----------|-----------|--------------|
| Simple RNN | ~74-78% | 75% | 0.62 | 0.851 | ~4 s |
| LSTM | 79.22% | 78% | 0.69 | 0.877 | ~110-200 s |
| GRU | 79.64% | 79% | 0.69 | 0.882 | ~90-140 s |
| **Transformer (Word2Vec)** | **81.33%** | **81%** | **0.75** | **0.897** | **~13-14 s** |
| Transformer (FastText) | 80.4% | 79% | 0.68 | 0.896 | ~13-14 s |

### Per-class F1 (test set)

| Model | Negative | Neutral | Positive |
|-------|----------|---------|----------|
| Simple RNN | 0.41 | 0.86 | 0.58 |
| LSTM | 0.55 | 0.88 | 0.64 |
| GRU | 0.56 | 0.89 | 0.64 |
| **Transformer (Word2Vec)** | **0.69** | 0.87 | **0.69** |

The RNN baseline detects only **30% of negative examples** (recall 0.30). The Transformer (Word2Vec) also makes far fewer negative-to-positive confusions (27) than the recurrent models (LSTM 100, GRU 77, RNN 153).

### Word2Vec vs FastText (Transformer)

| Metric | Word2Vec | FastText |
|--------|----------|----------|
| Test accuracy | 81% | 79% |
| Macro precision | 0.77 | 0.79 |
| Macro recall | 0.73 | 0.63 |
| Macro F1 | **0.75** | 0.68 |

FastText produced a more conservative classifier that defaults to *neutral* (97% neutral recall), which raises precision but hurts minority-class recall. Its AUC (0.896) is nearly identical to Word2Vec's (0.897), so the difference lies in the decision behaviour rather than the underlying ranking quality. Word2Vec was kept as the primary embedding.

### Qualitative check

| Input | Predicted | Confidence | Expected |
|-------|-----------|------------|----------|
| Company beats earnings expectations | positive | 0.55 | positive |
| Stock plummets after scandal | negative | 0.82 | negative |
| Markets remain flat today | negative | 0.55 | neutral (miss) |

The single miss was low-confidence, consistent with neutral being the hardest class boundary: language describing an absence of change can read as mildly negative.

---

## Key Findings

1. **Self-attention matched or beat gated recurrence at a fraction of the cost.** Best macro F1, best macro AUC and best validation loss, with ~6-10x faster epochs.
2. **Accuracy hides minority-class weakness.** Macro metrics showed that the gap between models is concentrated in the negative class.
3. **Ungated recurrence has a hard ceiling.** Three regularisation attempts on the RNN all plateaued at ~74-78% validation accuracy, confirming the limit is architectural, not a tuning artifact.
4. **Short tuning budgets can mislead.** For both LSTM and GRU, the "best" search configuration was not the best after full training, so manual refinement was needed.
5. **Embedding choice changes classifier behaviour**, not just accuracy: FastText made the Transformer far more neutral-biased.

---

## Engineering Notes and Debugging

- **cuDNN masking bug:** `Bidirectional` layers combined with `mask_zero=True` produce a reversed mask pattern that cuDNN's fused kernel rejects. Fix: `use_cudnn=False` on all LSTM/GRU layers, which explains their much longer epoch times.
- **Mojibake in the source text:** garbled currency symbols were diagnosed and removed during cleaning.
- **Over-regularised GRU (negative result):** an L1 strength of 0.2 produced an initial loss of ~19 and underperformed the simpler untuned model.
- **Transformer instability:** validation accuracy dropped to 21.6% at epoch 2 before recovering (FastText variant: near-random for the first two epochs). The best checkpoint (epoch 6) was restored.

---

## Limitations

- Class weighting was applied only to the LSTM, not to all four models (though macro metrics were applied uniformly in the final evaluation).
- Hyperparameter searches used few trials and short per-trial epoch budgets.
- No learning-rate warmup was used for the Transformer, which is standard practice and the likely fix for its early instability.
- One LSTM run was interrupted by a connectivity issue and not repeated; the RNN baseline was retrained for final test evaluation, so its test numbers come from a separate (architecturally identical) run.
- The FastText ablation was run on the Transformer only; a fully symmetric ablation would repeat it for every architecture.

---

## Project Structure

| Folder / File | Purpose |
|---------------|---------|
| `01_data_acquiring_cleaning/` | Combining the sources, cleaning, label mapping, stratified splits |
| `02_tokenization_embedding/` | Tokenizer, `MAX_LEN` analysis, Word2Vec and FastText embedding matrices |
| `03_modelling/` | RNN, LSTM, GRU and Transformer training, KerasTuner searches |
| `04_final_evaluation/` | Test-set classification reports, confusion matrices, macro AUC, embedding comparison |
| `data/` | Datasets and intermediate artifacts |
| `Financial_Sentiment_Project_Documentation.docx` | Full technical report (methodology, tuning runs, training curves, limitations) |

---

## Getting Started

```bash
git clone https://github.com/Dev-2004-DA/financial-sentiment-classification.git
cd financial-sentiment-classification

conda create -n fin-sentiment python=3.10 -y
conda activate fin-sentiment
pip install tensorflow keras keras-hub keras-tuner gensim emoji scikit-learn pandas numpy matplotlib seaborn joblib jupyter
```

Run the notebooks stage by stage:

```
01_data_acquiring_cleaning -> 02_tokenization_embedding -> 03_modelling -> 04_final_evaluation
```

> The pretrained Word2Vec (Google News) and FastText (Wiki + News) vectors are large and must be downloaded separately. A GPU is recommended for the modelling stage.

---

## Tech Stack

Python, TensorFlow / Keras, `keras_hub`, KerasTuner, gensim (Word2Vec / FastText), scikit-learn, pandas, NumPy, Matplotlib, Seaborn, Jupyter, VS Code.

---

## Author

**Dev**, M.Sc. Statistics, University of Delhi

- GitHub: [@Dev-2004-DA](https://github.com/Dev-2004-DA)
- LinkedIn: `<add your LinkedIn URL>`

## License

`<Choose a license, e.g. MIT, and add a LICENSE file>`
