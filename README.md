# Financial Text Sentiment Classification: RNN to Transformer Architectures

A comparative study of sequence models for **3-class sentiment classification** (negative / neutral / positive) on **14,000+ financial news headlines and tweets**. The project benchmarks Simple RNN, LSTM, GRU and a **Transformer encoder built from scratch** under the same preprocessing, embeddings and evaluation protocol, so that the architectures can be compared fairly.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Status](https://img.shields.io/badge/status-completed-brightgreen)

---

## Highlights

- **Transformer encoder reached 81% test accuracy and 0.75 macro F1**, outperforming all recurrent models while training **6-10x faster per epoch**.
- **Embedding ablation:** Word2Vec (300-d) beat FastText (300-d), 0.75 vs 0.68 macro F1.
- **Evaluation beyond accuracy:** macro precision, recall, F1 and AUC, which exposed a **~30% recall gap on the minority sentiment classes** that raw accuracy hides.
- **Hyperparameter tuning** with KerasTuner (RandomSearch).
- **Debugging documented:** a cuDNN masking bug (Bidirectional + `mask_zero`) that broke LSTM/GRU training was diagnosed and fixed, and an over-aggressive L1/L2 regularisation setup was corrected during tuning.

---

## Project Pipeline

```
Raw text -> Cleaning -> Tokenization -> Embeddings -> Models -> Evaluation
           (Stage 1)      (Stage 2)                  (Stage 3)  (Stage 4)
```

| Stage | Folder | What happens |
|-------|--------|--------------|
| 1 | `01_data_acquiring_cleaning/` | Data collection, text cleaning, EDA, stratified train/validation/test splits |
| 2 | `02_tokenization_embedding/` | Tokenization, vocabulary building, Word2Vec and FastText embedding matrices |
| 3 | `03_modelling/` | Simple RNN, LSTM, GRU and Transformer encoder; KerasTuner hyperparameter search |
| 4 | `04_final_evaluation/` | Final test-set evaluation, per-class metrics, model comparison |
| - | `data/` | Datasets and intermediate artifacts |
| - | `Financial_Sentiment_Project_Documentation.docx` | Full technical project report |

---

## Models Compared

| Model | Notes |
|-------|-------|
| Simple RNN | Baseline recurrent model |
| LSTM | Gated recurrent model, tuned with KerasTuner |
| GRU | Lighter gated recurrent model, tuned with KerasTuner |
| **Transformer encoder** | Built manually with `keras_hub` layers so pretrained embedding weights are preserved; **best performer** |

### Results

| Model | Test Accuracy | Macro F1 |
|-------|---------------|----------|
| Simple RNN | `<fill>` | `<fill>` |
| LSTM | `<fill>` | `<fill>` |
| GRU | `<fill>` | `<fill>` |
| **Transformer encoder** | **81%** | **0.75** |

### Embedding Ablation (300-dim)

| Embedding | Macro F1 |
|-----------|----------|
| **Word2Vec** | **0.75** |
| FastText | 0.68 |

---

## Key Findings

1. **Attention beats recurrence here.** The Transformer encoder outperformed the recurrent models on macro F1 and trained 6-10x faster per epoch, since it processes tokens in parallel.
2. **Accuracy is misleading on imbalanced sentiment data.** Macro-averaged metrics revealed a ~30% recall gap on minority classes.
3. **Embedding choice matters.** Word2Vec outperformed FastText on this domain-specific vocabulary.
4. **Framework details can silently break training.** Combining `Bidirectional` wrappers with `mask_zero=True` triggered a cuDNN masking issue in LSTM/GRU, documented in the project report.

---

## Tech Stack

- **Language:** Python
- **Deep learning:** TensorFlow / Keras, `keras_hub`
- **Tuning:** KerasTuner (RandomSearch)
- **Embeddings:** Word2Vec, FastText (gensim)
- **Data and evaluation:** pandas, NumPy, scikit-learn, Matplotlib / Seaborn
- **Environment:** VS Code, Jupyter, Anaconda

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Dev-2004-DA/financial-sentiment-classification.git
cd financial-sentiment-classification
```

### 2. Create an environment

```bash
conda create -n fin-sentiment python=3.10 -y
conda activate fin-sentiment
pip install tensorflow keras keras-hub keras-tuner gensim scikit-learn pandas numpy matplotlib seaborn jupyter
```

### 3. Run the notebooks in order

Open each stage folder and run its notebooks sequentially:

```
01_data_acquiring_cleaning  ->  02_tokenization_embedding  ->  03_modelling  ->  04_final_evaluation
```

> A GPU is recommended for the modelling stage.

---

## Repository Structure

```
financial-sentiment-classification/
├── 01_data_acquiring_cleaning/
├── 02_tokenization_embedding/
├── 03_modelling/
├── 04_final_evaluation/
├── data/
├── Financial_Sentiment_Project_Documentation.docx
├── .gitattributes
└── .gitignore
```

---

## Limitations and Future Work

- Minority sentiment classes still show lower recall; class weighting, focal loss or data augmentation could help.
- The Transformer is trained from scratch on a modest dataset; fine-tuning a pretrained financial language model (e.g. FinBERT) is a natural next step for comparison.
- Results are from a single data split; k-fold or repeated runs would give tighter confidence intervals.

---

## Documentation

A full write-up covering methodology, architecture rationale, debugging narratives and limitations is available in
[`Financial_Sentiment_Project_Documentation.docx`](Financial_Sentiment_Project_Documentation.docx).

---

## Author

**Dev**: M.Sc. Statistics, University of Delhi

- GitHub: [@Dev-2004-DA](https://github.com/Dev-2004-DA)
- LinkedIn: `<add your LinkedIn URL>`

---

## License

`<Choose a license, e.g. MIT, and add a LICENSE file>`
