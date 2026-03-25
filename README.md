# NLU Assignment 2

This repository contains solutions for two tasks:
1. Word2Vec training and analysis on IIT Jodhpur text data.
2. Character-level name generation using RNN variants.

## Project Overview

### Problem 1: Word2Vec on IIT Jodhpur Data
- Crawl IIT Jodhpur pages and PDFs.
- Clean and normalize text into a corpus.
- Train CBOW and Skip-gram with negative sampling.
- Evaluate nearest neighbors, analogies, and visualize word distributions.

### Problem 2: Character-Level Name Generation
- Train Vanilla RNN, BiLSTM, and Attention+RNN models.
- Generate names and evaluate novelty/diversity.
- Compare model quality with quantitative + qualitative analysis.

## Directory Structure

```
.
├── problem_1
│   ├── README.md
│   ├── clean_text.py
│   ├── web_crawler.py
│   ├── corpus_statistics.ipynb
│   ├── corpus
│   │   └── corpus.txt
│   ├── scrapped
│   │   └── scrapped-1.txt
│   └── word2vec
│       ├── cbow.py
│       ├── sgns.py
│       ├── train_cbow.py
│       ├── train_sgns.py
│       ├── infer_cbow.py
│       ├── infer_sgns.py
│       ├── experiments.py
│       └── models
│           ├── cbow
│           └── sgns
└── problem_2
	├── README.md
	├── dataset.py
	├── engine.py
	├── evaluate.py
	├── evaluation_summary.csv
	├── generate.py
	├── qualitative_analysis.txt
	├── TrainingNames.txt
	├── vocab.py
	├── rnn
	├── blstm
	├── rnn_attention
	└── models
```

## How to Run

For detailed steps, follow the README files inside each problem folder:
- Problem 1: [problem_1/README.md](problem_1/README.md)
- Problem 2: [problem_2/README.md](problem_2/README.md)
