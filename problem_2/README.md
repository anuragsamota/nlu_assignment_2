# NLU Assignment 2 Problem 2
In this problem we implement and compare character-level name generation models: Vanilla RNN, BLSTM, and RNN with Attention.

# How to run the project

> **Note**: Always run scripts from project root directory which is problem_2. Open terminal in this directory, then run scripts.

> **Note**: Create a Python virtual environment and install packages.

python3 -m venv .venv

For Linux, activate virtual environment using:
source ./.venv/bin/activate

Install packages using:
pip install -r requirements.txt

## Dataset
Training names are stored in TrainingNames.txt.

## How to train models (Task 1)
Steps :
1. Navigate to problem_2 directory in terminal.
2. Train Vanilla RNN:
python3 ./rnn/train_rnn.py
3. Train BLSTM:
python3 ./blstm/train_blstm.py
4. Train RNN with Attention:
python3 ./rnn_attention/train_rnn_attention.py

After training, checkpoints are saved in models:
- ./models/rnn.pt
- ./models/blstm.pt
- ./models/rnn_attention.pt

## How to run evaluation (Task 2)
Run:
python3 ./evaluate.py

This script evaluates all three models and computes:
- Novelty rate
- Diversity
- Trainable parameter count
- Hidden size, number of layers, and learning rate

Generated outputs:
- Evaluation summary CSV: ./evaluation_summary.csv
- Sample files:
  - ./samples/rnn_samples.txt
  - ./samples/blstm_samples.txt
  - ./samples/rnn_attention_samples.txt

## Notes
- Training scripts automatically use GPU if available, else CPU.
- Evaluation uses the trained checkpoints. Train models first before running evaluate.py.
- Number of generated samples for evaluation can be changed in evaluate.py by editing NUM_SAMPLES.
