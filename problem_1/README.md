# NLU Assignment 2 Problem 1
In this problem we need to tain word2vec models i.e CBOW and Skipgram with Negetive Sampling

# How to run the project

> **Note** : Always run scripts from project root directory which is `problem_1`. Open terminal in this directory then run the scripts.

> **Note** : Create a python virtual environment and install all packages.

```cmd
python3 -m venv .venv
```

For linux activate virtual environment using
```cmd
source "./.venv/bin/activate"
```

Install Packages using
```cmd
pip install -r requirements.txt
```

## How to train the models for Task 2
1) Navigate to the project directory in terminal or open terminal in that directory as given in the note above.

2) To train CBOW model
    ```cmd
    python3 ./word2vec/train_cbow.py
    ```
    In this is file there is just one function call to train the model with different parameters. Which can be tweaked to train model with different parameters.

3) To train SkipGram model
   ```cmd
   python3 ./word2vec/train_sgns.py
   ```
## How to infer trained models
> **Note** : Please change the variable `cbow_model_path` to correct model path you want to infer. Models are saved in `./word2vec/cbow/models` and `./word2vec/sgns/models` based on model type.

### Infer CBOW model
1) Change the print and function input statements below in script `./word2vec/infer_cbow.py` to your desired outputs.
2) Execute the python script
    ``` cmd
    python3 ./word2vec/infer_cbow.py
    ```

### Infer SkipGram model
1) Change the print and function input statements below in script `./word2vec/infer_sgns.py` to your desired outputs.
2) Execute the python script
    ``` cmd
    python3 ./word2vec/infer_sgns.py
    ```

## How to run experiments for Task 3
1) Run `./word2vec/experiments.py` file to execute the experiments
   ```cmd
   python3 ./word2vec/experiments.py
   ```

## How to generate visualizations for Task 4