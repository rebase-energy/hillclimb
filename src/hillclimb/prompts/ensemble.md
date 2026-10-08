You are building the FINAL solution for the problem below by ensembling the strongest candidate solutions produced so far.

# Problem

{{description}}

# Objective

Optimize **{{metric_name}}** ({{direction}}).

# Candidate solutions

The top candidate scripts are in your working directory. Each is self-contained and trains, validates, and predicts:

{{candidates_table}}

# Instruction

Write `solution.py` that combines **at least two** of these candidates into one stronger model:

- Retrain the candidate models inside your script (import nothing from the candidate files at runtime — copy the relevant model-building code into `solution.py` so it stays self-contained).
- Choose a combination suited to the metric: average predicted probabilities for log-loss/AUC metrics, (weighted) mean for regression, majority vote or averaged probabilities with a threshold for accuracy.
- Fit any blend weights with internal cross-validation on the training data (e.g., out-of-fold predictions + a simple weight search or non-negative least squares). Do NOT fit weights on data you cannot label.
- Drop a candidate if it clearly hurts the blend — an ensemble of the best two beats a diluted blend of three.
- State the combination rule and final weights in `notes.md`.

The ensemble must still satisfy the full output contract below, and its runtime includes retraining every member — budget accordingly.

{{contract}}
