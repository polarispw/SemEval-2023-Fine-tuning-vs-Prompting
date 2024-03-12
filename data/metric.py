"""
This file contains metrics for different datasets
"""
import difflib
from typing import Tuple

import evaluate
import numpy as np
import torch.nn

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_acc_f1(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, np.ndarray):  # seq classifier output
        predictions = predictions.flatten()
        predictions = np.where(predictions > 0.3, 1, 0)
        labels = labels.flatten()
    elif isinstance(predictions, Tuple):  # seq2seq classifier output
        predictions = predictions[0]
        predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)['accuracy']
    f1_score = f1.compute(predictions=predictions, references=labels, average='macro')['f1']
    category_f1 = []

    return {"accuracy": acc, "f1": f1_score}


def compute_simcse(eval_pred):
    predictions, _ = eval_pred
    sim_score = np.average(predictions)
    return {"avg_sim_ingroup": sim_score}

