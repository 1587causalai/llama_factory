from typing import Dict

import numpy as np
from transformers import EvalPrediction

from ...extras import logging

logger = logging.get_logger(__name__)

def ComputeAccuracy():
    def compute_accuracy(eval_pred: EvalPrediction) -> Dict[str, float]:
        predictions = eval_pred.predictions[0]
        if len(predictions.shape) == 3:
            predictions = predictions.mean(axis=1)  # mean on sequence_length dimension
            
        accuracies = []
        for i in range(0, len(predictions), 2):
            if i + 1 < len(predictions):
                acc = float(predictions[i] > predictions[i + 1])
                accuracies.append(acc)
                
        accuracy = np.mean(accuracies)
        return {"accuracy": accuracy}
        
    return compute_accuracy 