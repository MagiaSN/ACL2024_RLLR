import numpy as np
from typing import Dict, Sequence, Tuple, Union


def compute_accuracy(eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
    preds, _ = eval_preds
    return {"accuracy": (preds[0] > preds[1]).sum() / len(preds[0])}


def compute_accuracy_and_margin(eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
    preds, _ = eval_preds
    return {
        "accuracy": (preds[0] > preds[1]).sum() / len(preds[0]),
        "margin": (preds[0] - preds[1]).sum() / len(preds[0]),
    }
