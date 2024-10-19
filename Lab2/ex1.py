import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from timed_decorator.simple_timed import timed
from typing import Tuple

predicted = np.array([
    1,1,1,0,1,0,1,1,0,0
])
actual = np.array([
    1,1,1,1,0,0,1,0,0,0
])

big_size = 500000
big_actual = np.repeat(actual, big_size)
big_predicted = np.repeat(predicted, big_size)

@timed(use_seconds=True, show_args=True)
def tp_fp_fn_tn_sklearn(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, ...]:
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
    return tp, fp, fn, tn

@timed(use_seconds=True, show_args=True)
def tp_fp_fn_tn_numpy(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, ...]:
   
    combined = gt + pred
    subtracted = gt - pred
    tp = ((gt + pred) == 2)
    tn = ((gt + pred) == 0) 
    fn = ((gt - pred) == 1)
    fp = ((gt - pred) == -1)
    return combined[tp].size, subtracted[fp].size, subtracted[fn].size, combined[tn].size

@timed(use_seconds=True, show_args=True)
def accuracy_sklearn(gt: np.ndarray, pred: np.ndarray) -> float:
    return accuracy_score(gt, pred)


@timed(use_seconds=True, show_args=True)
def accuracy_numpy(gt: np.ndarray, pred: np.ndarray) -> float:
    tp,fp,fn,tn = tp_fp_fn_tn_numpy(gt,pred)
    z = tp + fp + fn + tn
    if z == 0:
        z = 1
    return (tp+tn) / z


@timed(use_seconds=True, show_args=True)
def f1_score_sklearn(gt: np.ndarray, pred: np.ndarray) -> float:
    return f1_score(gt, pred)


@timed(use_seconds=True, show_args=True)
def f1_score_numpy(gt: np.ndarray, pred: np.ndarray) -> float:
    tp,fp,fn,tn = tp_fp_fn_tn_numpy(gt,pred)
    prec_0 = tp + fp
    if prec_0 == 0:
        prec_0 = 1
    recall_0 = tp + fn
    if recall_0 == 0:
        recall_0 = 1
    prec = tp / prec_0
    recall = tp / recall_0
    return 2 * (prec * recall) / (prec + recall)


print(tp_fp_fn_tn_sklearn(big_actual,big_predicted))
print(tp_fp_fn_tn_numpy(big_actual, big_predicted))
print(accuracy_numpy(big_actual,big_predicted))
print(accuracy_sklearn(big_actual,big_predicted))

rez_1 = f1_score_sklearn(big_actual, big_predicted)
rez_2 = f1_score_numpy(big_actual, big_predicted)

assert np.isclose(rez_1, rez_2)