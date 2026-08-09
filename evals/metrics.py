# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.

"""评测指标：accuracy, F1, Matthews correlation。"""

from sklearn.metrics import accuracy_score as sk_accuracy
from sklearn.metrics import f1_score as sk_f1
from sklearn.metrics import matthews_corrcoef as sk_mcc


def accuracy_score(y_true, y_pred):
    """准确率。"""
    if len(y_true) == 0:
        raise ValueError("Empty input")
    return float(sk_accuracy(y_true, y_pred))


def f1_score(y_true, y_pred):
    """Binary F1 score。处理单一类别边界情况。"""
    if len(y_true) == 0:
        raise ValueError("Empty input")
    return float(sk_f1(y_true, y_pred, average="binary", pos_label=1, zero_division=0.0))


def matthews_corrcoef(y_true, y_pred):
    """Matthews 相关系数。"""
    if len(y_true) == 0:
        raise ValueError("Empty input")
    return float(sk_mcc(y_true, y_pred))
