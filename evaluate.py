from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def spearman_order(true_finish: pd.Series, pred_finish: pd.Series) -> float:
    return float (spearmanr(true_finish, pred_finish).correlation)

def top_n_accuracy(true_finish: pd.Series, pred_rank: pd.Series, n: int =10) ->float:
    true_top = set(true_finish.nsmallest(n).index)
    pred_top = set(pred_rank.nsmallest(n).index)
    return len(true_top & pred_top) / n

