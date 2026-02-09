import math
from typing import List, Dict

def dcg(rels: List[float], k: int) -> float:
    rels = rels[:k]
    s = 0.0
    for i, r in enumerate(rels, start=1):
        s += (2**r - 1) / math.log2(i + 1)
    return s

def ndcg(rels: List[float], k: int) -> float:
    ideal = sorted(rels, reverse=True)
    denom = dcg(ideal, k)
    return 0.0 if denom == 0 else dcg(rels, k) / denom

def precision_at_k(binary: List[int], k: int) -> float:
    binary = binary[:k]
    return sum(binary) / k if k else 0.0
