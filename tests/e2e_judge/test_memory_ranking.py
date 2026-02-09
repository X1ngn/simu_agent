import os
import pytest
from tests.helpers.metrics import ndcg, precision_at_k

@pytest.mark.e2e_judge
@pytest.mark.skipif(os.getenv("RUN_E2E_JUDGE") != "1", reason="E2E judge disabled by default")
def test_memory_ranking_quality():
    """
    TODO:
    - case: query + retrieved_ids（系统返回） + relevance（人工标注）
    - 计算 NDCG@K、P@K，设定回归阈值
    """
    retrieved_ids = ["m3", "m1", "m2"]
    relevance = {"m1": 3.0, "m2": 1.0, "m3": 2.0}

    rels = [relevance.get(i, 0.0) for i in retrieved_ids]
    # TODO: assert ndcg(rels, k=3) >= 0.8
    # TODO: assert precision_at_k([1 if r > 0 else 0 for r in rels], k=3) >= 0.67
