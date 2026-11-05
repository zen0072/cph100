import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from evaluation import auc_confidence_interval

for fmt in ["structured", "natural", "alternative"]:
    try:
        d = pd.read_csv(f"outputs/llm_{fmt}.csv")
    except FileNotFoundError:
        print(fmt, "not run yet"); continue
    bad = d["probability"].isna()
    print(f"\n{fmt}: rows={len(d)}, positives={int(d.target.sum())}, "
          f"invalid={int(bad.sum())} ({100*bad.mean():.1f}%), "
          f"distinct values={d.probability.nunique()}")
    ok = d[~bad]
    if ok.target.nunique() == 2:
        lo, hi = auc_confidence_interval(ok.target, ok.probability)
        print(f"  AUC (invalid dropped, n={len(ok)}): {roc_auc_score(ok.target, ok.probability):.3f} ({lo:.3f}-{hi:.3f})")
    p = d.probability.fillna(0.5)
    lo, hi = auc_confidence_interval(d.target, p)
    print(f"  AUC (invalid -> 0.5, n={len(d)}):    {roc_auc_score(d.target, p):.3f} ({lo:.3f}-{hi:.3f})")
