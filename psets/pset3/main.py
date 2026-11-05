"""Starter orchestration; no model results are assumed or supplied."""
import argparse
from pathlib import Path
from data_utils import load_data, add_age_group, split_data, make_llm_sample
import numpy as np
import pandas as pd
from models import tune_model
from evaluation import (plot_readmission_rates, compare_models, compute_auc,
                        auc_confidence_interval, analyze_age_shift)

LABELS = ["target", "readmitted", "readmit_30d"]
SEEDS = [42, 43]   # quick test; change to [42, 43, 44, 45, 46] for final numbers

def _X(df):
    return df.drop(columns=[c for c in LABELS if c in df.columns])


def _auc_str(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return f"undefined (n={len(y)}, positives={int(y.sum())})"
    lo, hi = auc_confidence_interval(y, p)
    return f"{compute_auc(y, p):.3f} ({lo:.3f}-{hi:.3f}), n={len(y)}, pos={int(y.sum())}"


def run_models(frame, train, val, test, seed):
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    if "readmit_30d" not in frame.columns:
        for d in (frame, train, val, test):
            d["readmit_30d"] = d["target"]

    plot_readmission_rates(frame, "outputs")

    sample = pd.read_csv("outputs/llm_sample.csv")
    if "encounter_id" in sample and "encounter_id" in train:
        overlap = set(sample["encounter_id"]) & set(train["encounter_id"])
        assert not overlap, f"{len(overlap)} sample encounters are in the training set"

    results, same_sample = {}, {}
    for name in ["linear", "tree", "neural"]:
        model, best, search = tune_model(
            name, (_X(train), train["target"]), (_X(val), val["target"]), seed=seed)
        search.to_csv(out / f"search_{name}.csv", index=False)
        print(f"{name}: best params {best}; val AUC {search.loc[0, 'val_auc']:.3f}")
        results[name] = [(test["target"].values, model.predict_proba(_X(test))[:, 1])]
        same_sample[name] = model.predict_proba(_X(sample))[:, 1]

    compare_models(results, "outputs")
    print("\nFull test set AUC:")

    sample = pd.read_csv("outputs/llm_sample.csv")
    SEEDS = [42]     
    results = {n: [] for n in ["linear", "tree", "neural"]}
    same_sample = {}

    for s in SEEDS:
        # Patient-disjoint split: no patient appears in more than one partition
        tr, va, te = split_data(frame, seed=s, patient_disjoint=True)
        # Keep the 200 LLM-sample encounters/patients out of training and validation
        for col in ("encounter_id", "patient_nbr"):
            if col in sample.columns and col in tr.columns:
                ids = set(sample[col])
                tr = tr[~tr[col].isin(ids)]
                va = va[~va[col].isin(ids)]
        for name in results:
            model, best, search = tune_model(
                name, (_X(tr), tr["target"]), (_X(va), va["target"]), seed=s)
            p = model.predict_proba(_X(te))[:, 1]
            results[name].append((te["target"].values, p))
            print(f"seed {s} {name}: best {best}; val AUC {search.loc[0, 'val_auc']:.3f}; "
                  f"test AUC {compute_auc(te['target'], p):.3f}")
            if s == SEEDS[0]:             # same-sample predictions from the first split
                search.to_csv(out / f"search_{name}.csv", index=False)
                same_sample[name] = model.predict_proba(_X(sample))[:, 1]

    compare_models(results, "outputs")    # prints mean/std/range, saves the plot
    print("\nHeld-out test AUC by split:")
    for name, splits in results.items():
        for s, (y, p) in zip(SEEDS, splits):
            print(f"  {name} seed {s}: {_auc_str(y, p)}")

    for name, [(y, p)] in results.items():
        print(f"  {name}: {_auc_str(y, p)}")

    y = sample["target"].values
    print(f"\nSame-sample comparison (n={len(y)}, positives={int(y.sum())}):")
    for name, p in same_sample.items():
        print(f"  model {name}: {_auc_str(y, p)}")
    llm = {}
    for fmt in ["structured", "natural", "alternative"]:
        f = Path(f"outputs/llm_{fmt}.csv")
        if not f.exists():
            print(f"  llm {fmt}: not run yet")
            continue
        d = pd.read_csv(f)
        assert (d["target"].values == y[: len(d)]).all(), "LLM rows do not match sample order"
        bad = d["probability"].isna()
        print(f"  llm {fmt} (invalid dropped, {int(bad.sum())} invalid): "
              f"{_auc_str(d.loc[~bad, 'target'], d.loc[~bad, 'probability'])}")
        print(f"  llm {fmt} (invalid -> 0.5): "
              f"{_auc_str(d['target'], d['probability'].fillna(0.5))}")
        llm[fmt] = d["probability"].fillna(0.5).values

    low = sample["age"].astype(str).str.extract(r"(\d+)")[0].astype(float).values
    for label, mask in [("under 50", low < 50), (">=50", low >= 50)]:
        print(f"\nSample subgroup {label}: n={mask.sum()}, positives={int(y[mask].sum())}")
        for name, p in same_sample.items():
            print(f"  model {name}: {_auc_str(y[mask], p[mask])}")
        for fmt, p in llm.items():
            m = mask[: len(p)]
            print(f"  llm {fmt}: {_auc_str(y[: len(p)][m], p[m])}")

    pool = frame
    if "encounter_id" in sample and "encounter_id" in frame:
        pool = frame[~frame["encounter_id"].isin(sample["encounter_id"])]
    shift = analyze_age_shift(pool, seed=seed)
    print("\nAge shift (linear model trained on under-50):")
    for k in ["n_young", "n_old", "rate_young", "rate_old", "best_params"]:
        print(f"  {k}: {shift[k]}")
    print(f"  under-50 held-out AUC: {shift['auc_young']:.3f} {tuple(round(v, 3) for v in shift['ci_young'])}")
    print(f"  50+ AUC:               {shift['auc_old']:.3f} {tuple(round(v, 3) for v in shift['ci_old'])}")
    print("\nLargest covariate differences (|SMD|):\n", shift["numeric_smd"].head(10))
    print("\nTop coefficients:\n", shift["top_coef"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/diabetic_data.csv")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--make-llm-sample", action="store_true")
    ap.add_argument("--sample-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patient-disjoint", action="store_true")
    args = ap.parse_args()
    frame = add_age_group(load_data(args.data))
    train, val, test = split_data(frame, seed=args.seed, patient_disjoint=args.patient_disjoint)
    if args.inspect:
        print(f"Full dataset: {len(frame)} rows; positives: {frame.target.mean():.3f}")
        for label, part in (("train", train), ("val", val), ("test", test)):
            print(f"{label}: {len(part)} rows; positives: {part.target.mean():.3f}")
        print("Readmission by age:")
        print(frame.groupby("age", dropna=False).target.agg(["size", "mean"]))
    if args.make_llm_sample:
        sample = make_llm_sample(test, n=args.sample_size, seed=args.seed)
        output = Path("outputs/llm_sample.csv")
        output.parent.mkdir(exist_ok=True)
        sample.to_csv(output, index=False)
        print(f"Saved {len(sample)} held-out encounters to {output}")
    if not args.inspect and not args.make_llm_sample:
        #print("TODO: Fit and compare three model classes on train/val/test splits.")
        #print("See models.py, evaluation.py and README.md.")
        run_models(frame, train, val, test, args.seed)


if __name__ == "__main__":
    main()
