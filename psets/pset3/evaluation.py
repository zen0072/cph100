"""Student implementations for Parts 1.1, 1.2, and 1.4."""
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from models import tune_model

def plot_readmission_rates(frame, output_dir):
    """TODO: Plot readmission rates and group sizes by age, gender, race."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    overall = frame["readmit_30d"].mean()


    for e, col in enumerate(["age","gender","race"]):
        g = frame[col].replace("?", None).fillna("Missing")
        stats = frame.groupby(g)["readmit_30d"].agg(rate="mean", n="size")
        if col != "age": 
            stats = stats.sort_values("rate", ascending=False)
        axes[0,e].bar(stats.index.astype(str), stats["rate"])
        axes[0, e].axhline(overall, color="red", ls="--", label="overall")
        axes[0, e].set_title(f"30-day readmission rate by {col}")
        axes[0, e].legend()

        axes[1, e].bar(stats.index.astype(str), stats["n"])
        axes[1, e].set_yscale("log")
        axes[1, e].set_title(f"Group size by {col}")
        for ax in axes[:, e]:
            ax.tick_params(axis="x", rotation=40)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "readmission_by_demographics.png"), dpi=150)
    plt.close(fig)
    #raise NotImplementedError("Student TODO: demographic exploration")


def compute_auc(y_true, probabilities):
    """TODO: Compute ROC AUC from predicted probabilities (not class labels)."""
    y=np.asarray(y_true)
    p=np.asarray(probabilities, dtype=float)
    return float(roc_auc_score(y, p))
    #raise NotImplementedError("Student TODO: AUC")


def auc_confidence_interval(y_true, probabilities, seed=42, confidence=0.95,
                            n_bootstrap=1000):
    """Return (lower, upper) percentile bootstrap CI for ROC AUC.

    Resample observation pairs, skipping single-class samples. This assumes
    independently sampled evaluation rows; repeated patient encounters may
    require patient-level clustered resampling instead.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("y_true and probabilities must be equal-length vectors")
    if len(y) == 0 or not np.isfinite(p).all():
        raise ValueError("Nonempty finite probabilities are required")
    if len(np.unique(y)) != 2:
        raise ValueError("ROC AUC requires both outcome classes")
    if not 0 < confidence < 1 or n_bootstrap < 1:
        raise ValueError("Invalid confidence or n_bootstrap")
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        aucs.append(roc_auc_score(y[indices], p[indices]))
    if not aucs:
        raise ValueError("No bootstrap samples contained both classes")
    alpha = (1 - confidence) / 2
    return tuple(float(v) for v in np.quantile(aucs, [alpha, 1 - alpha]))


def compare_models(results, output_dir):
    """TODO: Plot model AUCs with confidence intervals and describe variation."""
    #raise NotImplementedError("Student TODO: model comparison plot")
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for x, (name, splits) in enumerate(results.items()):
        aucs = np.array([compute_auc(y, p) for y, p in splits])
        cis = np.array([auc_confidence_interval(y, p) for y, p in splits])
        offs = np.linspace(-0.2, 0.2, len(splits)) if len(splits) > 1 else np.array([0.0])
        ax.errorbar(x + offs, aucs, yerr=[aucs - cis[:, 0], cis[:, 1] - aucs],
                    fmt="o", capsize=3)
        ax.hlines(aucs.mean(), x - 0.3, x + 0.3, color="k")
        print(f"{name}: mean AUC {aucs.mean():.3f}, std {aucs.std(ddof=1) if len(aucs) > 1 else 0:.3f}, "
              f"range {aucs.min():.3f}-{aucs.max():.3f}")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(results.keys())
    ax.set_ylabel("Held-out test AUC")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=150)
    plt.close(fig)

def analyze_age_shift(frame, seed=42):
    """TODO: Train on under-50 patients, evaluate on held-out under-50 and
    on 50-plus patients. Investigate outcome/feature differences and avoid
    using the older group's outcomes in training or model selection.
    """
    #raise NotImplementedError("Student TODO: distribution shift")
    low = frame["age"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    young = frame[low < 50].reset_index(drop=True)
    old = frame[low >= 50].reset_index(drop=True)

    drop = [c for c in ("age", "readmitted", "readmit_30d") if c in frame.columns]
    X = lambda d: d.drop(columns=drop)
    y = lambda d: d["readmit_30d"].astype(int).values

    g = young["patient_nbr"].values if "patient_nbr" in young else np.arange(len(young))
    tv, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(young, groups=g))
    tr, va = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(young.iloc[tv], groups=g[tv]))
    tr_d, va_d, te_d = young.iloc[tv[tr]], young.iloc[tv[va]], young.iloc[te]
    model, best, _ = tune_model("linear", (X(tr_d), y(tr_d)), (X(va_d), y(va_d)), seed=seed)

    p_young = model.predict_proba(X(te_d))[:, 1]
    p_old = model.predict_proba(X(old))[:, 1]
    out = {
        "rate_young": float(young["readmit_30d"].mean()),
        "rate_old": float(old["readmit_30d"].mean()),
        "auc_young": compute_auc(y(te_d), p_young),
        "ci_young": auc_confidence_interval(y(te_d), p_young, seed=seed),
        "auc_old": compute_auc(y(old), p_old),
        "ci_old": auc_confidence_interval(y(old), p_old, seed=seed),
    }

    skip = drop + [c for c in ("encounter_id", "patient_nbr") if c in frame.columns]
    num = young.drop(columns=skip).select_dtypes("number").columns
    sd = np.sqrt((young[num].var() + old[num].var()) / 2).replace(0, np.nan)
    out["numeric_smd"] = ((old[num].mean() - young[num].mean()) / sd).abs().sort_values(ascending=False)

    names = model.named_steps["prep"].get_feature_names_out()
    coef = pd.Series(model.named_steps["clf"].coef_[0], index=names)
    out["top_coef"] = coef.reindex(coef.abs().sort_values(ascending=False).index).head(15)
    out["best_params"] = best
    out["n_young"], out["n_old"] = len(young), len(old)
    return out

