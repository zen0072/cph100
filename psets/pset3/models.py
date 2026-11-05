"""Part 1.2: preprocessing and three prediction models.

You may use scikit-learn. Do not fit transformations on validation/test data.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import roc_auc_score
import pandas as pd



def build_preprocessor(X_train):
    """Return an unfitted preprocessing transformer for diabetes features.

    Numeric columns: median imputation. Categorical columns: most-frequent
    imputation followed by one-hot encoding with unknown categories ignored.
    The returned transformer must be fitted ONLY on the training partition.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # The caller supplies feature columns, not labels or patient identifiers.
    excluded = {"readmitted", "target", "encounter_id", "patient_nbr", "age_group"}
    feature_names = [c for c in X_train.columns if c not in excluded]
    numeric = X_train[feature_names].select_dtypes(include="number").columns.tolist()
    categorical = [c for c in feature_names if c not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric),
         ("categorical", categorical_pipe, categorical)],
        remainder="drop", sparse_threshold=0
    )



def build_model(kind, X_train, params=None, seed=42):
    """Return an UNFITTED model pipeline.

    kind is one of 'linear', 'tree', 'neural'.

    TODO:
      - Choose and instantiate one model from each family.
      - Apply the same train-only preprocessing appropriately.
      - Expose tunable hyperparameters via params.
      - Set random seeds where supported.
    """
    _DEFAULTS = {
    "linear": dict(C=1.0, penalty="l2", max_iter=2000),
    "tree":   dict(learning_rate=0.1, max_depth=None, max_leaf_nodes=31,
                   min_samples_leaf=20, l2_regularization=0.0, max_iter=200),
    "neural": dict(hidden_layer_sizes=(64, 32), alpha=1e-3,
                   learning_rate_init=1e-3, max_iter=300),
}
    p = {**_DEFAULTS[kind], **(params or {})}

    steps = [("prep", build_preprocessor(X_train))]
    if kind == "linear":
        clf = LogisticRegression(random_state=seed, **p)
    elif kind == "tree":
        clf = HistGradientBoostingClassifier(random_state=seed, **p)
    else:
        clf = MLPClassifier(early_stopping=True, n_iter_no_change=10,
                            random_state=seed, **p)
        
    steps.append(("clf", clf))
    return Pipeline(steps)




def tune_model(kind, train, val, seed=42):
    """Select hyperparameters using train and validation partitions only.

    TODO: Train candidate models, compute validation AUC, select a candidate,
    and record the search and results. Never select using the test set.
    """
    #raise NotImplementedError("Student TODO: hyperparameter search")
    _GRIDS = {
    "linear": {"C": [0.001, 0.01, 0.1, 1, 10]},
    "tree": {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31],
             "min_samples_leaf": [20, 100], "l2_regularization": [0.0, 1.0]},
    "neural": {"hidden_layer_sizes": [(32,), (64, 32)],
               "alpha": [1e-4, 1e-2], "learning_rate_init": [1e-3, 3e-3]},
}
    X_tr, y_tr = train
    X_va, y_va = val

    rows = []
    for params in ParameterGrid(_GRIDS[kind]):
        model = build_model(kind, X_tr, params, seed).fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
        rows.append({"params": params, "val_auc": auc})

    results = pd.DataFrame(rows).sort_values("val_auc", ascending=False).reset_index(drop=True)
    best = results.loc[0, "params"]
    final = build_model(kind, X_tr, best, seed).fit(X_tr, y_tr)
    return final, best, results