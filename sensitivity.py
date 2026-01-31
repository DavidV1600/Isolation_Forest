import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import os

import replicate_paper as rp

import warnings
warnings.filterwarnings("ignore")

def run_parameter_sensitivity():
    datasets = [
    "arrhythmia",
    "pima",
    "spambase",
    "satellite",
    "pendigits",
    "annthyroid",
    "mnist",
]
    seeds = list(range(10))
    out_csv = "./outputs/parameter_sensitivity.csv"

    def needs_scaling(model_name: str) -> bool:
        return model_name.startswith("LOF") or model_name.startswith("OCSVM")

    def fit_score_eval(model_name: str, factory, X, y, seed: int):
        if needs_scaling(model_name):
            X_eval = StandardScaler().fit_transform(X)
        else:
            X_eval = X

        t0 = time.time()
        model = factory(seed)
        model.fit(X_eval)
        scores = np.asarray(rp.score_model(model_name, model, X_eval)).reshape(-1)
        elapsed = time.time() - t0

        roc = roc_auc_score(y, scores)
        pr = average_precision_score(y, scores)
        return roc, pr, elapsed

    def eval_across_seeds(dataset_name: str, sweep_name: str, config_name: str, model_name: str, factory, X, y):
        rocs, prs, times = [], [], []
        for seed in seeds:
            roc, pr, t = fit_score_eval(model_name, factory, X, y, seed)
            rocs.append(roc); prs.append(pr); times.append(t)

        rocs = np.array(rocs); prs = np.array(prs); times = np.array(times)
        return {
            "dataset": dataset_name,
            "sweep": sweep_name,
            "model": model_name,
            "config": config_name,
            "roc_mean": float(rocs.mean()),
            "roc_std": float(rocs.std()),
            "pr_mean": float(prs.mean()),
            "pr_std": float(prs.std()),
            "time_mean": float(times.mean()),
            "time_std": float(times.std()),
        }

    # IMPORTANT: isotree has standardize_data=True by default.
    iso_common = dict(
        nthreads=-1,
        standardize_data=False,
        missing_action="auto",
        scoring_metric="depth",
    )

    standard_depth = 8

    sweeps = []

    # Sweep A: IF capacity (trees + depth)  if vs if-u
    sweeps.append((
        "IF_depth_and_trees",
        [
            ("IF_t100_d8", "IF", lambda s: rp.IsolationForest(
                ntrees=100, sample_size=256, ndim=1, max_depth=standard_depth,
                prob_pick_pooled_gain=0.0, prob_pick_avg_gain=0.0, random_seed=s, **iso_common
            )),
            ("IF_t200_d8", "IF", lambda s: rp.IsolationForest(
                ntrees=200, sample_size=256, ndim=1, max_depth=standard_depth,
                prob_pick_pooled_gain=0.0, prob_pick_avg_gain=0.0, random_seed=s, **iso_common
            )),
            ("IF_t100_dNone", "IF", lambda s: rp.IsolationForest(
                ntrees=100, sample_size=256, ndim=1, max_depth=None,
                prob_pick_pooled_gain=0.0, prob_pick_avg_gain=0.0, random_seed=s, **iso_common
            )),
            ("IF_t200_dNone", "IF", lambda s: rp.IsolationForest(
                ntrees=200, sample_size=256, ndim=1, max_depth=None,
                prob_pick_pooled_gain=0.0, prob_pick_avg_gain=0.0, random_seed=s, **iso_common
            )),
        ]
    ))

    # Sweep B: IF sample_size (locality vs stability)
    sweeps.append((
        "IF_sample_size",
        [
            ("IF_s64", "IF", lambda s: rp.IsolationForest(
                ntrees=100, sample_size=64, ndim=1, max_depth=standard_depth,
                prob_pick_pooled_gain=0.0, prob_pick_avg_gain=0.0, random_seed=s, **iso_common
            )),
            ("IF_s128", "IF", lambda s: rp.IsolationForest(
                ntrees=100, sample_size=128, ndim=1, max_depth=standard_depth,
                prob_pick_pooled_gain=0.0, prob_pick_avg_gain=0.0, random_seed=s, **iso_common
            )),
            ("IF_s256", "IF", lambda s: rp.IsolationForest(
                ntrees=100, sample_size=256, ndim=1, max_depth=standard_depth,
                prob_pick_pooled_gain=0.0, prob_pick_avg_gain=0.0, random_seed=s, **iso_common
            )),
            ("IF_s512", "IF", lambda s: rp.IsolationForest(
                ntrees=100, sample_size=512, ndim=1, max_depth=standard_depth,
                prob_pick_pooled_gain=0.0, prob_pick_avg_gain=0.0, random_seed=s, **iso_common
            )),
        ]
    ))

    # Sweep C: geometry (ndim=1 vs 2) while keeping random splits
    sweeps.append((
        "Split_geometry_ndim",
        [
            ("ndim1_random", "IF", lambda s: rp.IsolationForest(
                ntrees=100, sample_size=256, ndim=1, max_depth=standard_depth,
                prob_pick_pooled_gain=0.0, prob_pick_avg_gain=0.0, random_seed=s, **iso_common
            )),
            ("ndim2_random", "EIF", lambda s: rp.IsolationForest(
                ntrees=100, sample_size=256, ndim=2, max_depth=standard_depth,
                prob_pick_pooled_gain=0.0, prob_pick_avg_gain=0.0, random_seed=s, **iso_common
            )),
        ]
    ))

    # Sweep D: FCF trees + sample_size
    sweeps.append((
        "FCF_trees_and_samplesize",
        [
            ("FCF_t100_s256", "FCF", lambda s: rp.IsolationForest(
                ntrees=100, sample_size=256, ndim=2, ntry=1, max_depth=None,
                prob_pick_pooled_gain=1.0, prob_pick_avg_gain=0.0, random_seed=s, **iso_common
            )),
            ("FCF_t200_s256", "FCF", lambda s: rp.IsolationForest(
                ntrees=200, sample_size=256, ndim=2, ntry=1, max_depth=None,
                prob_pick_pooled_gain=1.0, prob_pick_avg_gain=0.0, random_seed=s, **iso_common
            )),
            ("FCF_t200_s128", "FCF", lambda s: rp.IsolationForest(
                ntrees=200, sample_size=128, ndim=2, ntry=1, max_depth=None,
                prob_pick_pooled_gain=1.0, prob_pick_avg_gain=0.0, random_seed=s, **iso_common
            )),
            ("FCF_t200_s512", "FCF", lambda s: rp.IsolationForest(
                ntrees=200, sample_size=512, ndim=2, ntry=1, max_depth=None,
                prob_pick_pooled_gain=1.0, prob_pick_avg_gain=0.0, random_seed=s, **iso_common
            )),
        ]
    ))

    # Sweep E: SCiForest sensitivity (ntry + penalize_range)
    sweeps.append((
        "SCiF_ntry_and_range_penalty",
        [
            ("SCiF_ntry1_pen0", "SCiF", lambda s: rp.IsolationForest(
                ntrees=100, sample_size=256, ndim=2, ntry=1, max_depth=standard_depth,
                prob_pick_avg_gain=1.0, prob_pick_pooled_gain=0.0,
                penalize_range=False, random_seed=s, **iso_common
            )),
            ("SCiF_ntry10_pen0", "SCiF", lambda s: rp.IsolationForest(
                ntrees=100, sample_size=256, ndim=2, ntry=10, max_depth=standard_depth,
                prob_pick_avg_gain=1.0, prob_pick_pooled_gain=0.0,
                penalize_range=False, random_seed=s, **iso_common
            )),
            ("SCiF_ntry10_pen1", "SCiF", lambda s: rp.IsolationForest(
                ntrees=100, sample_size=256, ndim=2, ntry=10, max_depth=standard_depth,
                prob_pick_avg_gain=1.0, prob_pick_pooled_gain=0.0,
                penalize_range=True, random_seed=s, **iso_common
            )),
        ]
    ))

    # Sweep F: scoring metric (depth vs density) for one representative model
    sweeps.append((
        "Scoring_metric",
        [
            ("IF_score_depth", "IF", lambda s: rp.IsolationForest(
                ntrees=100, sample_size=256, ndim=1, max_depth=standard_depth,
                prob_pick_pooled_gain=0.0, prob_pick_avg_gain=0.0,
                scoring_metric="depth", random_seed=s,
                **{k: v for k, v in iso_common.items() if k != "scoring_metric"}
            )),
            ("IF_score_density", "IF", lambda s: rp.IsolationForest(
                ntrees=100, sample_size=256, ndim=1, max_depth=standard_depth,
                prob_pick_pooled_gain=0.0, prob_pick_avg_gain=0.0,
                scoring_metric="density", random_seed=s,
                **{k: v for k, v in iso_common.items() if k != "scoring_metric"}
            )),
        ]
    ))









    # RUN ALL SWEEPS ACROSS DATASETS
    rows = []

    for ds in datasets:
        print("\n" + "=" * 80)
        print(f"DATASET: {ds.upper()}")
        print("=" * 80)
        X, y = rp.load_dataset(ds)

        for sweep_name, configs in sweeps:
            print(f"\n-- Sweep: {sweep_name} --")
            for config_name, model_name, factory in configs:
                res = eval_across_seeds(ds, sweep_name, config_name, model_name, factory, X, y)
                rows.append(res)
                print(
                    f"{config_name:22s} | ROC {res['roc_mean']:.4f}±{res['roc_std']:.4f} "
                    f"| PR {res['pr_mean']:.4f}±{res['pr_std']:.4f} "
                    f"| time {res['time_mean']:.3f}s"
                )

    df = pd.DataFrame(rows)
    df = df[[
        "dataset", "sweep", "model", "config",
        "roc_mean", "roc_std", "pr_mean", "pr_std",
        "time_mean", "time_std"
    ]]
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    return df















if __name__ == "__main__":
    run_parameter_sensitivity()