import math

import numpy as np


def parameter_stability(best_params: dict, top_params: list[dict]) -> float:
    if not top_params:
        return 0.0
    scores = []
    for k, v in best_params.items():
        vals = [p.get(k) for p in top_params if k in p]
        if not vals:
            continue
        if isinstance(v, (int, float)):
            arr = np.array([float(x) for x in vals if isinstance(x, (int, float))], dtype=float)
            if len(arr) < 2:
                scores.append(1.0)
            else:
                scale = abs(float(v)) + 1e-9
                scores.append(max(0.0, 1.0 - float(np.std(arr) / scale)))
        else:
            scores.append(float(sum(x == v for x in vals)) / len(vals))
    return float(np.mean(scores)) if scores else 0.0


def oos_degradation(train_score: float, test_score: float) -> float:
    denom = abs(train_score) + 1e-9
    return float((train_score - test_score) / denom)


def deflated_sharpe_proxy(sharpe: float, n_trials: int, n_obs: int, skew: float = 0.0, kurtosis: float = 3.0) -> float:
    # Lightweight proxy, not a full DSR implementation. Useful for ranking/auditing.
    if n_obs <= 2:
        return 0.0
    penalty = math.sqrt(max(0.0, 2.0 * math.log(max(n_trials, 1))) / n_obs)
    nonnormal = 1.0 + abs(skew) * 0.05 + max(kurtosis - 3.0, 0.0) * 0.02
    return float(sharpe - penalty * nonnormal)


def pbo_proxy(fold_scores: list[float]) -> float:
    if len(fold_scores) < 3:
        return 0.5
    arr = np.array(fold_scores, dtype=float)
    med = np.median(arr)
    bad = np.mean(arr < med)
    dispersion = np.std(arr) / (abs(np.mean(arr)) + 1e-9)
    return float(min(1.0, max(0.0, 0.5 * bad + 0.25 * min(dispersion, 2.0))))
