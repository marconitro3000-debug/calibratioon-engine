import math
import numpy as np
from .parser import AlphaParser
from .backtest import backtest_alpha
from .metrics import summarize
from .scoring import score_metrics
from .optimizers import candidate_stream, genetic_candidates
from .validation import walk_forward_folds, slice_data
from .overfitting import parameter_stability, deflated_sharpe_proxy, pbo_proxy
from .result import AlphaResult


class AlphaCalibrator:
    def __init__(self, *, costs=None, neutralization=None, groups=None, validation="walk_forward", objective="robust_fitness", seed=42):
        self.costs = costs or {}
        self.neutralization = neutralization or []
        self.groups = groups
        self.validation = validation
        self.objective = objective
        self.seed = seed
        self.parser = AlphaParser()

    def evaluate_formula(self, formula: str, data: dict, *, decay=1, delay_periods=1):
        signal = self.parser.evaluate(formula, data, groups=self.groups)
        bt = backtest_alpha(signal, data, costs=self.costs, neutralization=self.neutralization, groups=self.groups, decay=decay, delay_periods=delay_periods)
        metrics = summarize(bt, signal, data)
        score = score_metrics(metrics, self.objective)
        return signal, bt, metrics, score

    def _fold_scores(self, formula: str, data: dict, decay=1):
        folds = walk_forward_folds(len(next(iter(data.values()))))
        out = []
        for fold in folds:
            test_data = slice_data(data, fold, part="test")
            try:
                _, _, m, s = self.evaluate_formula(formula, test_data, decay=decay)
                out.append((m, s))
            except Exception:
                out.append(({"net_sharpe": 0.0}, -1e9))
        return out

    def calibrate(self, alpha_template: str, data: dict, param_space: dict, *, optimizer="auto", max_evals=500, min_sharpe=0.2, max_turnover=5.0):
        records = []

        def eval_params(params):
            formula = alpha_template.format(**params)
            decay = params.get("decay", 1)
            try:
                _, _, metrics, score = self.evaluate_formula(formula, data, decay=decay)
                return score
            except Exception:
                return -1e12

        if optimizer == "genetic":
            candidates = genetic_candidates(param_space, eval_params, seed=self.seed)
        else:
            candidates = candidate_stream(param_space, optimizer=optimizer, max_evals=max_evals, seed=self.seed)

        for i, params in enumerate(candidates):
            if i >= max_evals:
                break
            formula = alpha_template.format(**params)
            decay = params.get("decay", 1)
            try:
                signal, bt, metrics, score = self.evaluate_formula(formula, data, decay=decay)
            except Exception as e:
                records.append({"params": params, "formula": formula, "score": -1e12, "error": str(e)})
                continue
            records.append({"params": params, "formula": formula, "score": score, "metrics": metrics})

        records = sorted(records, key=lambda x: x.get("score", -1e12), reverse=True)
        if not records:
            raise ValueError("no calibration candidates evaluated")
        best = records[0]
        best_formula = best["formula"]
        best_params = best["params"]
        decay = best_params.get("decay", 1)
        signal, bt, metrics, score = self.evaluate_formula(best_formula, data, decay=decay)
        fold_pairs = self._fold_scores(best_formula, data, decay=decay)
        fold_metrics = [m for m, _ in fold_pairs]
        fold_scores = [s for _, s in fold_pairs]
        top_params = [r["params"] for r in records[: min(10, len(records))]]
        stability = parameter_stability(best_params, top_params)
        dsr = deflated_sharpe_proxy(metrics.get("net_sharpe", 0.0), n_trials=len(records), n_obs=len(bt["net_returns"]))
        pbo = pbo_proxy(fold_scores)

        rejection = []
        if metrics.get("net_sharpe", 0.0) < min_sharpe:
            rejection.append("net Sharpe below minimum threshold")
        if metrics.get("turnover", 0.0) > max_turnover:
            rejection.append("turnover too high for assumed costs")
        if metrics.get("net_return", 0.0) <= 0:
            rejection.append("negative net return after costs")
        if stability < 0.35:
            rejection.append("parameter stability is weak across top candidates")
        if pbo > 0.55:
            rejection.append("high probability of backtest overfitting proxy")
        if dsr < 0:
            rejection.append("deflated Sharpe proxy is negative")

        verdict = "ACCEPT" if not rejection else "REVIEW" if len(rejection) <= 2 else "REJECT"
        diagnostics = {
            "n_candidates": len(records),
            "optimizer": optimizer,
            "parameter_stability": round(stability, 4),
            "deflated_sharpe_proxy": round(dsr, 4),
            "pbo_proxy": round(pbo, 4),
            "fold_score_mean": round(float(np.mean(fold_scores)) if fold_scores else 0.0, 4),
            "fold_score_std": round(float(np.std(fold_scores)) if fold_scores else 0.0, 4),
        }
        metrics = dict(metrics)
        metrics.update({"parameter_stability": stability, "deflated_sharpe_proxy": dsr, "pbo_proxy": pbo, "fitness": score})
        return AlphaResult(
            formula=alpha_template,
            best_formula=best_formula,
            best_params=best_params,
            metrics=metrics,
            score=score,
            verdict=verdict,
            rejection_reasons=rejection,
            fold_metrics=fold_metrics,
            diagnostics=diagnostics,
            signal=signal,
            positions=bt["positions"],
            returns=bt["net_returns"],
        )
