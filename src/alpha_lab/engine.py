from .calibration import AlphaCalibrator
from .parser import AlphaParser
from .backtest import backtest_alpha
from .metrics import summarize, decay_curve
from .scoring import score_metrics
from .result import AlphaResult
from .clustering import build_alpha_book


class ResearchEngine:
    def __init__(self, *, neutralization=None, costs=None, groups=None, validation="walk_forward", overfitting_control=True, objective="robust_fitness"):
        self.neutralization = neutralization or []
        self.costs = costs or {}
        self.groups = groups
        self.validation = validation
        self.overfitting_control = overfitting_control
        self.objective = objective
        self.parser = AlphaParser()

    def test_alpha(self, alpha: str, data: dict, *, decay=1, delay_periods=1) -> AlphaResult:
        signal = self.parser.evaluate(alpha, data, groups=self.groups)
        bt = backtest_alpha(signal, data, costs=self.costs, neutralization=self.neutralization, groups=self.groups, decay=decay, delay_periods=delay_periods)
        metrics = summarize(bt, signal, data)
        metrics["ic_decay"] = decay_curve(signal, data["close"])
        score = score_metrics(metrics, self.objective)
        metrics["fitness"] = score
        rejection = []
        if metrics.get("net_return", 0.0) <= 0:
            rejection.append("negative net return after costs")
        if metrics.get("turnover", 0.0) > 5.0:
            rejection.append("turnover too high")
        verdict = "ACCEPT" if not rejection else "REVIEW"
        return AlphaResult(alpha, alpha, {}, metrics, score, verdict, rejection, signal=signal, positions=bt["positions"], returns=bt["net_returns"])

    def calibrate_alpha(self, alpha_template: str, data: dict, param_space: dict, *, optimizer="auto", max_evals=500, objective=None) -> AlphaResult:
        cal = AlphaCalibrator(costs=self.costs, neutralization=self.neutralization, groups=self.groups, validation=self.validation, objective=objective or self.objective)
        return cal.calibrate(alpha_template, data, param_space, optimizer=optimizer, max_evals=max_evals)

    def build_alpha_book(self, results, **kwargs):
        return build_alpha_book(results, **kwargs)


class AutoAlphaResearchEngine(ResearchEngine):
    def research(self, alpha_template: str, data: dict, param_space: dict | None = None, *, objective=None, optimizer="auto", max_evals=500):
        if param_space:
            return self.calibrate_alpha(alpha_template, data, param_space, optimizer=optimizer, max_evals=max_evals, objective=objective)
        return self.test_alpha(alpha_template, data)
