import re

import numpy as np

from . import operators as ops

_ALLOWED = {
    "rank": ops.rank,
    "zscore": ops.zscore,
    "scale": ops.scale,
    "delay": ops.delay,
    "delta": ops.delta,
    "returns": ops.returns,
    "ts_mean": ops.ts_mean,
    "ts_std": ops.ts_std,
    "ts_min": ops.ts_min,
    "ts_max": ops.ts_max,
    "ts_rank": ops.ts_rank,
    "ts_corr": ops.ts_corr,
    "ts_cov": ops.ts_cov,
    "decay_linear": ops.decay_linear,
    "signed_power": ops.signed_power,
    "winsorize": ops.winsorize,
    "neutralize": ops.neutralize,
    "np": np,
}


def _expand_adv_tokens(expr: str) -> str:
    # WorldQuant-style convenience token: adv20 -> adv(20)
    return re.sub(r"\badv(\d+)\b", r"adv(\1)", expr)


class AlphaParser:
    """Safe-ish formula evaluator for vectorized alpha expressions.

    It intentionally exposes only a small operator namespace and user-provided data frames.
    """

    def evaluate(self, expression: str, data: dict, groups: dict | None = None):
        expression = _expand_adv_tokens(expression)
        env = dict(_ALLOWED)
        env.update(data)

        def adv(n: int):
            if "volume" not in data:
                raise KeyError("adv(n) requires data['volume']")
            return data["volume"].rolling(int(n), min_periods=max(2, int(n) // 3)).mean()

        def group_neutralize(x):
            return ops.group_neutralize(x, groups=groups)

        env["adv"] = adv
        env["group_neutralize"] = group_neutralize
        return eval(expression, {"__builtins__": {}}, env)
