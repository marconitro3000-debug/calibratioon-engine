from .engine import AutoAlphaResearchEngine
from .clustering import build_alpha_book


def test_alpha(alpha: str, data: dict, **kwargs):
    return AutoAlphaResearchEngine(**{k: v for k, v in kwargs.items() if k in {"neutralization", "costs", "groups", "validation", "overfitting_control", "objective"}}).test_alpha(alpha, data)


def calibrate_alpha(alpha_template: str, data: dict, param_space: dict, **kwargs):
    engine_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in {"neutralization", "costs", "groups", "validation", "overfitting_control", "objective"}}
    return AutoAlphaResearchEngine(**engine_kwargs).calibrate_alpha(alpha_template, data, param_space, **kwargs)
