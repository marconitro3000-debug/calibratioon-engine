from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AlphaResult:
    formula: str
    best_formula: str
    best_params: dict
    metrics: dict
    score: float
    verdict: str
    rejection_reasons: list[str] = field(default_factory=list)
    fold_metrics: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    signal: pd.DataFrame | None = None
    positions: pd.DataFrame | None = None
    returns: pd.Series | None = None

    def report(self) -> str:
        lines = [
            "# Alpha Research Report",
            "",
            f"Verdict: **{self.verdict}**",
            f"Formula: `{self.best_formula}`",
            f"Best params: `{self.best_params}`",
            f"Score: `{self.score:.4f}`",
            "",
            "## Metrics",
        ]
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"- {k}: {v:.6f}")
            else:
                lines.append(f"- {k}: {v}")
        if self.rejection_reasons:
            lines += ["", "## Rejection / warning reasons"]
            lines += [f"- {r}" for r in self.rejection_reasons]
        if self.diagnostics:
            lines += ["", "## Diagnostics"]
            for k, v in self.diagnostics.items():
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)


@dataclass
class AlphaBook:
    selected: list[AlphaResult]
    rejected: list[AlphaResult]
    correlation_matrix: pd.DataFrame
    summary: dict

    def report(self) -> str:
        lines = ["# Alpha Book Report", "", f"Selected: {len(self.selected)}", f"Rejected: {len(self.rejected)}", ""]
        lines.append("## Selected formulas")
        for r in self.selected:
            lines.append(f"- score={r.score:.4f}, formula=`{r.best_formula}`")
        lines.append("\n## Summary")
        for k, v in self.summary.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)
