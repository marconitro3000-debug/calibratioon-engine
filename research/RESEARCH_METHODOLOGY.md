# OpenAlphaLab V3 Research Methodology

OpenAlphaLab is built around the alpha research workflow rather than a single backtest.

## Core research question

Given a formulaic alpha or model hypothesis, does it produce a statistically defensible, cost-adjusted, out-of-sample edge that is not redundant with existing alphas?

## Methodological pillars

1. **Formulaic alpha testing**: inspired by published formulaic-alpha literature and public alpha-research platforms.
2. **Cost-aware validation**: all signals must survive transaction costs and turnover penalties.
3. **Neutralization**: alpha should be separable from market/sector exposure when required.
4. **Decay analysis**: a valid signal should have interpretable horizon behavior.
5. **Walk-forward validation**: avoids using future data in parameter selection.
6. **Overfitting diagnostics**: parameter stability, OOS degradation, DSR proxy and PBO proxy.
7. **Alpha book construction**: selected alphas should be lowly correlated and robust.

## Explicit non-goals

- This is not affiliated with WorldQuant, Jane Street, Citadel or any proprietary firm.
- It does not claim to reproduce any private platform or private model.
- It is a public research framework for reproducible alpha testing.
