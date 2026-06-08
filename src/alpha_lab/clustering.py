import pandas as pd

from .result import AlphaBook


def alpha_correlation_matrix(results):
    series = {}
    for i, r in enumerate(results):
        if r.returns is not None:
            series[f"alpha_{i}"] = r.returns
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).corr()


def build_alpha_book(results, *, max_pairwise_corr=0.35, min_score=0.0, max_turnover=5.0):
    selected = []
    selected_indices = []
    rejected = []
    corr = alpha_correlation_matrix(results)
    for i, r in sorted(enumerate(results), key=lambda x: x[1].score, reverse=True):
        if r.score < min_score or r.metrics.get("turnover", 999) > max_turnover or r.verdict == "REJECT":
            rejected.append(r)
            continue
        ok = True
        for j in selected_indices:
            if corr.empty:
                continue
            ci, cj = f"alpha_{i}", f"alpha_{j}"
            if ci in corr.index and cj in corr.columns and abs(corr.loc[ci, cj]) > max_pairwise_corr:
                ok = False
                break
        if ok:
            selected.append(r)
            selected_indices.append(i)
        else:
            rejected.append(r)
    summary = {"selected": len(selected), "rejected": len(rejected), "max_pairwise_corr": max_pairwise_corr}
    return AlphaBook(selected=selected, rejected=rejected, correlation_matrix=corr, summary=summary)
