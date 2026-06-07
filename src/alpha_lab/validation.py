from dataclasses import dataclass

@dataclass(frozen=True)
class Fold:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def walk_forward_folds(n_obs: int, train_window: int = 252, test_window: int = 63, step: int | None = None):
    step = step or test_window
    folds = []
    start = 0
    while start + train_window + test_window <= n_obs:
        folds.append(Fold(start, start + train_window, start + train_window, start + train_window + test_window))
        start += step
    if not folds and n_obs > 20:
        split = int(n_obs * 0.7)
        folds.append(Fold(0, split, split, n_obs))
    return folds


def apply_fold(df, fold: Fold, part="test"):
    if part == "train":
        return df.iloc[fold.train_start:fold.train_end]
    return df.iloc[fold.test_start:fold.test_end]


def slice_data(data: dict, fold: Fold, part="test"):
    return {k: apply_fold(v, fold, part=part) for k, v in data.items()}
