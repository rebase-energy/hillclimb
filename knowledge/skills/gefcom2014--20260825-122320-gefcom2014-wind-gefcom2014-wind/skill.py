"""Ensemble of candidate_1 (regional_speed100 + regional_speed100_std +
speed_trend3h) and candidate_2 (same regional features + speed_trend2h), both
per-zone Quantile Regression Forests (Meinshausen 2006).

candidate_3 is dropped: it lacks `regional_speed100_std`, which candidate_1's
own notes measured as a real (if modest) win, and c3 scored the worst of the
three on validation. Blending the two features it's missing back in via a
3-way ensemble would only dilute the two features that actually earned their
keep -- per the brief, an ensemble of the best two beats a diluted blend of
three.

Combination rule: for each zone, both members are per-zone quantile forests
trained on the SAME base features and differing only in the wind-speed
momentum window (3h vs 2h). Final quantiles are a per-fleet weighted average
`w * pred_trend3h + (1 - w) * pred_trend2h`, clipped to [0, 1]. The weight
`w` is fit once (pooled across all 10 zones) via a time-based internal
validation split carved out of the training data: the last 15% of the
training timeline (by origin timestamp) is held out, both members are
refit on the remaining 85% with a smaller forest for speed, and `w` is
grid-searched over {0.0, 0.1, ..., 1.0} to minimize pooled pinball loss on
that held-out slice. The final members shipped for prediction are then
refit on the FULL training data at full size. See notes.md for the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from emflow.models.predictor import Predictor

SEED = 0
QUANTILES = tuple(i / 100 for i in range(1, 100))
ZONES = [f"z{i}" for i in range(1, 11)]
TREND_WINDOWS = (3, 2)  # candidate_1's window, candidate_2's window


def _feature_names(trend_window: int) -> list[str]:
    return [
        "speed10", "speed100", "speed100_sq", "speed100_cubed",
        "dir_sin", "dir_cos", "shear", f"speed_trend{trend_window}h",
        "hour_sin", "hour_cos", "regional_speed100", "regional_speed100_std",
        "regional_speed100_max",
    ]


def _regional_speed100_stats(nwp: pd.DataFrame, index: pd.DatetimeIndex):
    """Cross-zone mean/std/max of 100m wind speed -- synoptic-scale signals.

    `max` flags widespread wind extremes (a single storm/lull hitting many
    zones at once) that the mean and std alone smooth away.
    """
    nwp = nwp.reindex(index)
    speeds = []
    for zone in ZONES:
        u100 = nwp[f"{zone}_u100"].to_numpy(dtype=float)
        v100 = nwp[f"{zone}_v100"].to_numpy(dtype=float)
        speeds.append(np.sqrt(u100 ** 2 + v100 ** 2))
    stacked = np.stack(speeds, axis=0)
    return (np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0),
            np.nanmax(stacked, axis=0))


def _zone_features(nwp: pd.DataFrame, zone: str, index: pd.DatetimeIndex,
                    trend_window: int) -> pd.DataFrame:
    nwp = nwp.reindex(index)
    u10 = nwp[f"{zone}_u10"].to_numpy(dtype=float)
    v10 = nwp[f"{zone}_v10"].to_numpy(dtype=float)
    u100 = nwp[f"{zone}_u100"].to_numpy(dtype=float)
    v100 = nwp[f"{zone}_v100"].to_numpy(dtype=float)

    speed10 = np.sqrt(u10 ** 2 + v10 ** 2)
    speed100 = np.sqrt(u100 ** 2 + v100 ** 2)
    direction = np.arctan2(v100, u100)
    hour = index.hour.to_numpy(dtype=float)

    speed_series = pd.Series(speed100, index=index)
    trend = speed_series.diff(trend_window).to_numpy()
    regional_mean, regional_std, regional_max = _regional_speed100_stats(nwp, index)

    return pd.DataFrame({
        "speed10": speed10,
        "speed100": speed100,
        "speed100_sq": speed100 ** 2,
        "speed100_cubed": speed100 ** 3,
        "dir_sin": np.sin(direction),
        "dir_cos": np.cos(direction),
        "shear": speed100 - speed10,
        f"speed_trend{trend_window}h": trend,
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "regional_speed100": regional_mean,
        "regional_speed100_std": regional_std,
        "regional_speed100_max": regional_max,
    }, index=index)


def _fit_zone_forest(nwp_sub, targets_sub, zone, trend_window, n_estimators,
                      min_samples_leaf, max_features, quantiles):
    index = nwp_sub.index
    names = _feature_names(trend_window)
    feat = _zone_features(nwp_sub, zone, index, trend_window)
    feat["y"] = targets_sub[zone].reindex(index).to_numpy()
    df = feat.dropna(subset=names + ["y"])

    X = df[names].astype(float)
    y = df["y"].to_numpy(dtype=float)
    feature_medians = X.median(axis=0).to_numpy()
    fallback_quantiles = np.quantile(y, quantiles)

    X_arr = X.to_numpy()
    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=SEED,
    )
    forest.fit(X_arr, y)

    train_leaves = forest.apply(X_arr)
    tree_leaf_y = []
    for t in range(n_estimators):
        leaves_t = train_leaves[:, t]
        order = np.argsort(leaves_t, kind="stable")
        sorted_leaves = leaves_t[order]
        sorted_y = y[order]
        unique_leaves, start_idx = np.unique(sorted_leaves, return_index=True)
        groups = np.split(sorted_y, start_idx[1:])
        tree_leaf_y.append(dict(zip(unique_leaves.tolist(), groups)))

    return forest, tree_leaf_y, feature_medians, fallback_quantiles


def _predict_zone_forest(model_tuple, nwp, zone, index, trend_window, quantiles):
    forest, tree_leaf_y, feature_medians, fallback_quantiles = model_tuple
    names = _feature_names(trend_window)
    feat = _zone_features(nwp, zone, index, trend_window)
    arr = feat[names].to_numpy()
    nan_mask = ~np.isfinite(arr)
    if nan_mask.any():
        fill = np.broadcast_to(feature_medians, arr.shape)
        arr = np.where(nan_mask, fill, arr)

    leaf_ids = forest.apply(arr)
    n_rows = leaf_ids.shape[0]
    n_q = len(quantiles)
    out = np.empty((n_rows, n_q), dtype=float)
    for i in range(n_rows):
        row = leaf_ids[i]
        parts = [d.get(leaf) for d, leaf in zip(tree_leaf_y, row)]
        parts = [p for p in parts if p is not None]
        if parts:
            pooled = np.concatenate(parts)
            out[i] = np.quantile(pooled, quantiles)
        else:
            out[i] = fallback_quantiles
    return np.clip(out, 0.0, 1.0)


def _pinball_loss(y_true, y_pred, quantiles):
    diff = y_true[:, None] - y_pred
    q = np.asarray(quantiles)[None, :]
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


class WindQuantileForestBlend(Predictor):
    """Blend of two per-zone quantile forests differing only in the
    wind-speed momentum window (3h vs 2h); blend weight fit by internal CV.
    """

    output_kind = "quantiles"
    quantiles = QUANTILES

    def __init__(self, n_estimators=300, search_n_estimators=50,
                 min_samples_leaf=8, max_features=1.0, val_frac=0.15, name=None):
        super().__init__(name)
        self.n_estimators = n_estimators
        self.search_n_estimators = search_n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.val_frac = val_frac
        self.members = {}  # zone -> {3: tuple, 2: tuple}
        self.weight = 0.5  # weight on the trend3h (candidate_1) member

    def _search_weight(self, nwp, targets):
        full_index = nwp.index
        n = len(full_index)
        cut = max(int(n * (1 - self.val_frac)), 1)
        train_idx = full_index[:cut]
        val_idx = full_index[cut:]
        if len(val_idx) < 50:
            return 0.5

        nwp_tr = nwp.reindex(train_idx)
        targets_tr = targets.reindex(train_idx)
        nwp_val = nwp.reindex(val_idx)

        zone_val_preds = {}
        zone_val_y = {}
        for zone in ZONES:
            models = {
                w: _fit_zone_forest(nwp_tr, targets_tr, zone, w, self.search_n_estimators,
                                     self.min_samples_leaf, self.max_features, self.quantiles)
                for w in TREND_WINDOWS
            }
            preds = {
                w: _predict_zone_forest(models[w], nwp_val, zone, val_idx, w, self.quantiles)
                for w in TREND_WINDOWS
            }
            y_val = targets[zone].reindex(val_idx).to_numpy(dtype=float)
            mask = np.isfinite(y_val)
            if mask.sum() == 0:
                continue
            zone_val_preds[zone] = (preds[3][mask], preds[2][mask])
            zone_val_y[zone] = y_val[mask]

        if not zone_val_y:
            return 0.5

        weight_grid = np.linspace(0.0, 1.0, 11)
        losses = np.zeros_like(weight_grid)
        for wi, w in enumerate(weight_grid):
            total, count = 0.0, 0
            for zone, y_val in zone_val_y.items():
                p3, p2 = zone_val_preds[zone]
                blend = w * p3 + (1 - w) * p2
                total += _pinball_loss(y_val, blend, self.quantiles) * len(y_val)
                count += len(y_val)
            losses[wi] = total / count
        return float(weight_grid[np.argmin(losses)])

    def fit(self, train):
        targets = train.history("wind_targets")
        nwp = train.forecasts("wind_nwp")

        self.weight = self._search_weight(nwp, targets)

        for zone in ZONES:
            self.members[zone] = {
                w: _fit_zone_forest(nwp, targets, zone, w, self.n_estimators,
                                     self.min_samples_leaf, self.max_features, self.quantiles)
                for w in TREND_WINDOWS
            }
        return self

    def predict(self, obs) -> pd.DataFrame:
        zone = obs.column
        nwp = obs.forecasts("wind_nwp")
        p3 = _predict_zone_forest(self.members[zone][3], nwp, zone, obs.target_index, 3, self.quantiles)
        p2 = _predict_zone_forest(self.members[zone][2], nwp, zone, obs.target_index, 2, self.quantiles)
        blend = np.clip(self.weight * p3 + (1 - self.weight) * p2, 0.0, 1.0)
        return pd.DataFrame(blend, index=obs.target_index, columns=list(self.quantiles))


def get_model():
    return WindQuantileForestBlend()
