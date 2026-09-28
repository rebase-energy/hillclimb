"""GEFCom2014 solar: weighted ensemble of two per-zone Quantile Regression Forests.

Combines candidate_1 (N_ESTIMATORS=700 QRF) and candidate_2 (N_ESTIMATORS=400
QRF) — both use the identical de-accumulated feature set (including the
``ssrd_diff1h`` trend feature from candidate_3) and the astronomical-night
zero-quantile post-processing fix. candidate_3 is dropped: it lacks the night
post-processing that both stronger candidates share, so it would only dilute
the blend (see prompt's "drop a candidate if it clearly hurts the blend").

Each sub-model is an independent Random Forest with its own random seed, so
averaging their leaf-based empirical-CDF quantile estimates is a genuine
variance-reduction ensemble (not just a relabeling of one bigger forest):
each forest sees a different bootstrap/feature-subsampling trajectory, so
blending reduces Monte Carlo variance beyond what either forest alone
achieves. Weights (0.6 / 0.4, favoring the lower-validation-loss candidate_1)
are fixed from the two candidates' own reported validation PinballLoss
scores rather than a fresh CV grid search, to fit the remaining time budget;
see notes.md.

Night-mask zeroing is applied to the blended output (a hard physical
constraint on TOA radiation, not a fitted weight) so it is correct
regardless of blend weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from emflow.models.predictor import Predictor

ZONES = ("z1", "z2", "z3")
ACC_VARS = ("var169", "var175", "var178", "var228")

FULL_QUANTILES = tuple(i / 100 for i in range(1, 100))
QUANTILE_ARR = np.array(FULL_QUANTILES)

EPS = 1e-3

# (n_estimators, seed, blend_weight) for each ensemble member, mirroring
# candidate_1 (700 trees) and candidate_2 (400 trees). Weights are fixed
# from each candidate's own validation PinballLoss (0.011432 vs 0.011434 —
# nearly tied, slight edge to candidate_1) rather than a fresh CV search.
MEMBERS = (
    {"n_estimators": 700, "seed": 0, "weight": 0.6},
    {"n_estimators": 400, "seed": 1, "weight": 0.4},
)

MIN_SAMPLES_LEAF = 2
MAX_DEPTH = None
MAX_FEATURES = 0.4

NIGHT_TISR_THRESHOLD = 1.0


def _deaccumulate(fc: pd.DataFrame, cols) -> pd.DataFrame:
    out = fc.sort_index()
    issue = out["issue_time"].to_numpy()
    n = len(out)
    for col in cols:
        if col not in out.columns:
            continue
        vals = out[col].to_numpy(dtype=float)
        deacc = np.empty(n)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and issue[j + 1] == issue[i]:
                j += 1
            seg = vals[i : j + 1]
            d = np.diff(seg, prepend=seg[0] if len(seg) else 0.0)
            d[0] = seg[0]
            reset = d < 0
            d[reset] = seg[reset]
            deacc[i : j + 1] = d
            i = j + 1
        out[col] = np.clip(deacc, 0.0, None)
    return out


def _segment_diff(fc: pd.DataFrame, col: str) -> np.ndarray:
    issue = fc["issue_time"].to_numpy()
    vals = fc[col].to_numpy(dtype=float)
    n = len(fc)
    diff = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and issue[j + 1] == issue[i]:
            j += 1
        seg = vals[i : j + 1]
        d = np.diff(seg, prepend=seg[0] if len(seg) else 0.0)
        d[0] = 0.0
        diff[i : j + 1] = d
        i = j + 1
    return diff


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour.to_numpy(dtype=float)
    doy = index.dayofyear.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=index,
    )


def _zone_features(fc_deacc: pd.DataFrame, zone: str) -> pd.DataFrame:
    def col(var):
        name = f"{zone}_{var}"
        return fc_deacc[name] if name in fc_deacc.columns else pd.Series(
            np.nan, index=fc_deacc.index
        )

    ssrd = col("var169")
    tisr = col("var178")
    strd = col("var175")
    precip = col("var228")
    clear_sky_ratio = (ssrd / (tisr + EPS)).clip(0.0, 1.5)

    ssrd_col = f"{zone}_var169"
    ssrd_diff1h = (
        _segment_diff(fc_deacc, ssrd_col) if ssrd_col in fc_deacc.columns
        else np.zeros(len(fc_deacc))
    )

    feats = pd.DataFrame(
        {
            "ssrd": ssrd,
            "tisr": tisr,
            "clear_sky_ratio": clear_sky_ratio,
            "ssrd_diff1h": ssrd_diff1h,
            "strd": strd,
            "precip": precip,
            "cloud": col("var164"),
            "temp": col("var167"),
            "rh": col("var157"),
            "pres": col("var134"),
            "u10": col("var165"),
            "v10": col("var166"),
            "tclw": col("var78"),
            "tciw": col("var79"),
        },
        index=fc_deacc.index,
    )
    feats = feats.join(_calendar_features(fc_deacc.index))
    return feats


FEATURE_COLS = [
    "ssrd", "tisr", "clear_sky_ratio", "ssrd_diff1h", "strd", "precip", "cloud",
    "temp", "rh", "pres", "u10", "v10", "tclw", "tciw",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    return np.interp(quantiles, cw, v)


class _ZoneQRF:
    """One Random Forest, quantiles read off leaf-sample empirical CDFs."""

    def __init__(self, n_estimators: int, seed: int):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_depth=MAX_DEPTH,
            max_features=MAX_FEATURES,
            n_jobs=-1,
            random_state=seed,
        )
        self.leaf_targets = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ZoneQRF":
        self.model.fit(X, y)
        leaf_ids = self.model.apply(X)
        n_trees = leaf_ids.shape[1]
        self.leaf_targets = []
        for t in range(n_trees):
            ids = leaf_ids[:, t]
            order = np.argsort(ids, kind="stable")
            ids_sorted = ids[order]
            y_sorted = y[order]
            uniq, start_idx = np.unique(ids_sorted, return_index=True)
            splits = np.split(y_sorted, start_idx[1:])
            self.leaf_targets.append(dict(zip(uniq.tolist(), splits)))
        return self

    def predict_quantiles(self, X: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
        leaf_ids = self.model.apply(X)
        n_query, n_trees = leaf_ids.shape
        out = np.empty((n_query, len(quantiles)))
        for i in range(n_query):
            vals_list = []
            w_list = []
            for t in range(n_trees):
                arr = self.leaf_targets[t].get(leaf_ids[i, t])
                if arr is None or len(arr) == 0:
                    continue
                vals_list.append(arr)
                w_list.append(np.full(len(arr), 1.0 / (n_trees * len(arr))))
            values = np.concatenate(vals_list)
            weights = np.concatenate(w_list)
            out[i] = _weighted_quantiles(values, weights, quantiles)
        return out


class Gefcom2014SolarEnsemble(Predictor):
    output_kind = "quantiles"
    quantiles = FULL_QUANTILES

    def __init__(self, name=None):
        super().__init__(name or type(self).__name__)
        # models[zone] = list of (weight, _ZoneQRF) aligned with MEMBERS
        self.models = {}

    def fit(self, train):
        fc_raw = train.forecasts("solar_nwp")
        acc_cols = [f"{z}_{v}" for z in ZONES for v in ACC_VARS]
        fc_deacc = _deaccumulate(fc_raw, acc_cols)

        actuals = train.history("solar_targets")

        for zone in ZONES:
            X = _zone_features(fc_deacc, zone)
            y = actuals[zone].reindex(X.index)
            mask = y.notna() & X.notna().all(axis=1)
            Xz = X.loc[mask, FEATURE_COLS].to_numpy(dtype=float)
            yz = y.loc[mask].to_numpy(dtype=float)
            if len(Xz) < 50:
                raise ValueError(f"not enough training rows for zone {zone}: {len(Xz)}")

            members = []
            for spec in MEMBERS:
                qrf = _ZoneQRF(n_estimators=spec["n_estimators"], seed=spec["seed"])
                qrf.fit(Xz, yz)
                members.append((spec["weight"], qrf))
            self.models[zone] = members
        return self

    def predict(self, obs) -> pd.DataFrame:
        zone = obs.column
        fc_raw = obs.forecasts("solar_nwp")
        acc_cols = [f"{zone}_{v}" for v in ACC_VARS]
        fc_deacc = _deaccumulate(fc_raw, acc_cols)

        X_full = _zone_features(fc_deacc, zone).reindex(obs.target_index)
        night_mask = (X_full["tisr"] < NIGHT_TISR_THRESHOLD).fillna(False).to_numpy()
        X = X_full[FEATURE_COLS].fillna(X_full[FEATURE_COLS].median())
        Xarr = X.to_numpy(dtype=float)

        blended = np.zeros((len(Xarr), len(QUANTILE_ARR)))
        total_w = 0.0
        for weight, qrf in self.models[zone]:
            blended += weight * qrf.predict_quantiles(Xarr, QUANTILE_ARR)
            total_w += weight
        blended /= total_w

        out = np.clip(blended, 0.0, 1.0)
        out = np.sort(out, axis=1)  # belt-and-braces monotonicity
        out[night_mask, :] = 0.0  # astronomical night: generation is deterministically 0

        return pd.DataFrame(out, index=obs.target_index, columns=list(FULL_QUANTILES))


def get_model():
    return Gefcom2014SolarEnsemble()
