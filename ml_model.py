"""
Hybrid physics + machine-learning tool-life estimation (fully offline).

Strategy:
1. Stratified (Latin-hypercube style) sampling of cutting-parameter combinations
   across the active pairing's verified parameter domain, plus overhang and
   roughing variations, labeled by the physics engine (Extended Taylor + setup
   multipliers) via tool_physics.calculate_tool_life, invoked with lightweight
   stand-in objects so the real physics code path runs unchanged.
2. Three-model ensemble on log10(tool life): GradientBoosting + HistGradient-
   Boosting + RandomForest, combined with fixed weights.
3. Uncertainty band via a conformal 80% half-width from held-out simulated
   runs, widened by any measured-trial disagreement.
4. Measured trials calibrate the sim-trained ensemble out-of-sample: a median
   bias correction (with >= 3 anchors) applied to every prediction, plus band
   widening from their disagreement. They never enter the training set, which
   would make the bias residuals in-sample and collapse the correction.
5. Blend ML and physics estimates with a log-space weighted geometric mean.

Honesty contract: the ML layer is a learned refinement of the physics model,
not independent ground truth. Holdout metrics describe agreement with the
physics model on held-out simulated conditions; measured-trial agreement is
the only real-world signal reported.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.metrics import r2_score
    SKLEARN_AVAILABLE = True
    SKLEARN_IMPORT_ERROR: Optional[str] = None
except ImportError as exc:  # App Control / missing install — app falls back to physics-only
    GradientBoostingRegressor = HistGradientBoostingRegressor = RandomForestRegressor = None
    r2_score = None
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = str(exc)


class NumpyRidgeRegressor:
    """Small dependency-free regression fallback for locked-down Windows PCs.

    It intentionally uses only NumPy, which is already required by the physics engine.
    The engineered log-space features preserve the smooth Taylor response while allowing
    local measured anchors to shift the estimate.
    """
    def __init__(self, ridge: float = 1e-4):
        self.ridge = ridge
        self.coefficients_ = None
        self.feature_importances_ = np.ones(5, dtype=float) / 5.0

    @staticmethod
    def _design(X):
        X = np.asarray(X, dtype=float)
        base = X[:, :5]
        return np.column_stack((
            np.ones(len(base)), base, base[:, 0] ** 2, base[:, 1] ** 2,
            base[:, 2] ** 2, base[:, 0] * base[:, 1], base[:, 0] * base[:, 2],
            base[:, 1] * base[:, 2],
        ))

    def fit(self, X, y, sample_weight=None):
        design = self._design(X)
        y = np.asarray(y, dtype=float)
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        root_w = np.sqrt(np.clip(weights, 1e-9, None))[:, None]
        weighted_design = design * root_w
        weighted_y = y * root_w[:, 0]
        penalty = np.eye(design.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        self.coefficients_ = np.linalg.solve(
            weighted_design.T @ weighted_design + penalty,
            weighted_design.T @ weighted_y,
        )
        linear = np.abs(self.coefficients_[1:6])
        self.feature_importances_ = linear / linear.sum() if linear.sum() else np.ones(5) / 5.0
        return self

    def predict(self, X):
        return self._design(X) @ self.coefficients_

from tool_physics import calculate_tool_life

ENSEMBLE_WEIGHTS = {"gbr": 0.5, "hgb": 0.3, "rf": 0.2}
FEATURE_NAMES = (
    "log10 cutting speed",
    "log10 feed",
    "log10 depth of cut",
    "overhang L/D",
    "roughing",
)
_NEUTRAL_OVERHANG = 3.0
_LIFE_FLOOR = 0.5
_LIFE_CEILING = 2000.0
_BAND_COVERAGE = 0.80
_MIN_ANCHORS_FOR_BIAS = 3


class MLModelParams(NamedTuple):
    """Hashable snapshot of every physics input the trainer depends on."""

    taylor_C: float
    taylor_n: float
    taylor_x: float
    taylor_y: float
    tool_material: str
    v_min: float
    v_max: float
    f_min: float
    f_max: float
    ap_min: float
    ap_max: float
    rigidity: float
    coolant_name: str
    coolant_multiplier: float
    operation_name: str
    milling_tooling_name: str
    holder_name: str

    @classmethod
    def from_setup(
        cls,
        pairing,
        machine,
        coolant,
        operation_name: Optional[str] = None,
        milling_tooling_name: Optional[str] = None,
        holder_name: Optional[str] = None,
    ) -> "MLModelParams":
        return cls(
            taylor_C=float(pairing.taylor_C),
            taylor_n=float(pairing.taylor_n),
            taylor_x=float(pairing.taylor_x),
            taylor_y=float(pairing.taylor_y),
            tool_material=str(pairing.tool_material),
            v_min=float(pairing.v_min),
            v_max=float(pairing.v_max),
            f_min=float(pairing.f_min),
            f_max=float(pairing.f_max),
            ap_min=float(pairing.ap_min),
            ap_max=float(pairing.ap_max),
            rigidity=float(machine.rigidity_factor),
            coolant_name=str(coolant.name),
            coolant_multiplier=float(coolant.life_multiplier),
            operation_name=operation_name or "",
            milling_tooling_name=milling_tooling_name or "",
            holder_name=holder_name or "",
        )


@dataclass(frozen=True)
class MLLifeEstimate:
    life_min: float
    band_low_min: float
    band_high_min: float


@dataclass(frozen=True)
class MLToolLifeModel:
    params: MLModelParams
    gbr: object
    hgb: object
    forest: object
    n_sim: int
    n_measured: int
    n_train: int
    test_r_squared: float
    test_rmse_log: float
    measured_median_abs_pct_error: Optional[float]
    band_halfwidth_log: float
    bias_log10: float
    feature_importances: Tuple[Tuple[str, float], ...]

    @property
    def test_rmse_pct(self) -> float:
        return (10.0 ** self.test_rmse_log - 1.0) * 100.0


def _physics_stubs(params: MLModelParams):
    pairing = SimpleNamespace(
        taylor_C=params.taylor_C,
        taylor_n=params.taylor_n,
        taylor_x=params.taylor_x,
        taylor_y=params.taylor_y,
        tool_material=params.tool_material,
    )
    machine = SimpleNamespace(rigidity_factor=params.rigidity)
    coolant = SimpleNamespace(
        name=params.coolant_name,
        life_multiplier=params.coolant_multiplier,
    )
    return pairing, machine, coolant


def _safe_range(lo: float, hi: float, lo_factor: float, hi_factor: float, floor: float):
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        hi = lo * 1.05
    return max(floor, lo * lo_factor), hi * hi_factor


def _lhs(rng: np.random.Generator, n: int, lo: float, hi: float, log_space: bool = True):
    """Latin-hypercube sample: one draw per stratum, jittered and shuffled."""
    u = (np.arange(n) + rng.random(n)) / n
    rng.shuffle(u)
    if log_space:
        return 10.0 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo)))
    return lo + u * (hi - lo)


def _feature_matrix(vc, feed, ap, overhang, is_roughing) -> np.ndarray:
    vc = np.asarray(vc, dtype=float)
    feed = np.asarray(feed, dtype=float)
    ap = np.asarray(ap, dtype=float)
    roughing = np.asarray(is_roughing, dtype=float)
    if vc.ndim > 0:
        if feed.ndim == 0:
            feed = np.full(vc.shape, float(feed))
        if ap.ndim == 0:
            ap = np.full(vc.shape, float(ap))
    if overhang is None or (isinstance(overhang, float) and np.isnan(overhang)):
        if vc.ndim:
            over = np.full(vc.shape, _NEUTRAL_OVERHANG)
        else:
            over = np.array(_NEUTRAL_OVERHANG)
    else:
        over = np.asarray(overhang, dtype=float)
        if over.ndim == 0 and vc.ndim > 0:
            over = np.full(vc.shape, float(over))
        over = np.where(np.isnan(over), _NEUTRAL_OVERHANG, over)
    if roughing.ndim == 0 and vc.ndim > 0:
        roughing = np.full(vc.shape, float(roughing))
    return np.column_stack(
        (
            np.log10(np.clip(vc, 1e-6, None)),
            np.log10(np.clip(feed, 1e-6, None)),
            np.log10(np.clip(ap, 1e-6, None)),
            over,
            roughing,
        )
    )


def _measured_rows(measured: Sequence[Sequence[Optional[float]]]) -> list:
    rows = []
    for row in measured or ():
        if row is None or len(row) < 6:
            continue
        vc, feed, ap, overhang, roughing, life = row[:6]
        try:
            vc = float(vc)
            feed = float(feed)
            ap = float(ap)
            life = float(life)
        except (TypeError, ValueError):
            continue
        if min(vc, feed, ap, life) <= 0 or not all(np.isfinite([vc, feed, ap, life])):
            continue
        if overhang is None or (isinstance(overhang, float) and np.isnan(overhang)):
            overhang = _NEUTRAL_OVERHANG
        else:
            overhang = float(overhang)
        roughing = 1.0 if roughing else 0.0
        rows.append((vc, feed, ap, overhang, roughing, life))
    return rows


def measured_tuples_from_trials(trials: pd.DataFrame) -> Tuple[Tuple[Optional[float], ...], ...]:
    """Convert matched calibration trials into hashable anchor tuples."""
    rows = []
    for row in trials.itertuples(index=False):
        overhang = getattr(row, "overhang_ratio", None)
        if overhang is None or (isinstance(overhang, float) and np.isnan(overhang)):
            overhang = None
        else:
            overhang = float(overhang)
        roughing = getattr(row, "is_roughing", False)
        if roughing is None or (isinstance(roughing, float) and np.isnan(roughing)):
            roughing = False
        rows.append(
            (
                float(row.cutting_speed_m_min),
                float(row.feed_rate_mm),
                float(row.depth_of_cut_mm),
                overhang,
                bool(roughing),
                float(row.measured_tool_life_min),
            )
        )
    return tuple(rows)


def _ensemble_predict(model: "MLToolLifeModel", X: np.ndarray) -> np.ndarray:
    return (
        ENSEMBLE_WEIGHTS["gbr"] * model.gbr.predict(X)
        + ENSEMBLE_WEIGHTS["hgb"] * model.hgb.predict(X)
        + ENSEMBLE_WEIGHTS["rf"] * model.forest.predict(X)
    )


def _sample_training_grid(rng: np.random.Generator, params: MLModelParams, n_sim: int):
    """LHS across speed/feed/depth, stratified overhang bands, half roughing."""
    vc_lo, vc_hi = _safe_range(params.v_min, params.v_max, 0.95, 1.05, 1.0)
    feed_lo, feed_hi = _safe_range(params.f_min, params.f_max, 0.95, 1.05, 0.002)
    ap_lo, ap_hi = _safe_range(params.ap_min, params.ap_max, 0.95, 1.05, 0.06)

    vc = _lhs(rng, n_sim, vc_lo, vc_hi)
    feed = _lhs(rng, n_sim, feed_lo, feed_hi)
    ap = _lhs(rng, n_sim, ap_lo, ap_hi)

    # Overhang is piecewise in physics (neutral <= 3.0, linear penalty above).
    # Stratify into thirds so the kink at 3.0 is densely covered on both sides.
    n_third = n_sim // 3
    n_rest = n_sim - 2 * n_third
    overhang = np.concatenate(
        [
            rng.uniform(2.2, 3.0, n_third),
            rng.uniform(3.0, 4.2, n_third),
            rng.uniform(4.2, 6.0, n_rest),
        ]
    )
    rng.shuffle(overhang)

    roughing = np.concatenate([np.zeros(n_sim // 2), np.ones(n_sim - n_sim // 2)])
    rng.shuffle(roughing)
    return vc, feed, ap, overhang, roughing


def train_tool_life_model(
    params: MLModelParams,
    measured: Sequence[Sequence[Optional[float]]] = (),
    n_sim: int = 3000,
    test_fraction: float = 0.25,
    random_state: int = 42,
) -> MLToolLifeModel:
    """Train the hybrid ensemble on physics-simulated runs plus measured anchors."""
    rng = np.random.default_rng(random_state)
    pairing, machine, coolant = _physics_stubs(params)
    operation_name = params.operation_name or None
    milling_tooling_name = params.milling_tooling_name or None
    holder_name = params.holder_name or None

    n_sim = max(400, int(n_sim))
    vc, feed, ap, overhang, roughing = _sample_training_grid(rng, params, n_sim)

    sim_life = np.array(
        [
            calculate_tool_life(
                pairing,
                machine,
                coolant,
                float(v),
                float(f),
                float(a),
                bool(r),
                operation_name=operation_name,
                milling_tooling_name=milling_tooling_name,
                holder_name=holder_name,
                overhang_ratio=(float(o) if o > 3.0 else None),
            )
            for v, f, a, o, r in zip(vc, feed, ap, overhang, roughing)
        ],
        dtype=float,
    )

    X_sim = _feature_matrix(vc, feed, ap, overhang, roughing)
    y_sim = np.log10(sim_life)

    measured_rows = _measured_rows(measured)
    if measured_rows:
        m_vc = np.array([r[0] for r in measured_rows])
        m_feed = np.array([r[1] for r in measured_rows])
        m_ap = np.array([r[2] for r in measured_rows])
        m_over = np.array([r[3] for r in measured_rows])
        m_rough = np.array([r[4] for r in measured_rows])
        m_life = np.array([r[5] for r in measured_rows])
        X_meas = _feature_matrix(m_vc, m_feed, m_ap, m_over, m_rough)
        y_meas = np.log10(m_life)
    else:
        X_meas = np.empty((0, X_sim.shape[1]))
        y_meas = np.empty(0)

    perm = rng.permutation(n_sim)
    n_test = max(1, int(n_sim * min(max(test_fraction, 0.1), 0.4)))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    # Measured anchors stay out of the training set by design: the bias
    # residuals below must be out-of-sample or the correction collapses to ~0.
    X_train = X_sim[train_idx]
    y_train = y_sim[train_idx]
    n_measured = len(y_meas)

    if SKLEARN_AVAILABLE:
        gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.04, max_depth=3, min_samples_leaf=4, subsample=0.8, random_state=random_state)
        hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False, random_state=random_state)
        forest = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.9, n_jobs=-1, random_state=random_state)
    else:
        # Application Control may block sklearn's compiled extensions. Keep an ML
        # estimator available rather than silently degrading to physics-only.
        gbr, hgb, forest = NumpyRidgeRegressor(1e-5), NumpyRidgeRegressor(1e-4), NumpyRidgeRegressor(1e-3)

    gbr.fit(X_train, y_train)
    hgb.fit(X_train, y_train)
    forest.fit(X_train, y_train)

    X_test = X_sim[test_idx]
    y_test = y_sim[test_idx]
    holdout_pred = (
        ENSEMBLE_WEIGHTS["gbr"] * gbr.predict(X_test)
        + ENSEMBLE_WEIGHTS["hgb"] * hgb.predict(X_test)
        + ENSEMBLE_WEIGHTS["rf"] * forest.predict(X_test)
    )
    r_squared = float(r2_score(y_test, holdout_pred)) if r2_score else float(
        1.0 - np.sum((y_test - holdout_pred) ** 2) / max(np.sum((y_test - y_test.mean()) ** 2), 1e-12)
    )
    rmse_log = float(np.sqrt(np.mean((y_test - holdout_pred) ** 2)))

    # Conformal half-width: the |residual| quantile on held-out simulated runs.
    holdout_abs_resid = np.abs(y_test - holdout_pred)
    conformal_hw = float(
        np.quantile(holdout_abs_resid, _BAND_COVERAGE)
        if len(holdout_abs_resid)
        else 0.10
    )

    # Measured anchoring: median bias correction + band widening from spread.
    bias_log10 = 0.0
    measured_median_abs_pct = None
    measured_spread = 0.0
    if n_measured:
        meas_pred = (
            ENSEMBLE_WEIGHTS["gbr"] * gbr.predict(X_meas)
            + ENSEMBLE_WEIGHTS["hgb"] * hgb.predict(X_meas)
            + ENSEMBLE_WEIGHTS["rf"] * forest.predict(X_meas)
        )
        meas_resid = y_meas - meas_pred
        corrected = meas_pred + float(np.median(meas_resid))
        pct = np.abs(10.0 ** corrected - 10.0 ** y_meas) / 10.0 ** y_meas * 100.0
        measured_median_abs_pct = float(np.median(pct))
        if n_measured >= _MIN_ANCHORS_FOR_BIAS:
            bias_log10 = float(np.median(meas_resid))
            measured_spread = float(np.std(meas_resid))

    band_halfwidth_log = float(max(conformal_hw, measured_spread, 0.02))

    imp_parts = []
    imp_weight = 0.0
    for _w, _member in (
        (ENSEMBLE_WEIGHTS["gbr"], gbr),
        (ENSEMBLE_WEIGHTS["hgb"], hgb),
        (ENSEMBLE_WEIGHTS["rf"], forest),
    ):
        _member_imp = getattr(_member, "feature_importances_", None)
        if _member_imp is None:
            continue  # e.g. HGB >= sklearn 1.9 dropped impurity-based importances
        imp_parts.append(_w * np.asarray(_member_imp, dtype=float))
        imp_weight += _w
    if imp_parts and imp_weight > 0:
        importances = sum(imp_parts) / imp_weight
    else:
        importances = np.zeros(len(FEATURE_NAMES), dtype=float)
    total = importances.sum()
    if total > 0:
        importances = importances / total
    ranked = sorted(zip(FEATURE_NAMES, importances.tolist()), key=lambda kv: kv[1], reverse=True)

    probe = MLToolLifeModel(
        params=params,
        gbr=gbr,
        hgb=hgb,
        forest=forest,
        n_sim=n_sim,
        n_measured=n_measured,
        n_train=len(y_train),
        test_r_squared=r_squared,
        test_rmse_log=rmse_log,
        measured_median_abs_pct_error=measured_median_abs_pct,
        band_halfwidth_log=band_halfwidth_log,
        bias_log10=bias_log10,
        feature_importances=tuple((str(name), float(value)) for name, value in ranked),
    )
    return probe


def predict_lives(model: MLToolLifeModel, vc, feed, ap, overhang=None, is_roughing=False):
    """Predict ensemble life and the uncertainty band. Scalars in, scalars out; arrays in, arrays out."""
    X = _feature_matrix(vc, feed, ap, overhang, is_roughing)
    ensemble = _ensemble_predict(model, X) + model.bias_log10
    low = ensemble - model.band_halfwidth_log
    high = ensemble + model.band_halfwidth_log
    life = np.clip(10.0 ** ensemble, _LIFE_FLOOR, _LIFE_CEILING)
    band_low = np.clip(10.0 ** low, _LIFE_FLOOR, _LIFE_CEILING)
    band_high = np.clip(10.0 ** high, _LIFE_FLOOR, _LIFE_CEILING)
    if np.ndim(vc) == 0 and np.ndim(feed) == 0 and np.ndim(ap) == 0:
        return float(np.asarray(life).item()), float(np.asarray(band_low).item()), float(np.asarray(band_high).item())
    return life, band_low, band_high


def predict_life(
    model: MLToolLifeModel,
    vc: float,
    feed: float,
    ap: float,
    overhang: Optional[float] = None,
    is_roughing: bool = False,
) -> MLLifeEstimate:
    life, low, high = predict_lives(model, vc, feed, ap, overhang, is_roughing)
    return MLLifeEstimate(life_min=life, band_low_min=low, band_high_min=high)


def blend_lives(physics_life: float, ml_life: float, ml_weight: float) -> float:
    """Log-space weighted blend; ml_weight 0 keeps physics, 1 takes ML fully."""
    weight = float(np.clip(float(ml_weight), 0.0, 1.0))
    log_blend = weight * np.log10(max(float(ml_life), _LIFE_FLOOR)) + (
        1.0 - weight
    ) * np.log10(max(float(physics_life), _LIFE_FLOOR))
    return float(np.clip(10.0 ** log_blend, _LIFE_FLOOR, _LIFE_CEILING))


if __name__ == "__main__":
    from springer_database import (
        COOLANT_DATABASE,
        MACHINE_DATABASE,
        SPRINGER_DATABASE,
    )

    pairing = SPRINGER_DATABASE[next(iter(SPRINGER_DATABASE))]
    machine = MACHINE_DATABASE[next(iter(MACHINE_DATABASE))]
    coolant = COOLANT_DATABASE[next(iter(COOLANT_DATABASE))]
    test_params = MLModelParams.from_setup(pairing, machine, coolant)

    mid_vc = 0.5 * (pairing.v_min + pairing.v_max)
    mid_feed = 0.5 * (pairing.f_min + pairing.f_max)
    mid_ap = 0.5 * (pairing.ap_min + pairing.ap_max)
    physics_mid = calculate_tool_life(
        pairing, machine, coolant, mid_vc, mid_feed, mid_ap, True, overhang_ratio=3.5
    )

    # Case 1: physics-only training (no measured anchors).
    model_plain = train_tool_life_model(test_params, n_sim=3000)
    life, low, high = predict_lives(
        model_plain, mid_vc, mid_feed, mid_ap, overhang=3.5, is_roughing=True
    )
    print(f"[plain] holdout R^2 vs physics: {model_plain.test_r_squared:.4f}")
    print(f"[plain] holdout RMSE: ±{model_plain.test_rmse_pct:.1f}%")
    print(f"[plain] ML {life:.1f} min (band {low:.1f}-{high:.1f}) vs physics {physics_mid:.1f} min")
    assert low <= life <= high
    assert model_plain.test_r_squared > 0.95
    assert model_plain.bias_log10 == 0.0
    assert abs(life - physics_mid) / physics_mid < 0.15

    # Case 2: anchored to measured trials running 10% short of the physics model.
    rng = np.random.default_rng(7)
    fake_measured = []
    for _ in range(10):
        v = float(rng.uniform(pairing.v_min, pairing.v_max))
        f = float(rng.uniform(pairing.f_min, pairing.f_max))
        a = float(rng.uniform(pairing.ap_min, pairing.ap_max))
        life_p = calculate_tool_life(
            pairing, machine, coolant, v, f, a, False, overhang_ratio=None
        )
        fake_measured.append((v, f, a, None, False, 0.9 * life_p))
    model_anchored = train_tool_life_model(test_params, measured=tuple(fake_measured), n_sim=3000)
    life_a, low_a, high_a = predict_lives(model_anchored, mid_vc, mid_feed, mid_ap)
    print(f"[anchored] bias correction: x{10 ** model_anchored.bias_log10:.3f}")
    print(f"[anchored] measured median error: {model_anchored.measured_median_abs_pct_error:.1f}%")
    print(f"[anchored] ML {life_a:.1f} min vs physics {physics_mid:.1f} min "
          f"-> blended(0.7) {blend_lives(physics_mid, life_a, 0.7):.1f} min")
    assert model_anchored.bias_log10 < -0.01
    assert model_anchored.measured_median_abs_pct_error < 15.0
    assert life_a < physics_mid

    for name, importance in model_plain.feature_importances:
        print(f"  {name}: {importance:.3f}")
    print("ml_model self-test OK")
