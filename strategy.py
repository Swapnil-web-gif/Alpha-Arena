"""
AGENTS WAR - Alpha Arena (Meraz 7.0) Trading Agent
Team: AlphaForge
AI-assisted: Yes (Google Antigravity / Claude)

Architecture: Core-Satellite with Turnover Minimization
  - CORE (65%): Risk-parity (inverse-vol) for stable diversification
  - SATELLITE (35%): ML alpha tilt (RidgeCV on cross-sectional rank features)
  - Regime filter: market trend SMA to switch bull/bear exposure
  - Vol targeting: constant annualized portfolio vol ~13%
  - Turnover control: EMA smoothing + deadband filter
  - Drawdown protection: dynamic de-leveraging on portfolio drawdown
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

TEAM_NAME = "AlphaForge"
TRAIN_WINDOW = (1188, 3707)      # days 1188-3707 (most recent 10 years: recency to secret period)
MODEL = {}

# ─────────────────────────── HELPERS ───────────────────────────

def _compute_features(close_arr):
    """Compute stationary cross-sectional features from close prices array.
    close_arr: 2D numpy array, shape (T, N_stocks), last row = today.
    Returns: features (N_stocks, n_features), vol21 (N_stocks,)
    """
    T, N = close_arr.shape
    pct = close_arr[1:] / close_arr[:-1] - 1.0

    # Multi-horizon returns
    r5 = close_arr[-1] / close_arr[-6] - 1.0 if T >= 6 else np.zeros(N)
    r10 = close_arr[-1] / close_arr[-11] - 1.0 if T >= 11 else np.zeros(N)
    r21 = close_arr[-1] / close_arr[-22] - 1.0 if T >= 22 else np.zeros(N)
    r63 = close_arr[-1] / close_arr[-64] - 1.0 if T >= 64 else np.zeros(N)
    r126 = close_arr[-1] / close_arr[-127] - 1.0 if T >= 127 else np.zeros(N)

    # Momentum minus reversal (skip recent 5 days)
    mom_residual = r63 - r5

    # Volatilities
    vol21 = np.std(pct[-21:], axis=0, ddof=1) * np.sqrt(252) + 1e-8 if T > 22 else np.ones(N) * 0.25
    vol63 = np.std(pct[-63:], axis=0, ddof=1) * np.sqrt(252) + 1e-8 if T > 64 else vol21

    # Risk-adjusted momentum (Sharpe-style)
    sharpe63 = r63 / vol63

    # Mean-reversion z-score (normalized distance from SMA20)
    if T >= 21:
        sma20 = np.mean(close_arr[-20:], axis=0)
        std20 = np.std(close_arr[-20:], axis=0, ddof=1) + 1e-8
        z20 = (close_arr[-1] - sma20) / std20
    else:
        z20 = np.zeros(N)

    # RSI-like feature (5-day)
    if T >= 7:
        delta = np.diff(close_arr[-6:], axis=0)  # 5 deltas
        gains = np.maximum(delta, 0).mean(axis=0)
        losses = np.maximum(-delta, 0).mean(axis=0) + 1e-8
        rsi5 = gains / (gains + losses) - 0.5  # centered at 0
    else:
        rsi5 = np.zeros(N)

    # Stack features: (N_stocks, n_features)
    features = np.column_stack([
        r5, r10, r21, r63, r126,
        mom_residual, sharpe63, z20, rsi5, vol21
    ])

    # Cross-sectional standardization (rank normalize across stocks)
    for j in range(features.shape[1]):
        col = features[:, j]
        mu, sigma = col.mean(), col.std() + 1e-8
        features[:, j] = (col - mu) / sigma

    return features, vol21


def _rsi(prices, period=14):
    """RSI for pandas DataFrame."""
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─────────────────────────── TRAIN ───────────────────────────

def train(data):
    """Called once with TRAIN_WINDOW data. Fits RidgeCV model + learns parameters."""
    close = data["close"]
    stocks = list(close.columns)
    n_stocks = len(stocks)
    MODEL["stocks"] = stocks

    close_arr = close.values  # (T, 12)
    T = close_arr.shape[0]

    # ── 1. Fit RidgeCV on cross-sectional rank prediction ──
    X_list, y_list = [], []
    # Step by 5 days to reduce overlap / serial correlation
    for t in range(252, T - 5, 5):
        win = close_arr[max(0, t - 252):t]
        if len(win) < 130:
            continue
        try:
            feats, _ = _compute_features(win)
            # Target: forward 5-day return, rank-normalized to [-1, 1]
            fwd_ret = close_arr[t + 4] / close_arr[t - 1] - 1.0
            order = fwd_ret.argsort().argsort().astype(float)
            fwd_rank = (order - (n_stocks - 1) / 2.0) / ((n_stocks - 1) / 2.0)
            X_list.append(feats)
            y_list.append(fwd_rank)
        except Exception:
            continue

    if len(X_list) > 10:
        X_train = np.concatenate(X_list, axis=0)
        y_train = np.concatenate(y_list, axis=0)
        # Clean NaN/Inf
        mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        X_train, y_train = X_train[mask], y_train[mask]
        if len(X_train) > 20:
            alphas = np.logspace(-1, 4, 20)
            MODEL["ridge"] = RidgeCV(alphas=alphas).fit(X_train, y_train)
            print(f"  Ridge fitted: alpha={MODEL['ridge'].alpha_:.1f}, "
                  f"R²={MODEL['ridge'].score(X_train, y_train):.3f}, "
                  f"samples={len(X_train)}")

    # ── 2. Learn per-stock volatilities ──
    daily_ret = close.pct_change().dropna()
    stock_vol = (daily_ret.std() * np.sqrt(252)).to_dict()
    MODEL["stock_vol"] = stock_vol

    # ── 3. Volatility regime thresholds ──
    market_ret = daily_ret.mean(axis=1)
    market_vol = market_ret.rolling(21).std() * np.sqrt(252)
    MODEL["vol_median"] = float(market_vol.median())
    MODEL["vol_90"] = float(market_vol.quantile(0.90))
    MODEL["vol_95"] = float(market_vol.quantile(0.95))

    # ── 4. Learn EMA alpha via grid search ──
    next_day = daily_ret.shift(-1)
    cutoff = int(T * 0.3)
    best_sh, best_alpha = -np.inf, 0.12

    for alpha in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        # Simulate with EMA smoothing on risk-parity
        inv_vol = 1.0 / daily_ret.rolling(60).std()
        rp = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(1/n_stocks)
        # Smooth
        smooth = rp.iloc[0:1].copy()
        rows = [rp.iloc[0].values]
        for i in range(1, len(rp)):
            new_row = alpha * rp.iloc[i].values + (1 - alpha) * rows[-1]
            rows.append(new_row)
        smooth_df = pd.DataFrame(rows, index=rp.index, columns=rp.columns)
        # Cost
        turnover = smooth_df.diff().abs().sum(axis=1)
        cost = 0.001 * turnover
        ret = (daily_ret * smooth_df.shift(1)).sum(axis=1) - cost
        ret = ret.iloc[cutoff:].dropna()
        sh = ret.mean() / ret.std() * np.sqrt(252) if len(ret) > 30 and ret.std() > 0 else 0
        if sh > best_sh:
            best_sh, best_alpha = sh, alpha

    MODEL["ema_alpha"] = best_alpha
    MODEL["deadband"] = 0.015  # 1.5% minimum weight change to trigger trade

    # ── 5. Target volatility (annualized) ──
    MODEL["target_vol"] = 0.13  # 13%

    # ── 6. Learn optimal MR lookback (for pure MR strategy) ──
    best_mr_sh, best_mr_lb = -np.inf, 5
    for lb in [3, 5, 7, 10]:
        past = close.pct_change(lb)
        vol = daily_ret.rolling(max(lb, 20)).std()
        z = past / (vol * np.sqrt(lb)).replace(0, np.nan)
        sig = (-z).rank(axis=1, pct=True) - 0.5
        ret = (sig * next_day).sum(axis=1).dropna()
        sh = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
        if sh > best_mr_sh:
            best_mr_sh, best_mr_lb = sh, lb
    MODEL["mr_lookback"] = best_mr_lb

    print(f"  EMA alpha={MODEL['ema_alpha']:.2f}, deadband={MODEL['deadband']:.3f}")
    print(f"  MR lookback={MODEL['mr_lookback']}, target_vol={MODEL['target_vol']:.2f}")
    print(f"  Vol median={MODEL['vol_median']:.4f}, vol_90={MODEL['vol_90']:.4f}")


# ─────────────────────────── STRATEGIES ───────────────────────────


def decide_momentum(history):
    """Pure cross-sectional momentum: buy top-3 performers over 20 days."""
    close = history["close"]
    if len(close) < 21:
        return {}
    past_return = close.iloc[-1] / close.iloc[-21] - 1
    winners = past_return.nlargest(3).index
    return {stock: 1 / 3 for stock in winners}


def decide_risk_parity(history):
    """Pure inverse-volatility weighted portfolio, rebalanced every ~20 days."""
    close = history["close"]
    stocks = list(close.columns)
    n = len(stocks)
    if len(close) < 63:
        return {s: 1/n for s in stocks}

    # Rebalance infrequently
    if not hasattr(decide_risk_parity, '_cnt'):
        decide_risk_parity._cnt = 0
        decide_risk_parity._prev = None
    decide_risk_parity._cnt += 1
    if decide_risk_parity._prev is not None and decide_risk_parity._cnt % 20 != 1:
        return decide_risk_parity._prev

    daily_ret = close.pct_change().dropna()
    vol = daily_ret.iloc[-60:].std()
    inv_vol = 1.0 / vol.replace(0, np.nan).fillna(0)
    total = inv_vol.sum()
    if total == 0:
        return {s: 1/n for s in stocks}
    weights = inv_vol / total * 0.92
    result = {s: float(weights[s]) for s in stocks}
    decide_risk_parity._prev = result
    return result


def decide_mean_reversion(history):
    """Cross-sectional mean reversion with z-score ranking."""
    close = history["close"]
    stocks = list(close.columns)
    n = len(stocks)
    lb = MODEL.get("mr_lookback", 5)
    if len(close) < max(lb + 20, 40):
        return {s: 1/n for s in stocks}

    daily_ret = close.pct_change().dropna()
    past = close.iloc[-1] / close.iloc[-lb-1] - 1
    vol = daily_ret.iloc[-20:].std()
    z = past / (vol * np.sqrt(lb)).replace(0, np.nan)
    z = z.fillna(0)
    ranks = z.rank(pct=True) - 0.5
    signal = -ranks  # buy losers, sell winners
    weights = signal / signal.abs().sum() * 0.85 if signal.abs().sum() > 0 else signal
    return {s: float(weights[s]) for s in stocks}


def decide_multifactor(history):
    """Risk-parity base + mean-reversion alpha overlay. No ML."""
    close = history["close"]
    stocks = list(close.columns)
    n = len(stocks)
    mr_lb = MODEL.get("mr_lookback", 5)
    if len(close) < 63:
        return {s: 1/n for s in stocks}

    daily_ret = close.pct_change().dropna()

    # Risk-parity base
    vol = daily_ret.iloc[-60:].std()
    inv_vol = 1.0 / vol.replace(0, np.nan).fillna(0)
    total = inv_vol.sum()
    rp = inv_vol / total if total > 0 else pd.Series(1/n, index=stocks)

    # Mean-reversion alpha
    past_mr = close.iloc[-1] / close.iloc[-mr_lb-1] - 1
    vol_mr = daily_ret.iloc[-20:].std()
    z = past_mr / (vol_mr * np.sqrt(mr_lb)).replace(0, np.nan)
    sig_mr = -(z.rank(pct=True) - 0.5).fillna(0)

    # RSI alpha
    r = _rsi(close, 10)
    if len(r) > 10 and r.iloc[-1].notna().all():
        sig_rsi = -(r.iloc[-1].rank(pct=True) - 0.5)
    else:
        sig_rsi = pd.Series(0, index=stocks)

    alpha = 0.6 * sig_mr + 0.4 * sig_rsi
    final = 0.65 * rp + 0.35 * alpha
    gross = final.abs().sum()
    if gross > 0.92:
        final = final / gross * 0.92
    return {s: float(final[s]) for s in stocks}


def decide_adaptive_hybrid(history):
    """FINAL AGENT: Core-Satellite with ML alpha, regime detection,
    volatility targeting, EMA smoothing + deadband for minimal turnover.

    Core (65%): Inverse-vol risk parity
    Satellite (35%): RidgeCV ML alpha tilt on cross-sectional rank features
    + Regime filter: market SMA trend → bull/bear exposure
    + Vol targeting: constant ~13% annualized portfolio vol
    + Turnover control: EMA weight smoothing + 1.5% deadband
    + Drawdown protection: dynamic de-leverage on drawdowns
    """
    close = history["close"]
    stocks = list(close.columns)
    n = len(stocks)

    # Initialize persistent state
    if not hasattr(decide_adaptive_hybrid, '_prev_w'):
        decide_adaptive_hybrid._prev_w = None
        decide_adaptive_hybrid._port_returns = []

    if len(close) < 63:
        w = {s: 1/n for s in stocks}
        decide_adaptive_hybrid._prev_w = np.array([w[s] for s in stocks])
        return w

    close_arr = close.values
    daily_ret = close.pct_change().dropna()
    T = close_arr.shape[0]

    # ── 1. Risk-parity base weights ──
    vol60 = daily_ret.iloc[-60:].std().values
    inv_vol = 1.0 / (vol60 + 1e-8)
    rp_w = inv_vol / inv_vol.sum()

    # ── 2. ML alpha tilt ──
    alpha_tilt = np.zeros(n)
    if "ridge" in MODEL and T >= 130:
        try:
            feats, _ = _compute_features(close_arr)
            scores = MODEL["ridge"].predict(feats)
            # Center and scale tilt
            scores_centered = scores - scores.mean()
            max_abs = np.abs(scores_centered).max() + 1e-8
            alpha_tilt = 0.35 * (scores_centered / max_abs)
        except Exception:
            alpha_tilt = np.zeros(n)

    # ── 3. Combine: Core (65%) + Satellite (35%) ──
    raw_w = 0.65 * rp_w + 0.35 * alpha_tilt

    # ── 4. Market regime filter ──
    mkt_index = close_arr.mean(axis=1)
    if T >= 50:
        sma50 = mkt_index[-50:].mean()
        bull = mkt_index[-1] > sma50
    else:
        bull = True

    # ── 5. Volatility targeting ──
    target_vol = MODEL.get("target_vol", 0.13)
    port_ret = daily_ret.mean(axis=1)  # proxy: equal-weighted
    if len(port_ret) >= 21:
        realized_vol = float(port_ret.iloc[-21:].std() * np.sqrt(252))
        vol_scale = min(1.0, target_vol / max(realized_vol, 0.04))
    else:
        vol_scale = 0.90

    if bull:
        # Bull: full exposure, scaled by vol
        raw_w = raw_w * vol_scale
    else:
        # Bear: reduce to 40-60% exposure, favor relative strength
        raw_w = raw_w * vol_scale * 0.55

    # ── 6. Drawdown protection ──
    if T >= 60:
        eq_equity = (1 + port_ret).cumprod()
        eq_peak = eq_equity.cummax()
        dd = float(eq_equity.iloc[-1] / eq_peak.iloc[-1] - 1)
        if dd < -0.20:
            raw_w *= 0.40
        elif dd < -0.12:
            raw_w *= 0.65
        elif dd < -0.08:
            raw_w *= 0.85

    # ── 7. EMA smoothing + deadband ──
    ema_alpha = MODEL.get("ema_alpha", 0.12)
    deadband = MODEL.get("deadband", 0.015)

    if decide_adaptive_hybrid._prev_w is not None:
        prev = decide_adaptive_hybrid._prev_w

        # Account for price drift of previous weights
        if T >= 2:
            last_ret = close_arr[-1] / close_arr[-2] - 1.0
            drift_denom = 1.0 + np.dot(prev, last_ret)
            if drift_denom > 0:
                drifted = prev * (1.0 + last_ret) / drift_denom
            else:
                drifted = prev.copy()
        else:
            drifted = prev.copy()

        # EMA smoothing towards target
        smoothed = (1 - ema_alpha) * drifted + ema_alpha * raw_w

        # Deadband: only trade where change exceeds threshold
        delta = np.abs(smoothed - drifted)
        final_w = np.where(delta > deadband, smoothed, drifted)
    else:
        final_w = raw_w

    # ── 8. Final constraints ──
    final_w = np.clip(final_w, -0.30, 0.30)  # max 30% per stock
    gross = np.abs(final_w).sum()
    if gross > 0.95:
        final_w = final_w * (0.95 / gross)
    # Ensure minimum 30% invested (anti-cash rule)
    if gross < 0.30 and gross > 0:
        final_w = final_w * (0.35 / gross)

    decide_adaptive_hybrid._prev_w = final_w.copy()
    return {s: float(final_w[i]) for i, s in enumerate(stocks)}


def decide_conservative_rp(history):
    """Conservative risk-parity with trend filter. Only hold uptrending stocks."""
    close = history["close"]
    stocks = list(close.columns)
    n = len(stocks)
    if len(close) < 120:
        return {s: 1/n for s in stocks}

    daily_ret = close.pct_change().dropna()
    vol = daily_ret.iloc[-60:].std()
    inv_vol = 1.0 / vol.replace(0, np.nan).fillna(0)
    total = inv_vol.sum()
    rp = inv_vol / total if total > 0 else pd.Series(1/n, index=stocks)

    ma100 = close.iloc[-100:].mean()
    in_uptrend = close.iloc[-1] > ma100
    weights = rp * in_uptrend.astype(float)
    total_w = weights.sum()
    if total_w > 0:
        weights = weights / total_w * 0.90
    else:
        weights = rp * 0.50
    return {s: float(weights[s]) for s in stocks}


def decide_pure_mr(history):
    """Pure mean-reversion long/short with slight long bias."""
    close = history["close"]
    stocks = list(close.columns)
    n = len(stocks)
    mr_lb = MODEL.get("mr_lookback", 5)
    if len(close) < 40:
        return {s: 1/n for s in stocks}

    daily_ret = close.pct_change().dropna()
    past_ret = close.iloc[-1] / close.iloc[-mr_lb-1] - 1
    vol_short = daily_ret.iloc[-20:].std()
    z = past_ret / (vol_short * np.sqrt(mr_lb)).replace(0, np.nan)
    sig = -(z.rank(pct=True) - 0.5).fillna(0)
    r = _rsi(close, 10)
    sig_rsi = -(r.iloc[-1].rank(pct=True) - 0.5) if r.iloc[-1].notna().all() else pd.Series(0, index=stocks)
    signal = 0.6 * sig + 0.4 * sig_rsi
    signal = signal - signal.mean() + 0.02  # small long bias
    gross = signal.abs().sum()
    if gross > 0:
        signal = signal / gross * 0.85
    return {s: float(signal[s]) for s in stocks}


# ─────────────────────────── FINAL AGENT ───────────────────────────

decide = decide_adaptive_hybrid   # <- the agent we submit


# ─────────────────────────── MAIN ───────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    import engine

    data = engine.load(Path(__file__).resolve().parent / "data" / "train.csv")

    # Reset state
    decide_adaptive_hybrid._prev_w = None
    decide_adaptive_hybrid._port_returns = []
    if hasattr(decide_risk_parity, '_cnt'):
        decide_risk_parity._cnt = 0
        decide_risk_parity._prev = None

    window = engine.train_agent(sys.modules[__name__], data)
    result = engine.backtest(decide, data, verbose=False)
    log = result["log"]
    inside = (log.index >= window[0]) & (log.index <= window[1])
    print(f"\n{TEAM_NAME}: trained on days {window[0]}-{window[1]}")
    print(f"Strategy: {decide.__name__}")
    print("\nInside the training window:")
    engine.print_metrics(engine.metrics(log[inside]))
    print("Outside the training window (validation):")
    engine.print_metrics(engine.metrics(log[~inside]))

    m_in = engine.metrics(log[inside])
    m_out = engine.metrics(log[~inside])
    print(f"\nGeneralization gap (val Sharpe - train Sharpe): {m_out['sharpe'] - m_in['sharpe']:.3f}")
    print(f"Total costs: {log['cost'].sum()*100:.2f}% of capital")
    print(f"Avg daily turnover: {log['cost'].mean()/0.001:.3f}")
