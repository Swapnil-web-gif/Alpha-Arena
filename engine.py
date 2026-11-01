"""
AGENTS WAR - backtest engine.  DO NOT EDIT (organizers use the same file to score you).

How it works, in plain words:
  * First the engine calls your  train(data)  ONCE, with only the days in your TRAIN_WINDOW
    (at most 10 years = 2520 days). Learn whatever you need there.
  * Then, every day after the market closes, it calls your  decide(history).
  * `history` holds only the last year (252 days) up to and including today -
    you can never see the future, and you cannot re-learn from the full data inside decide.
  * You return how much of your money to put in each stock (a "weight").
  * Your portfolio earns those stocks' returns from today's close to tomorrow's close.
  * Every change in weights pays a trading cost.
"""
import math

import numpy as np
import pandas as pd

STARTING_CAPITAL = 100000     # Rs 1 lakh (virtual)
COST_PER_TRADE = 0.001         # 0.1% of the amount bought or sold
ALLOW_SHORT = True             # negative weights allowed?
MAX_GROSS = 1.0                # sum of |weights| may not exceed 1 (no leverage)
FIELDS = ["open", "high", "low", "close", "volume"]
LOOKBACK_DAYS = 252            # decide() sees at most the last year of prices
MAX_TRAIN_DAYS = 2520          # train() may learn from at most 10 years


def load(path):
    """Read a prices csv into {field: DataFrame(rows = day, columns = stocks)}."""
    df = pd.read_csv(path)
    return {f: df.pivot(index="day", columns="stock", values=f).sort_index() for f in FIELDS}


def clean_weights(raw, stocks):
    """Turn whatever decide() returned into legal weights."""
    raw = dict(raw) if raw is not None else {}
    w = np.array([float(raw.get(s, 0.0)) for s in stocks])
    w[~np.isfinite(w)] = 0.0
    w = np.clip(w, -1.0 if ALLOW_SHORT else 0.0, 1.0)
    gross = np.abs(w).sum()
    if gross > MAX_GROSS:
        w = w * (MAX_GROSS / gross)
    return w


def train_agent(module, data, window=None):
    """Check the agent's TRAIN_WINDOW (or the given window) and call its train() with only those days."""
    start, end = window or getattr(module, "TRAIN_WINDOW", (None, None))
    if start is None or end is None:
        raise ValueError("strategy.py must set TRAIN_WINDOW = (first_day, last_day)")
    days = data["close"].index
    if not (days[0] <= start <= end <= days[-1]):
        raise ValueError(f"TRAIN_WINDOW must be inside days {days[0]}..{days[-1]}, got ({start}, {end})")
    n = int(((days >= start) & (days <= end)).sum())
    if n > MAX_TRAIN_DAYS:
        raise ValueError(f"TRAIN_WINDOW covers {n} days; the limit is {MAX_TRAIN_DAYS} (10 years)")
    if hasattr(module, "train"):
        module.train({f: df.loc[start:end].copy() for f, df in data.items()})
    return start, end


def backtest(decide, data, start_day=50, on_error="raise", verbose=True):
    """
    Run `decide` day by day from `start_day` to the end of `data`.
    Returns a dict with daily returns, equity curve, weights and metrics.
    on_error="raise" stops on a bug (use while developing);
    on_error="hold" keeps yesterday's weights (used in the final evaluation).
    """
    close = data["close"]
    stocks = list(close.columns)
    days = list(close.index)
    start = days.index(start_day)
    fwd_ret = close.pct_change().shift(-1).fillna(0.0).to_numpy()   # today's close -> tomorrow's close

    w_prev = np.zeros(len(stocks))
    log, weights = [], []
    for i in range(start, len(days) - 1):
        lo = max(0, i + 1 - LOOKBACK_DAYS)
        history = {f: data[f].iloc[lo: i + 1] for f in FIELDS}
        error = 0
        try:
            w = clean_weights(decide(history), stocks)
        except Exception:
            if on_error == "raise":
                raise
            error = 1
            w = w_prev
        cost = COST_PER_TRADE * np.abs(w - w_prev).sum()
        held = w != 0
        hits = int((w[held] * fwd_ret[i][held] > 0).sum())   # positions that moved the right way
        log.append((float(w @ fwd_ret[i] - cost), cost, int(held.sum()), hits, float(np.abs(w).sum()), error))
        weights.append(w)
        w_prev = w

    # one row per day: the return earned that day and how it was earned
    log = pd.DataFrame(log, index=pd.Index(days[start + 1:], name="day"),
                       columns=["return", "cost", "positions", "hits", "exposure", "error"])
    weights = pd.DataFrame(weights, index=days[start:-1], columns=stocks)
    result = {
        "daily_returns": log["return"],
        "log": log,
        "equity": STARTING_CAPITAL * (1 + log["return"]).cumprod(),
        "weights": weights,
        "metrics": metrics(log),
        "errors": int(log["error"].sum()),
    }
    if verbose:
        print_metrics(result["metrics"])
    return result


def metrics(log):
    """The numbers you are judged on (see README). `log` is result["log"] or a slice of it."""
    daily = log["return"]
    equity = (1 + daily).cumprod()
    years = len(daily) / 252
    total = equity.iloc[-1] - 1
    annual = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    std = daily.std()
    downside = math.sqrt((daily.clip(upper=0) ** 2).mean())
    max_dd = float((equity / equity.cummax().clip(lower=1.0) - 1).min())
    positions = log["positions"].sum()
    return {
        "total_return_%": 100 * total,
        "annual_return_%": 100 * annual,
        "sharpe": float(daily.mean() / std * math.sqrt(252)) if std > 0 else 0.0,      # return per unit of risk
        "sortino": float(daily.mean() / downside * math.sqrt(252)) if downside > 0 else 0.0,  # ... of downside risk
        "calmar": annual / -max_dd if max_dd < 0 else 0.0,                             # yearly return / worst fall
        "max_drawdown_%": 100 * max_dd,
        "trading_accuracy_%": 100 * log["hits"].sum() / positions if positions else 0.0,
        "costs_paid_%": 100 * log["cost"].sum(),   # total trading costs as % of capital
        "avg_exposure_%": 100 * log["exposure"].mean(),   # how much of the money was invested, on average
        "avg_daily_turnover_%": 100 * log["cost"].mean() / COST_PER_TRADE,
    }


def print_metrics(m):
    print("-" * 40)
    for k, v in m.items():
        print(f"{k:>22}: {v:10.2f}")
    print("-" * 40)


def equal_weight(history):
    """Benchmark: put 1/N in every stock, every day."""
    stocks = history["close"].columns
    return {s: 1 / len(stocks) for s in stocks}
