"""
Compare your strategies:         python compare.py
Compare training windows:        python compare.py windows

Rule: your agent may learn from at most 10 years (2520 days) - ANY block inside the data,
set by TRAIN_WINDOW in strategy.py. train() gets only those days.

  * default mode: trains once on your TRAIN_WINDOW, then tests every decide_xxx function
  * windows mode: trains your final `decide` on 6 different 10-year windows
                  (days 0-2519, 252-2771, ... 1188-3707) so you can see which window works best

All numbers are measured OUTSIDE the training window (= validation), except fit_sharpe
(inside the window). generalization = validation Sharpe - fit Sharpe (closer to 0 or above is better).
Saves comparison.csv, or windows.csv - put them in your slides.
"""
import sys
from pathlib import Path

import pandas as pd

import engine
import strategy

HERE = Path(__file__).resolve().parent
SHOW = ["total_return_%", "sharpe", "sortino", "calmar", "max_drawdown_%", "trading_accuracy_%",
        "costs_paid_%", "avg_exposure_%", "fit_sharpe", "generalization"]


def split_metrics(log, window):
    inside = (log.index >= window[0]) & (log.index <= window[1])
    m = engine.metrics(log[~inside])
    m["fit_sharpe"] = engine.metrics(log[inside])["sharpe"]
    m["generalization"] = m["sharpe"] - m["fit_sharpe"]
    return m


def show(table, title, csv_name):
    pd.set_option("display.width", 200)
    print(f"\n{title}\n")
    print(table.round(2).to_string())
    table.round(3).to_csv(HERE / csv_name)
    print(f"\nSaved {csv_name}")


def compare_strategies(data):
    window = engine.train_agent(strategy, data)
    candidates = {n: f for n, f in vars(strategy).items() if n.startswith("decide_") and callable(f)}
    if strategy.decide not in candidates.values():
        candidates["decide"] = strategy.decide
    candidates["BENCHMARK equal weight"] = engine.equal_weight

    rows = {}
    for name, fn in candidates.items():
        print(f"running {name} ...")
        res = engine.backtest(fn, data, verbose=False)
        label = name + ("  <- submitted" if fn is strategy.decide else "")
        rows[label] = split_metrics(res["log"], window)

    table = pd.DataFrame(rows).T[SHOW].sort_values("sharpe", ascending=False)
    show(table, f"Trained on days {window[0]}-{window[1]}. VALIDATION results (all other days), "
                f"best Sharpe first:", "comparison.csv")


def compare_windows(data):
    days = data["close"].index
    last_start = days[-1] - engine.MAX_TRAIN_DAYS + 1
    starts = sorted(set(range(days[0], last_start, 252)) | {last_start})
    rows = {}
    for s in starts:
        window = (s, s + engine.MAX_TRAIN_DAYS - 1)
        print(f"training on days {window[0]}-{window[1]} ...")
        engine.train_agent(strategy, data, window=window)
        res = engine.backtest(strategy.decide, data, verbose=False)
        mine = tuple(strategy.TRAIN_WINDOW) == window
        rows[f"days {window[0]}-{window[1]}" + ("  <- your TRAIN_WINDOW" if mine else "")] = \
            split_metrics(res["log"], window)
    engine.train_agent(strategy, data)          # back to the agent's own window
    table = pd.DataFrame(rows).T[SHOW]
    show(table, f"`{strategy.decide.__name__}` trained on different 10-year windows. "
                f"VALIDATION results (days outside each window):", "windows.csv")


def main(mode="strategies"):
    data = engine.load(HERE / "data" / "train.csv")
    (compare_windows if mode == "windows" else compare_strategies)(data)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "strategies")
