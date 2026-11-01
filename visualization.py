"""
visualization.py - Cumulative Portfolio Growth Graph
Alpha Arena (Meraz 7.0) - Team AlphaForge

Requirements:
  - X-axis: Day (data has no dates)
  - Y-axis: Portfolio value in Rs, starting from Rs 1,00,000
  - At most 10 years (2520 days) on one graph (we plot validation + recent period)
  - Equal-weight benchmark as a second line for direct comparison
  - Saves portfolio_growth.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import engine
import strategy

HERE = Path(__file__).resolve().parent


def plot_portfolio_growth():
    # 1. Load data
    data = engine.load(HERE / "data" / "train.csv")

    # 2. Reset persistent state in strategy
    strategy.decide_adaptive_hybrid._prev_w = None
    strategy.decide_adaptive_hybrid._port_returns = []

    # 3. Train agent on TRAIN_WINDOW
    window = engine.train_agent(strategy, data)
    print(f"Agent trained on days {window[0]}-{window[1]}")

    # 4. Run backtest for our strategy
    res_strat = engine.backtest(strategy.decide, data, verbose=False)
    equity_strat = res_strat["equity"]
    log_strat = res_strat["log"]

    # 5. Run backtest for equal-weight benchmark
    res_bm = engine.backtest(engine.equal_weight, data, verbose=False)
    equity_bm = res_bm["equity"]

    # 6. Select the 10-year window to display:
    # Rule: "At most 10 years (2520 days) on one graph"
    # We display the last 2520 days (days 1188 to 3707) which covers the training window,
    # OR we can show the validation period + part of training.
    # Let's show the final 2520 days (days 1188-3707), re-based to Rs 1,00,000 at day 1188.
    plot_start = 1188
    plot_end = 3707

    # Slice and normalize to Rs 1,00,000 at plot_start
    sub_equity_strat = equity_strat.loc[plot_start:plot_end]
    sub_equity_bm = equity_bm.loc[plot_start:plot_end]

    norm_strat = 100000 * (sub_equity_strat / sub_equity_strat.iloc[0])
    norm_bm = 100000 * (sub_equity_bm / sub_equity_bm.iloc[0])
    days = norm_strat.index

    # 7. Create professional figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # Top plot: Portfolio Value
    ax1.plot(days, norm_strat, label=f"AlphaForge ({strategy.decide.__name__})", color="#1f77b4", linewidth=2.0)
    ax1.plot(days, norm_bm, label="Benchmark (Equal Weight)", color="#ff7f0e", linewidth=1.5, linestyle="--")

    # Annotate final values
    final_strat = norm_strat.iloc[-1]
    final_bm = norm_bm.iloc[-1]
    ax1.annotate(f"Rs. {final_strat:,.0f} (+{(final_strat/100000 - 1)*100:.1f}%)",
                 xy=(days[-1], final_strat), xytext=(days[-1] - 300, final_strat * 1.05),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4"),
                 fontsize=10, fontweight="bold", color="#1f77b4")
    ax1.annotate(f"Rs. {final_bm:,.0f} (+{(final_bm/100000 - 1)*100:.1f}%)",
                 xy=(days[-1], final_bm), xytext=(days[-1] - 300, final_bm * 0.90),
                 arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
                 fontsize=10, fontweight="bold", color="#ff7f0e")

    ax1.set_title("Cumulative Portfolio Growth: AlphaForge vs Benchmark\n(10-Year Period: Days 1188 to 3707)",
                  fontsize=14, fontweight="bold", pad=15)
    ax1.set_ylabel("Portfolio Value (Rs.)", fontsize=11, fontweight="bold")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"Rs. {int(x):,}"))
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9, facecolor="white")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Bottom plot: Drawdown
    peak_strat = norm_strat.cummax()
    dd_strat = (norm_strat - peak_strat) / peak_strat * 100

    peak_bm = norm_bm.cummax()
    dd_bm = (norm_bm - peak_bm) / peak_bm * 100

    ax2.fill_between(days, dd_strat, 0, color="#1f77b4", alpha=0.3, label="AlphaForge DD")
    ax2.plot(days, dd_bm, color="#ff7f0e", linestyle="--", linewidth=1.0, label="Benchmark DD")
    ax2.set_ylabel("Drawdown (%)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Day", fontsize=11, fontweight="bold")
    ax2.set_ylim(-45, 2)
    ax2.legend(loc="lower left", frameon=True, fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = HERE / "portfolio_growth.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Graph saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    plot_portfolio_growth()
