"""
portfolio_optimizer.py
======================
Markowitz Mean-Variance Portfolio Optimization

Fetches real historical price data, computes the efficient frontier,
finds the maximum Sharpe ratio and minimum volatility portfolios,
and produces publication-quality charts.

Requirements:
    pip install numpy pandas scipy matplotlib yfinance

Usage:
    python portfolio_optimizer.py
    python portfolio_optimizer.py --tickers AAPL MSFT GOOGL TSLA --start 2020-01-01
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────
# DEFAULT CONFIGURATION
# ─────────────────────────────────────────────

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "BRK-B"]
DEFAULT_START   = "2019-01-01"
DEFAULT_END     = "2024-01-01"
RISK_FREE_RATE  = 0.045          # ~current 3-month T-bill yield
TRADING_DAYS    = 252            # annualisation factor
NUM_SIMULATIONS = 15_000         # random portfolios to simulate
NUM_FRONTIER    = 80             # points on the efficient frontier curve


# ─────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────

def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted closing prices from Yahoo Finance.
    Drops any ticker that has >5% missing data so bad tickers
    don't silently corrupt the analysis.
    """
    print(f"\nFetching price data for: {', '.join(tickers)}")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]

    if isinstance(raw, pd.Series):          # single ticker edge case
        raw = raw.to_frame(name=tickers[0])

    # Drop columns with too many NaNs
    threshold = 0.05 * len(raw)
    bad = [col for col in raw.columns if raw[col].isna().sum() > threshold]
    if bad:
        print(f"  Warning: dropping tickers with missing data: {bad}")
        raw.drop(columns=bad, inplace=True)

    raw.dropna(inplace=True)

    if raw.empty:
        sys.exit("Error: no usable price data returned. Check tickers and date range.")

    print(f"  OK — {len(raw)} trading days, {len(raw.columns)} assets")
    return raw


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Log returns:  r_t = ln(P_t / P_{t-1})
    Preferred over simple returns because they are:
      - Time-additive (annual = sum of daily log returns)
      - More normally distributed (better for statistics)
    """
    return np.log(prices / prices.shift(1)).dropna()


# ─────────────────────────────────────────────
# PORTFOLIO MATH
# ─────────────────────────────────────────────

def portfolio_performance(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    rf: float = RISK_FREE_RATE
) -> tuple[float, float, float]:
    """
    Core formula implementation.

    Return (annual_return, annual_volatility, sharpe_ratio).

    Math:
        E[Rp] = wᵀ · μ                     (weighted mean returns)
        σ²p   = wᵀ · Σ · w                 (matrix form of variance)
        S     = (E[Rp] − Rf) / σp           (Sharpe ratio)
    """
    port_return = float(np.dot(weights, mean_returns))
    port_vol    = float(np.sqrt(weights @ cov_matrix.values @ weights))
    sharpe      = (port_return - rf) / port_vol if port_vol > 0 else 0.0
    return port_return, port_vol, sharpe


# ─────────────────────────────────────────────
# OPTIMIZATION
# ─────────────────────────────────────────────

def _base_constraints(n: int) -> list[dict]:
    """Weights must sum to 1."""
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]


def _bounds(n: int) -> tuple:
    """Each weight in [0, 1] — long-only portfolio (no short selling)."""
    return tuple((0.0, 1.0) for _ in range(n))


def maximize_sharpe(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    rf: float = RISK_FREE_RATE
) -> np.ndarray:
    """
    Find the portfolio with the highest Sharpe ratio.
    Solved via SLSQP (Sequential Least Squares Programming).
    """
    n = len(mean_returns)
    w0 = np.array([1 / n] * n)

    def neg_sharpe(w):
        _, _, s = portfolio_performance(w, mean_returns, cov_matrix, rf)
        return -s

    result = minimize(
        neg_sharpe, w0,
        method="SLSQP",
        bounds=_bounds(n),
        constraints=_base_constraints(n),
        options={"maxiter": 1000, "ftol": 1e-12}
    )

    if not result.success:
        print(f"  Warning (max Sharpe): {result.message}")
    return result.x


def minimize_volatility(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame
) -> np.ndarray:
    """
    Find the minimum variance portfolio (the leftmost point
    on the efficient frontier — the 'safest' achievable portfolio).
    """
    n = len(mean_returns)
    w0 = np.array([1 / n] * n)

    def portfolio_vol(w):
        return portfolio_performance(w, mean_returns, cov_matrix)[1]

    result = minimize(
        portfolio_vol, w0,
        method="SLSQP",
        bounds=_bounds(n),
        constraints=_base_constraints(n),
        options={"maxiter": 1000, "ftol": 1e-12}
    )

    if not result.success:
        print(f"  Warning (min vol): {result.message}")
    return result.x


def efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = NUM_FRONTIER
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trace the efficient frontier by solving, for each target return,
    the minimum-volatility portfolio that achieves exactly that return.

    This gives you the 'upper edge' of the risk-return cloud.
    """
    n = len(mean_returns)
    w0 = np.array([1 / n] * n)

    # Return range: from min-vol portfolio return up to max single-asset return
    min_vol_w   = minimize_volatility(mean_returns, cov_matrix)
    min_ret     = portfolio_performance(min_vol_w, mean_returns, cov_matrix)[0]
    max_ret     = mean_returns.max() * 1.05
    target_rets = np.linspace(min_ret, max_ret, n_points)

    frontier_vols = []
    frontier_rets = []

    for target in target_rets:
        constraints = _base_constraints(n) + [
            {"type": "eq",
             "fun": lambda w, t=target: portfolio_performance(w, mean_returns, cov_matrix)[0] - t}
        ]
        res = minimize(
            lambda w: portfolio_performance(w, mean_returns, cov_matrix)[1],
            w0, method="SLSQP",
            bounds=_bounds(n),
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10}
        )
        if res.success:
            frontier_vols.append(res.fun)
            frontier_rets.append(target)

    return np.array(frontier_vols), np.array(frontier_rets)


# ─────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────

def simulate_portfolios(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_sim: int = NUM_SIMULATIONS,
    rf: float = RISK_FREE_RATE
) -> dict:
    """
    Monte Carlo simulation: generate n_sim random weight vectors,
    compute performance for each, return as arrays for plotting.

    Uses Dirichlet distribution for clean uniform sampling
    over the simplex (all weights positive and summing to 1).
    """
    n = len(mean_returns)
    rets, vols, sharpes = [], [], []

    for _ in range(n_sim):
        w = np.random.dirichlet(np.ones(n))
        r, v, s = portfolio_performance(w, mean_returns, cov_matrix, rf)
        rets.append(r)
        vols.append(v)
        sharpes.append(s)

    return {
        "returns":  np.array(rets),
        "vols":     np.array(vols),
        "sharpes":  np.array(sharpes),
    }


# ─────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────

def print_portfolio(
    label: str,
    weights: np.ndarray,
    tickers: list[str],
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    rf: float = RISK_FREE_RATE
) -> None:
    ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, rf)
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  {'Asset':<10} {'Weight':>8}")
    for ticker, w in zip(tickers, weights):
        bar = "█" * int(w * 30)
        print(f"  {ticker:<10} {w*100:>6.1f}%  {bar}")
    print(f"{'─'*50}")
    print(f"  Expected annual return : {ret*100:>7.2f}%")
    print(f"  Annual volatility      : {vol*100:>7.2f}%")
    print(f"  Sharpe ratio           : {sharpe:>7.3f}")


def print_individual_stats(
    tickers: list[str],
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    returns: pd.DataFrame
) -> None:
    print(f"\n{'─'*60}")
    print("  Individual Asset Statistics (annualised)")
    print(f"{'─'*60}")
    print(f"  {'Ticker':<8} {'Return':>8} {'Volatility':>12} {'Sharpe':>8} {'Skew':>7}")
    print(f"  {'─'*8} {'─'*8} {'─'*12} {'─'*8} {'─'*7}")
    for t in tickers:
        r = mean_returns[t]
        v = np.sqrt(cov_matrix.loc[t, t])
        s = (r - RISK_FREE_RATE) / v
        sk = returns[t].skew()
        print(f"  {t:<8} {r*100:>7.2f}% {v*100:>11.2f}% {s:>8.3f} {sk:>7.3f}")


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def plot_results(
    sim: dict,
    frontier_vols: np.ndarray,
    frontier_rets: np.ndarray,
    ms_weights: np.ndarray,
    mv_weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    tickers: list[str],
    output_path: str = "efficient_frontier.png"
) -> None:
    """
    Three-panel figure:
      Left  — efficient frontier scatter with key portfolios marked
      Top right  — max Sharpe portfolio allocation bar chart
      Bottom right — min vol portfolio allocation bar chart
    """
    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("#fafaf8")

    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], hspace=0.45, wspace=0.35)
    ax_main  = fig.add_subplot(gs[:, 0])
    ax_ms    = fig.add_subplot(gs[0, 1])
    ax_mv    = fig.add_subplot(gs[1, 1])

    # ── Main scatter ──────────────────────────────────
    sc = ax_main.scatter(
        sim["vols"] * 100, sim["returns"] * 100,
        c=sim["sharpes"], cmap="viridis",
        alpha=0.35, s=6, zorder=1
    )
    cbar = fig.colorbar(sc, ax=ax_main, pad=0.02)
    cbar.set_label("Sharpe ratio", fontsize=10)

    # Efficient frontier line
    ax_main.plot(
        frontier_vols * 100, frontier_rets * 100,
        color="#1D9E75", linewidth=2.5, zorder=3, label="Efficient frontier"
    )

    # Max Sharpe portfolio
    ms_ret, ms_vol, ms_sharpe = portfolio_performance(ms_weights, mean_returns, cov_matrix)
    ax_main.scatter(
        ms_vol * 100, ms_ret * 100,
        marker="*", color="#E24B4A", s=320, zorder=5,
        label=f"Max Sharpe  ({ms_sharpe:.2f})"
    )

    # Min volatility portfolio
    mv_ret, mv_vol, mv_sharpe = portfolio_performance(mv_weights, mean_returns, cov_matrix)
    ax_main.scatter(
        mv_vol * 100, mv_ret * 100,
        marker="D", color="#534AB7", s=100, zorder=5,
        label=f"Min volatility  ({mv_sharpe:.2f})"
    )

    ax_main.set_xlabel("Annual volatility — risk  (%)", fontsize=11)
    ax_main.set_ylabel("Expected annual return  (%)", fontsize=11)
    ax_main.set_title("Markowitz Efficient Frontier", fontsize=13, fontweight="normal", pad=12)
    ax_main.legend(fontsize=9, framealpha=0.85)
    ax_main.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax_main.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax_main.set_facecolor("#fafaf8")

    # ── Allocation bar charts ─────────────────────────
    colors_ms = ["#1D9E75" if w > 0.05 else "#9FE1CB" for w in ms_weights]
    colors_mv = ["#534AB7" if w > 0.05 else "#AFA9EC" for w in mv_weights]

    for ax, weights, colors, title, ret, vol, sharpe in [
        (ax_ms, ms_weights, colors_ms, "Max Sharpe portfolio",   ms_ret, ms_vol, ms_sharpe),
        (ax_mv, mv_weights, colors_mv, "Min volatility portfolio", mv_ret, mv_vol, mv_sharpe),
    ]:
        bars = ax.barh(tickers, weights * 100, color=colors, height=0.55)
        ax.set_xlabel("Weight (%)", fontsize=9)
        ax.set_title(
            f"{title}\nReturn {ret*100:.1f}%  |  Vol {vol*100:.1f}%  |  Sharpe {sharpe:.2f}",
            fontsize=9, fontweight="normal", linespacing=1.6
        )
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.set_facecolor("#fafaf8")
        for bar, w in zip(bars, weights):
            if w > 0.03:
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{w*100:.1f}%", va="center", fontsize=8)

    plt.savefig(output_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Chart saved → {output_path}")
    plt.show()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Markowitz portfolio optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python portfolio_optimizer.py
  python portfolio_optimizer.py --tickers AAPL TSLA NVDA AMZN --start 2021-01-01
  python portfolio_optimizer.py --tickers SPY QQQ GLD TLT --rf 0.05 --no-plot
        """
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                        help="Space-separated list of Yahoo Finance ticker symbols")
    parser.add_argument("--start",   default=DEFAULT_START,
                        help="Start date YYYY-MM-DD (default: 2019-01-01)")
    parser.add_argument("--end",     default=DEFAULT_END,
                        help="End date YYYY-MM-DD (default: 2024-01-01)")
    parser.add_argument("--rf",      type=float, default=RISK_FREE_RATE,
                        help="Risk-free rate as decimal (default: 0.045)")
    parser.add_argument("--sims",    type=int,   default=NUM_SIMULATIONS,
                        help="Number of random portfolios to simulate")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip chart generation (useful in headless environments)")
    parser.add_argument("--output",  default="efficient_frontier.png",
                        help="Output filename for chart (default: efficient_frontier.png)")
    return parser.parse_args()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=" * 50)
    print("  Markowitz Portfolio Optimizer")
    print("=" * 50)

    # 1. Data
    prices       = fetch_prices(args.tickers, args.start, args.end)
    returns      = compute_returns(prices)
    tickers      = list(returns.columns)
    mean_returns = returns.mean() * TRADING_DAYS
    cov_matrix   = returns.cov()   * TRADING_DAYS

    # 2. Individual stats
    print_individual_stats(tickers, mean_returns, cov_matrix, returns)

    # 3. Optimize
    print("\nRunning optimization...")
    ms_weights = maximize_sharpe(mean_returns, cov_matrix, rf=args.rf)
    mv_weights = minimize_volatility(mean_returns, cov_matrix)
    print("  Done.")

    print_portfolio("Maximum Sharpe Ratio Portfolio",
                    ms_weights, tickers, mean_returns, cov_matrix, args.rf)
    print_portfolio("Minimum Volatility Portfolio",
                    mv_weights, tickers, mean_returns, cov_matrix, args.rf)

    # 4. Simulate random portfolios
    print(f"\nSimulating {args.sims:,} random portfolios...")
    sim = simulate_portfolios(mean_returns, cov_matrix, n_sim=args.sims, rf=args.rf)
    print("  Done.")

    # 5. Efficient frontier
    print("Tracing efficient frontier...")
    f_vols, f_rets = efficient_frontier(mean_returns, cov_matrix)
    print("  Done.")

    # 6. Plot
    if not args.no_plot:
        plot_results(sim, f_vols, f_rets,
                     ms_weights, mv_weights,
                     mean_returns, cov_matrix,
                     tickers, output_path=args.output)

    print("\n" + "=" * 50)
    print("  Complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
