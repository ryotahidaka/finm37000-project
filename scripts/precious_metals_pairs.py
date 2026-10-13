"""Download precious-metals futures and run a walk-forward pairs backtest.

The backtest uses daily continuous futures prices.  Signals are formed using
data available at the close of day t and are applied to the close-to-close
return from t to t+1.  This deliberately avoids using future prices when
selecting pairs, estimating hedge ratios, or standardising the spread.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


DEFAULT_SYMBOLS = ("GC.c.0", "SI.c.0", "PL.c.0", "PA.c.0")


@dataclass(frozen=True)
class BacktestConfig:
    lookback: int = 252
    formation_min_observations: int = 126
    entry_z: float = 2.0
    exit_z: float = 0.5
    adf_pvalue: float = 0.10
    transaction_cost_bps: float = 2.0
    annualization: int = 252
    initial_capital: float = 1_000_000.0


def parse_date(value: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tz is not None:
        parsed = parsed.tz_localize(None)
    return parsed.normalize()


def download_prices(
    api_key: str,
    symbols: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
) -> pd.DataFrame:
    """Download daily continuous contracts, caching each request as CSV."""
    import databento as db

    client = db.Historical(api_key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    series: dict[str, pd.Series] = {}
    for symbol in symbols:
        cache_file = cache_dir / f"{symbol.replace('.', '_')}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
        if cache_file.exists():
            data = pd.read_csv(cache_file, parse_dates=["date"])
        else:
            data = client.timeseries.get_range(
                dataset="GLBX.MDP3",
                schema="ohlcv-1d",
                symbols=[symbol],
                stype_in="continuous",
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            ).to_df().reset_index()
            if "ts_event" not in data or "close" not in data:
                raise ValueError(f"Databento response for {symbol} lacks ts_event/close columns.")
            data["date"] = pd.to_datetime(data["ts_event"], utc=True).dt.tz_localize(None).dt.normalize()
            data = data[["date", "close"]].dropna().drop_duplicates("date")
            data.to_csv(cache_file, index=False)
        series[symbol] = data.set_index("date")["close"].rename(symbol)

    prices = pd.concat(series.values(), axis=1).sort_index()
    prices = prices.loc[(prices.index >= start) & (prices.index <= end)]
    if prices.dropna(how="all").shape[0] < 2:
        raise ValueError("No daily prices were returned for the requested symbols.")
    return prices


def load_prices(path: Path) -> pd.DataFrame:
    """Load a wide date-indexed price CSV for offline/reproducible reruns."""
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices.index = pd.DatetimeIndex(prices.index).normalize()
    prices = prices.apply(pd.to_numeric, errors="coerce").sort_index().dropna(how="all")
    if prices.shape[1] < 2 or len(prices) < 2:
        raise ValueError("Price CSV must contain at least two symbols and two dates.")
    return prices


def fit_spread(x: pd.Series, y: pd.Series) -> tuple[float, float, pd.Series]:
    """Fit y = alpha + beta*x and return alpha, beta, and residuals."""
    design = np.column_stack([np.ones(len(x)), x.to_numpy()])
    alpha, beta = np.linalg.lstsq(design, y.to_numpy(), rcond=None)[0]
    residuals = y - alpha - beta * x
    return float(alpha), float(beta), residuals


def select_pair(
    prices: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[str, str, float, float, float] | None:
    """Select the lowest-ADF-p-value pair from a trailing formation window."""
    candidates: list[tuple[float, str, str, float, float]] = []
    for i, left in enumerate(prices.columns):
        for right in prices.columns[i + 1 :]:
            pair_prices = prices[[left, right]].dropna().tail(config.lookback)
            if len(pair_prices) < config.formation_min_observations:
                continue
            x, y = np.log(pair_prices[left]), np.log(pair_prices[right])
            alpha, beta, residuals = fit_spread(x, y)
            if residuals.std(ddof=1) == 0:
                continue
            try:
                pvalue = float(adfuller(residuals, maxlag=1, autolag="AIC")[1])
            except (ValueError, np.linalg.LinAlgError):
                continue
            if np.isfinite(pvalue) and pvalue <= config.adf_pvalue:
                candidates.append((pvalue, left, right, alpha, beta))
    if not candidates:
        return None
    return min(candidates)


def backtest(prices: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a daily walk-forward, dollar-neutral mean-reversion backtest."""
    log_prices = np.log(prices)
    asset_returns = prices.pct_change().shift(-1)
    records: list[dict[str, object]] = []
    previous_pair: tuple[str, str] | None = None
    previous_signal = 0
    previous_weights = pd.Series(0.0, index=prices.columns)

    for t in range(config.lookback, len(prices) - 1):
        formation = log_prices.iloc[t - config.lookback : t]
        selected = select_pair(formation, config)
        weights = pd.Series(0.0, index=prices.columns)
        signal = 0
        zscore = np.nan
        pvalue = np.nan
        pair_name = ""

        if selected is not None:
            pvalue, left, right, alpha, beta = selected
            pair = (left, right)
            pair_formation = formation[[left, right]].dropna()
            residuals = pair_formation[right] - alpha - beta * pair_formation[left]
            scale = residuals.std(ddof=1)
            current = log_prices[[left, right]].iloc[t]
            next_prices = asset_returns[[left, right]].iloc[t]
            if current.isna().any() or next_prices.isna().any() or scale == 0:
                selected = None
            else:
                current_residual = current[right] - alpha - beta * current[left]
                zscore = (current_residual - residuals.mean()) / scale
            pair_name = f"{left}/{right}"
            if selected is not None and pair == previous_pair and abs(zscore) <= config.exit_z:
                signal = 0
            elif selected is not None and pair == previous_pair and abs(zscore) < config.entry_z:
                signal = previous_signal
            elif selected is not None and zscore >= config.entry_z:
                signal = -1
            elif selected is not None and zscore <= -config.entry_z:
                signal = 1
            if selected is not None and signal:
                # Positive signal buys the residual (long right, short beta*left).
                denominator = 1.0 + abs(beta)
                weights[left] = -signal * beta / denominator
                weights[right] = signal / denominator

        turnover = float((weights - previous_weights).abs().sum())
        gross_return = float((weights * asset_returns.iloc[t]).sum())
        cost = turnover * config.transaction_cost_bps / 10_000
        net_return = gross_return - cost
        records.append(
            {
                "date": prices.index[t],
                "pair": pair_name,
                "pvalue": pvalue,
                "zscore": zscore,
                "signal": signal,
                "turnover": turnover,
                "gross_return": gross_return,
                "transaction_cost": cost,
                "net_return": net_return,
            }
        )
        previous_pair, previous_signal, previous_weights = (
            (selected[1], selected[2]) if selected is not None else None,
            signal,
            weights,
        )

    daily = pd.DataFrame(records).set_index("date")
    daily["equity"] = config.initial_capital * (1 + daily["net_return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1
    return daily, _performance_summary(daily, config)


def _performance_summary(daily: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    returns = daily["net_return"]
    annual_return = (1 + returns).prod() ** (config.annualization / len(returns)) - 1
    annual_volatility = returns.std(ddof=1) * np.sqrt(config.annualization)
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else np.nan
    return pd.DataFrame(
        {
            "value": {
                "start": daily.index.min().date().isoformat(),
                "end": daily.index.max().date().isoformat(),
                "observations": len(daily),
                "annual_return": annual_return,
                "annual_volatility": annual_volatility,
                "sharpe": sharpe,
                "max_drawdown": daily["drawdown"].min(),
                "total_transaction_cost": daily["transaction_cost"].sum(),
                "days_in_market": int((daily["signal"] != 0).sum()),
            }
        }
    )


def save_results(daily: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, config: BacktestConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_dir / "daily_backtest.csv")
    summary.to_csv(output_dir / "summary.csv")
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    ax = daily["equity"].plot(figsize=(11, 5), title="Precious-metals pairs strategy equity")
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.3)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=150)
    plt.close(fig)


def main() -> None:
    today = pd.Timestamp(date.today())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=parse_date, default=today - pd.Timedelta(days=365 * 20 + 5))
    parser.add_argument("--end", type=parse_date, default=today)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/databento"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/precious_metals_pairs"))
    parser.add_argument("--prices-csv", type=Path, help="Use a local wide price CSV instead of downloading.")
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0)
    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("--end must be after --start")

    if args.prices_csv:
        prices = load_prices(args.prices_csv)
    else:
        api_key = os.environ.get("DATABENTO_API_KEY")
        if not api_key:
            parser.error("DATABENTO_API_KEY is required unless --prices-csv is supplied.")
        prices = download_prices(api_key, args.symbols, args.start, args.end, args.cache_dir)

    config = BacktestConfig(lookback=args.lookback, transaction_cost_bps=args.transaction_cost_bps)
    daily, summary = backtest(prices, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(args.output_dir / "prices.csv")
    save_results(daily, summary, args.output_dir, config)
    print(summary.to_string())
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
