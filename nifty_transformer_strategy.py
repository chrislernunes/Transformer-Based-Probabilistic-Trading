#!/usr/bin/env python3
"""
Transformer-Based Probabilistic Trading Strategy on the Nifty 50.

One transformer classifier per stock predicts the next-day up-move; stop-loss / take-profit /
time-exit rules turn the predictions into trades. Each stock is an independent capital "sleeve";
the portfolio curve is the sum of sleeves and is compared against a Nifty 50 index (^NSEI)
buy-and-hold benchmark.

Data window: 2020-01-01 -> latest available date. Models train on data before TEST_START and are
tested from that date on (more than two years).
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import ta
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
START_DATE = "2020-01-01"        # training history starts here
END_DATE = None                  # None = latest available date
BENCHMARK = "^NSEI"              # added: Nifty 50 index, price return
SEQ_LEN = 20
TEST_START = "2024-07-01"        # train before this date, test from it onward (2+ years)
MIN_TRAIN_SEQS = 300             # skip stocks with too little history before TEST_START (recent listings)
EPOCHS = 20
BATCH_SIZE = 32
LR = 0.001
THRESHOLD = 0.5                  # entry probability threshold
SL_PCT, TP_PCT = -0.02, 0.04
MAX_HOLD = 5                     # maximum holding period in trading days
ALLOW_OVERLAP = False            # False: at most one open position per stock at a time
INITIAL_CASH = 100_000           # per stock sleeve (INR)
POSITION_SIZE = 10_000           # per trade, 10% of a sleeve (INR)

NIFTY50 = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDIGO.NS", "INFY.NS", "ITC.NS",
    "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "MAXHEALTH.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS",
    "SUNPHARMA.NS", "TATACONSUM.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS",
    "TITAN.NS", "TMPV.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS"
]

os.makedirs("results", exist_ok=True)


# ===================== MODEL =====================
class TransformerBinaryClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model * seq_len, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc_out(x.reshape(x.size(0), -1))


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def load_close(ticker):
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df.columns, pd.MultiIndex):          # flatten multi-level columns
        df.columns = [col[0] for col in df.columns]
    return df


def build_features(df):
    df = df.copy()
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Momentum"] = df["Close"] - df["Close"].rolling(window=5).mean()
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    df["MACD_diff"] = ta.trend.MACD(close=df["Close"]).macd_diff()
    return df.dropna()


# ===================== BACKTEST =====================
def run_backtest(ticker, dates, close_prices, entry_signal):
    """
    Signal i buys close_prices[i+1]; then for j=1..5 the close close_prices[i+j+1] is checked for
    take-profit (+4%) / stop-loss (-2%), otherwise the position exits at the day-5 close.
    TP/SL fill exactly at the threshold. The last signal is skipped (no later prices).
    With ALLOW_OVERLAP = False a stock holds one position at a time: signals while a position is
    open are ignored, and a new entry must be on a later day than the previous exit.
    PnL is booked on the EXIT date so the equity curve has a real date axis.
    """
    trades = []
    n = len(close_prices)
    last_exit = -1
    for i in range(len(entry_signal) - 1):
        if not entry_signal[i]:
            continue
        if not ALLOW_OVERLAP and i + 1 <= last_exit:
            continue
        buy_price = close_prices[i + 1]
        entry_date = dates[i + 1]
        exit_idx, pct, reason = None, None, None
        for j in range(1, MAX_HOLD + 1):
            if i + j + 1 >= n:                          # ran out of data: close at last price
                exit_idx = n - 1
                pct = (close_prices[-1] - buy_price) / buy_price
                reason = "END"
                break
            chg = (close_prices[i + j + 1] - buy_price) / buy_price
            if chg >= TP_PCT:
                exit_idx, pct, reason = i + j + 1, TP_PCT, "TP"
                break
            elif chg <= SL_PCT:
                exit_idx, pct, reason = i + j + 1, SL_PCT, "SL"
                break
        if exit_idx is None:                            # no TP/SL in 5 days: exit at day-5 close
            exit_idx = i + MAX_HOLD + 1
            pct = (close_prices[exit_idx] - buy_price) / buy_price
            reason = "TIME"
        last_exit = exit_idx
        trades.append({
            "ticker": ticker, "signal_date": dates[i], "entry_date": entry_date,
            "exit_date": dates[exit_idx], "entry_px": buy_price,
            "exit_px": close_prices[exit_idx], "pct": round(float(pct), 4),
            "pnl": round(float(POSITION_SIZE * pct), 2), "reason": reason,
        })
    return trades


def sleeve_equity(dates, trades):
    """INITIAL_CASH plus cumulative realised PnL by exit date, on the stock's test dates."""
    pnl = pd.Series(0.0, index=pd.DatetimeIndex(dates))
    for t in trades:
        pnl.loc[t["exit_date"]] += t["pnl"]
    return INITIAL_CASH + pnl.cumsum()


def perf_stats(curve):
    r = curve.pct_change().dropna()
    yrs = max((len(curve) - 1) / 252, 1e-9)
    tot = (curve.iloc[-1] / curve.iloc[0] - 1) * 100
    cagr = ((curve.iloc[-1] / curve.iloc[0]) ** (1 / yrs) - 1) * 100
    sharpe = (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    dd = (curve - curve.cummax()) / curve.cummax() * 100
    return tot, cagr, sharpe, abs(dd.min()), dd


# ===================== MAIN =====================
if __name__ == "__main__":
    print(f"Downloading benchmark {BENCHMARK} ...")
    bench_raw = load_close(BENCHMARK)

    sleeves, all_trades, summary, skipped = {}, [], [], []

    for ticker in NIFTY50:
        try:
            raw = load_close(ticker)
        except Exception as e:
            skipped.append((ticker, f"download error: {e}"))
            continue
        if len(raw) == 0:
            skipped.append((ticker, "no data"))
            continue

        df = build_features(raw)
        features = df[["Return", "Momentum", "RSI", "MACD_diff"]].values
        targets = (df["Close"].shift(-1) > df["Close"]).astype(int).values
        features, targets = features[:-1], targets[:-1]            # drop last row (no next-day label)
        X_seq, y_seq = create_sequences(features, targets, SEQ_LEN)

        n = len(X_seq)
        # date-based split: sequence k is dated at row k+SEQ_LEN
        seq_dates = df.index[SEQ_LEN: SEQ_LEN + n]
        split = int((seq_dates < pd.Timestamp(TEST_START)).sum())
        n_test = n - split
        if split < MIN_TRAIN_SEQS or n_test < 20:
            skipped.append((ticker, f"train sequences {split}, test sequences {n_test}"))
            continue
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]

        torch.manual_seed(42)
        np.random.seed(42)
        model = TransformerBinaryClassifier(input_dim=4, seq_len=SEQ_LEN)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=BATCH_SIZE)  # no shuffle
        for _ in range(EPOCHS):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()

        last = X_test[:, -1, :]                                    # last timestep only
        entry_signal = (probs > THRESHOLD) & (last[:, 2] > 50) & (last[:, 3] > 0)

        # first test price row: len(df) - len(entry_signal) - 1  (== split + SEQ_LEN)
        start = len(df) - len(entry_signal) - 1
        close_prices = df["Close"].values[start:]
        dates = df.index[start:]

        trades = run_backtest(ticker, dates, close_prices, entry_signal)
        all_trades.extend(trades)
        sleeves[ticker] = sleeve_equity(dates, trades)

        acc = float(((probs > 0.5).astype(int) == y_test).mean())
        try:
            auc = float(roc_auc_score(y_test, probs))
        except ValueError:
            auc = float("nan")
        summary.append({"ticker": ticker, "signals": int(entry_signal.sum()), "trades": len(trades),
                        "test_accuracy": round(acc, 4), "test_auc": round(auc, 4),
                        "total_pnl": round(sum(t["pnl"] for t in trades), 2)})
        print(f"  {ticker}: test {len(entry_signal)}d, signals {int(entry_signal.sum())}, "
              f"trades {len(trades)}, acc {acc:.3f}, auc {auc:.3f}")

    if not sleeves:
        raise SystemExit("No stocks were usable.")
    for t, why in skipped:
        print(f"  Skipped {t}: {why}")

    # ---- aggregate portfolio = sum of independent sleeves ----
    union = sorted(set().union(*[s.index for s in sleeves.values()]))
    port = pd.concat([s.reindex(union).ffill().fillna(INITIAL_CASH) for s in sleeves.values()], axis=1).sum(axis=1)
    total_cap = INITIAL_CASH * len(sleeves)

    bench = bench_raw["Close"].reindex(port.index).ffill().dropna()
    bench = total_cap * bench / bench.iloc[0]                      

    tot, cagr, sharpe, maxdd, dd = perf_stats(port)
    b_tot, b_cagr, b_sharpe, b_dd, b_ddser = perf_stats(bench)
    wins = sum(1 for t in all_trades if t["pnl"] > 0)
    win_rate = wins / len(all_trades) * 100 if all_trades else 0.0

    metrics = {
        "Stocks_Modeled": len(sleeves), "Stocks_Skipped": len(skipped),
        "Capital_Per_Stock_INR": INITIAL_CASH, "Total_Capital_INR": total_cap,
        "Final_Value_INR": round(port.iloc[-1], 2),
        "Total_Return_pct": round(tot, 2), "CAGR_pct": round(cagr, 2),
        "Sharpe": round(sharpe, 2), "Max_Drawdown_pct": round(maxdd, 2),
        "Win_Rate_pct": round(win_rate, 2), "Total_Trades": len(all_trades), "Winning_Trades": wins,
        "Train_End_Before": TEST_START,
        "Test_Start": str(port.index.min().date()), "Test_End": str(port.index.max().date()),
        "Benchmark": BENCHMARK, "Benchmark_Total_Return_pct": round(b_tot, 2),
        "Benchmark_CAGR_pct": round(b_cagr, 2), "Benchmark_Sharpe": round(b_sharpe, 2),
        "Benchmark_Max_Drawdown_pct": round(b_dd, 2),
        "Excess_Return_vs_Benchmark_pct": round(tot - b_tot, 2),
    }
    print("\n========== PORTFOLIO RESULTS (date split) ==========")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    pd.DataFrame([metrics]).to_csv("results/portfolio_metrics.csv", index=False)
    pd.DataFrame({"strategy": port, "benchmark": bench}).to_csv("results/equity_curve.csv")
    pd.DataFrame(all_trades).to_csv("results/trades.csv", index=False)
    pd.DataFrame(summary).to_csv("results/signal_summary.csv", index=False)

    # ---- chart: strategy on the left axis, Nifty 50 index level on the right axis ----
    nifty_level = bench_raw["Close"].reindex(port.index).ffill()
    fig, ax = plt.subplots(figsize=(12, 6))
    c_s, c_b = "#0B7A6F", "#D98E04"
    l1, = ax.plot(port.index, port.values, linewidth=2, color=c_s,
                  label=f"Strategy, left axis ({tot:+.1f}%, Sharpe {sharpe:.2f}, MaxDD {maxdd:.1f}%)")
    ax.axhline(total_cap, color="gray", linestyle="--", alpha=0.4)
    ax.set_ylabel("Strategy equity (INR)", color=c_s)
    ax.tick_params(axis="y", colors=c_s)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    l2, = ax2.plot(nifty_level.index, nifty_level.values, linewidth=2, color=c_b,
                   label=f"Nifty 50 index, right axis ({b_tot:+.1f}%, Sharpe {b_sharpe:.2f}, MaxDD {b_dd:.1f}%)")
    ax2.set_ylabel("Nifty 50 index level", color=c_b)
    ax2.tick_params(axis="y", colors=c_b)
    ax.set_title(f"Nifty 50 Transformer Strategy vs Nifty 50 ({metrics['Test_Start']} to {metrics['Test_End']})")
    ax.legend(handles=[l1, l2], loc="upper left")
    plt.tight_layout()
    plt.savefig("results/equity_curve.png", dpi=150)
    print("\nResults saved to results/\nDone.")
