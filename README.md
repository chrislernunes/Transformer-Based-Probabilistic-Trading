# Nifty 50 Transformer-Based Probabilistic Trading Strategy

Port of the QuantWolf post "A Transformer-Based Probabilistic Trading Strategy" (2025-04-08):
https://yibowang622.github.io/machine%20learning/quantitative%20trading/2025/04/08/A-Transformer-Based-Probabilistic-Trading-Strategy.html

The blog trains one model on AAPL. Here the same pipeline is run once per Nifty 50 stock.

## Kept identical to the blog
- Features: log Return, Momentum (Close - MA5), RSI(14), MACD_diff; sequence length 20
- Model: Linear 4->32, TransformerEncoder (2 layers, 4 heads, default dropout 0.1), flatten, FC 64, sigmoid
- Training: BCELoss, Adam lr=0.001, 20 epochs, batch 32, no shuffling
- Target = next-day up-move; temporal split with no shuffling across the boundary (blog: 80/20, here: by date, see Changes)
- Entry: P(up) > 0.5 AND RSI > 50 AND MACD_diff > 0, using the last timestep of the window
- Backtest indexing: signal i buys close_prices[i+1]; TP +4% / SL -2% checked on daily closes
  (exact fills at the threshold), otherwise exit at the day-5 close; last signal skipped
- Fixed position size, no compounding; trades may overlap; no costs or slippage

## Changes
- Universe / period: Nifty 50 stocks, data from 2020-01-01 to the latest date. Models train on data before 2024-07-01 and are tested from then on (2+ years) instead of the blog's last-20% split
- Each stock is an independent sleeve (INR 100,000, INR 10,000 per trade = the blog's 10% ratio);
  portfolio = sum of sleeves. Stocks with under 300 training sequences before the test start (recent listings) are skipped
- PnL is booked on the exit date (the blog books it on the signal date) so the curve has a true date axis
- Threshold fixed at 0.5. The blog's F1 search compares string key '1' against sklearn's '1.0' keys,
  so it appears to always fall back to 0.5
- Added: Nifty 50 (^NSEI) buy-and-hold benchmark in metrics and on the chart (strategy on the left axis, index level on the right); per-stock test accuracy and AUC

## Known quirk inherited from the blog
Entry occurs two trading days after the last day in the input window, one day after the day the
model's label refers to. Kept for fidelity.

## Run
pip install -r requirements.txt
python nifty_transformer_strategy.py

Outputs go to results/: portfolio_metrics.csv, equity_curve.csv/.png, trades.csv, signal_summary.csv.
Research recreation, not investment advice.
