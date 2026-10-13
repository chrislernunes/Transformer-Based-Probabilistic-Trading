# Transformer-Based Probabilistic Trading Strategy on the Nifty 50

A transformer classifier is trained per stock to predict the next day's direction; fixed exit rules turn
the predictions into trades. Each stock is an independent capital sleeve and the portfolio is the sum of
the sleeves, compared with a Nifty 50 (^NSEI) buy-and-hold benchmark.

## Method
- Features: log return, momentum (close minus 5-day MA), RSI(14), MACD difference; 20-day sequences
- Model: Linear 4->32, TransformerEncoder (2 layers, 4 heads), flatten, dense 64 (ReLU), sigmoid
- Training: BCE loss, Adam lr=0.001, 20 epochs, batch 32, no shuffling
- Split: by date. Train on data before 2024-07-01 (from 2020-01-01); test from 2024-07-01 to the latest date
- Entry: P(up) > 0.5 AND RSI > 50 AND MACD difference > 0
- Exits: +4% take-profit, -2% stop-loss (daily closes, filled at the threshold), or the day-5 close
- Sizing: INR 10,000 per trade inside an INR 100,000 sleeve per stock; no compounding; trades may overlap;
  no costs or slippage
- Stocks with under 300 training sequences before the test start are skipped
- PnL is booked on the exit date. A trade opens two trading days after the last day in the model's input
  window, one day after the move the label refers to.

## Run
pip install -r requirements.txt
python nifty_transformer_strategy.py

Outputs go to results/: portfolio_metrics.csv, equity_curve.csv/.png, trades.csv, signal_summary.csv
(per-stock test accuracy and AUC). Research only, not investment advice.
