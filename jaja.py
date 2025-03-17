import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf  # <-- pip install mplfinance if needed

##############################################################################
# 1) READ & PREPARE DATA
##############################################################################
file_path = "/Users/oscarolsen/Desktop/Algorithmic Trading/Algorithmic-Trading/Stor pik.csv"

df = pd.read_csv(
    file_path,
    sep=";",         # Adjust if your CSV uses a different delimiter
    decimal=",",     # Interpret commas as decimal points
    header=0         # If the first row is: Date;Last Price
)

# Rename columns (assuming CSV header is "Date" and "Last Price")
df.rename(columns={"Date": "timestamp", "Last Price": "close"}, inplace=True)

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d/%m/%y %H.%M.%S", errors="coerce")
df.dropna(subset=["timestamp"], inplace=True)
df.set_index("timestamp", inplace=True)

df.sort_index(inplace=True)

# (Optional) Filter to standard market hours: 09:30-16:00 (if you have intraday data)
# If your data is daily, remove this line!
df = df.between_time("09:30", "16:00")

if len(df) == 0:
    raise ValueError("No rows remain. Remove the 'between_time' filter if you have daily data.")

# Duplicate close into open/high/low for ATR or candlestick
df["open"] = df["close"]
df["high"] = df["close"]
df["low"] = df["close"]

##############################################################################
# 2) DEFINE HELPER FUNCTIONS: RSI, MACD, ATR
##############################################################################

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def compute_atr(df, period=14):
    df["H-L"] = df["high"] - df["low"]
    df["H-PC"] = (df["high"] - df["close"].shift(1)).abs()
    df["L-PC"] = (df["low"] - df["close"].shift(1)).abs()
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(window=period).mean()
    return df["ATR"]

##############################################################################
# 3) CALCULATE INDICATORS: MACD, RSI, ATR, 50-SMA
##############################################################################

df["MACD"], df["MACD_Signal"] = compute_macd(df["close"], fast=12, slow=26, signal=9)
df["RSI"] = compute_rsi(df["close"], period=14)
df["ATR"] = compute_atr(df, period=14)
df["SMA_50"] = df["close"].rolling(window=50).mean()

df.dropna(subset=["RSI", "ATR", "SMA_50"], inplace=True)

##############################################################################
# 4) GENERATE ENTRY/EXIT SIGNALS (TREND FILTER)
##############################################################################
# For example:
# - LONG if MACD>Signal, RSI<70, close>SMA_50
# - SHORT if MACD<Signal, RSI>30, close<SMA_50

df["Position"] = 0
df.loc[
    (df["MACD"] > df["MACD_Signal"]) &
    (df["RSI"] < 70) &
    (df["close"] > df["SMA_50"]),
    "Position"
] = 1

df.loc[
    (df["MACD"] < df["MACD_Signal"]) &
    (df["RSI"] > 30) &
    (df["close"] < df["SMA_50"]),
    "Position"
] = -1

df["Trade_Signal"] = df["Position"].diff().fillna(0)

##############################################################################
# 5) BACKTEST: RISK-BASED POSITION SIZING + TRAILING STOP + EOD FLATTEN (optional)
##############################################################################

initial_cash = 1_000_000
cash = initial_cash
position = 0
entry_price = None
stop_loss_price = None
trade_cost = 0
total_pnl = 0
trade_log = []
portfolio_value = []

stop_loss_multiplier = 2.0
transaction_cost = 1.00
risk_per_trade_percent = 0.02

timestamps = df.index.tolist()

for i in range(len(df)):
    current_time = timestamps[i]
    price = df["close"].iloc[i]
    signal_val = df["Trade_Signal"].iloc[i]
    current_atr = df["ATR"].iloc[i]

    current_equity = cash + position * price

    # Update trailing stop
    if position > 0:  # LONG
        new_stop = price - stop_loss_multiplier * current_atr
        if stop_loss_price is None:
            stop_loss_price = new_stop
        else:
            stop_loss_price = max(stop_loss_price, new_stop)

        # Check stop
        if price <= stop_loss_price:
            proceeds = position * price - transaction_cost
            trade_pnl = proceeds - trade_cost
            cash += proceeds
            total_pnl += trade_pnl
            trade_log.append(f"{current_time} - STOP LOSS LONG {position} @ {price:.2f} | PnL: ${trade_pnl:.2f}")
            position = 0
            entry_price = None
            stop_loss_price = None
            trade_cost = 0

    elif position < 0:  # SHORT
        new_stop = price + stop_loss_multiplier * current_atr
        if stop_loss_price is None:
            stop_loss_price = new_stop
        else:
            stop_loss_price = min(stop_loss_price, new_stop)

        # Check stop
        if price >= stop_loss_price:
            proceeds = abs(position) * price - transaction_cost
            trade_pnl = proceeds - trade_cost
            cash += proceeds
            total_pnl += trade_pnl
            trade_log.append(f"{current_time} - STOP LOSS SHORT {abs(position)} @ {price:.2f} | PnL: ${trade_pnl:.2f}")
            position = 0
            entry_price = None
            stop_loss_price = None
            trade_cost = 0

    # Check new signals if flat
    if position == 0:
        if signal_val == 1:  # BUY
            stop_distance = stop_loss_multiplier * current_atr
            risk_amount = risk_per_trade_percent * current_equity if stop_distance > 0 else 0
            shares = int(risk_amount / stop_distance) if stop_distance > 0 else 0
            max_affordable_shares = int((cash - transaction_cost) // price)
            shares = min(shares, max_affordable_shares)
            cost_to_open = shares * price + transaction_cost

            if shares > 0 and cost_to_open <= cash:
                cash -= cost_to_open
                position = shares
                entry_price = price
                trade_cost = cost_to_open
                stop_loss_price = price - stop_distance
                trade_log.append(f"{current_time} - BUY {shares} @ {price:.2f}")

        elif signal_val == -1:  # SHORT
            stop_distance = stop_loss_multiplier * current_atr
            risk_amount = risk_per_trade_percent * current_equity if stop_distance > 0 else 0
            shares = int(risk_amount / stop_distance) if stop_distance > 0 else 0
            max_affordable_shares = int((cash - transaction_cost) // price)
            shares = min(shares, max_affordable_shares)
            cost_to_open = shares * price + transaction_cost

            if shares > 0 and cost_to_open <= cash:
                cash -= cost_to_open
                position = -shares
                entry_price = price
                trade_cost = cost_to_open
                stop_loss_price = price + stop_distance
                trade_log.append(f"{current_time} - SHORT {shares} @ {price:.2f}")

    # Track portfolio value each bar
    portfolio_value.append(cash + position * price)

    # (Optional) Flatten end-of-day if desired:
    # Check if next bar is a different day, if so, close position
    # ... (commented out for brevity; see previous example if you want EOD flatten)

# Final flatten if still open
if position != 0:
    final_price = df["close"].iloc[-1]
    proceeds = abs(position) * final_price - transaction_cost
    trade_pnl = proceeds - trade_cost
    cash += proceeds
    total_pnl += trade_pnl
    trade_log.append(f"{timestamps[-1]} - FINAL CLOSE @ {final_price:.2f} | PnL: ${trade_pnl:.2f}")
    position = 0
    entry_price = None
    stop_loss_price = None
    trade_cost = 0

final_equity = cash
overall_pnl = final_equity - initial_cash

##############################################################################
# 6) BUY & HOLD COMPARISON
##############################################################################
buy_hold_shares = initial_cash / df["close"].iloc[0]
df["Buy_Hold_Value"] = buy_hold_shares * df["close"]
final_buy_hold = df["Buy_Hold_Value"].iloc[-1]
buy_hold_pnl = final_buy_hold - initial_cash

##############################################################################
# 7) PRINT RESULTS
##############################################################################
print("="*60)
print("FINAL TRADING SUMMARY")
print("="*60)
print(f"Total Trades Executed: {len(trade_log)}")
print(f"Strategy Final Equity: ${final_equity:,.2f}")
print(f"Strategy PnL:          ${overall_pnl:,.2f}")
print(f"Buy & Hold Final:      ${final_buy_hold:,.2f}")
print(f"Buy & Hold PnL:        ${buy_hold_pnl:,.2f}")
print("="*60)
print("Trade Log:")
for log_entry in trade_log:
    print(log_entry)

##############################################################################
# 8) MPLFINANCE CANDLESTICK PLOT (SMOOTHER, NO DAY JUMPS)
##############################################################################
#  - This won't automatically show your strategy equity or MACD/RSI in the same figure.
#  - But it will compress non-trading periods so the chart looks continuous.

# Prepare a DataFrame for mplfinance:
#   - Must have columns named: 'Open', 'High', 'Low', 'Close' (capitalized)
df_for_mpf = df.copy()
df_for_mpf.rename(
    columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close"
    },
    inplace=True
)
# If you had volume, you'd also have a 'Volume' column.

# Markers for buy/sell signals using mpf.make_addplot
buy_locs = df_for_mpf.index[df["Trade_Signal"] == 1]
sell_locs = df_for_mpf.index[df["Trade_Signal"] == -1]

# We'll place markers slightly above/below the candle's Close
buy_markers = df_for_mpf["Close"].loc[buy_locs] * 1.0005  # tiny offset above
sell_markers = df_for_mpf["Close"].loc[sell_locs] * 0.9995  # tiny offset below

apds = []
# Buy signals (green '^')
apds.append(
    mpf.make_addplot(
        buy_markers,
        type='scatter',
        marker='^',
        markersize=75,       # adjust marker size
        color='g'
    )
)
# Sell signals (red 'v')
apds.append(
    mpf.make_addplot(
        sell_markers,
        type='scatter',
        marker='v',
        markersize=75,
        color='r'
    )
)

# Plot with mplfinance
mpf.plot(
    df_for_mpf,
    type='candle',           # candlestick
    style='yahoo',           # color/style theme
    addplot=apds,
    show_nontrading=False,   # compress non-trading hours/days
    figsize=(12, 8),
    title="Candlestick Chart (Compressed) with Buy/Sell Signals",
    tight_layout=True
)
