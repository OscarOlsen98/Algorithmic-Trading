# Algorithmic-Trading

This repository contains the full codebase for our master's thesis in Economics at the University of Copenhagen (2025), focused on rule-based algorithmic trading using technical oscillators.

**Thesis title:** *Algorithmic Trading with RSI, MACD, and Stochastic Oscillators*  
**University:** University of Copenhagen — Department of Economics  
**Data:** 1-minute OHLCV data from S&P 500 E-mini futures  
**Strategies:** RSI, MACD, and Stochastic (optimized via parameter sweeps)

---

##  Repository Contents

```bash
.
├── combined_all_three.csv     # Input dataset (minute-level OHLCV)
├── RSI.ipynb                  # Full RSI backtest and optimization
├── MACD.ipynb                 # MACD grid scan and parameter optimization
├── Stochastic.ipynb           # Stochastic oscillator logic and optimization
├── Output files/              # Auto-saved visualizations, heatmaps, summaries
├── README.md                  # You're reading it


---

##  Method Summary

- **Indicators**: All indicators are calculated using pandas (rolling/EMA functions).
- **Entry Logic**: Based on crossovers or threshold breaches defined in literature.
- **Volatility Regime Filter**: Only trades when volatility ratio (`VOL_RATIO = σₛ / σₗ`) is outside the neutral range (0.9–1.1).
- **Exit Logic**: Fixed ATR-based stop-loss and take-profit (typically 2×ATR SL, 3×ATR TP).
- **Capital Model**: \$1,000,000 starting capital, 20% risk per trade, max 500 units.

---

## Outputs

Each script outputs:

- Terminal equity
- Number of trades
- Win rate
- % of trades during swing vs. momentum regimes
- Heatmaps (if optimization enabled)
- Top-10 performing parameter combinations

---

##  Requirements

- Python 3.8+
- Packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - jupyter (optional for notebooks)

You can install them via:

```bash
pip install -r requirements.txt

