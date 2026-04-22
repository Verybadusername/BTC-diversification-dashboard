import datetime as dt
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


DOWNLOAD_START = dt.date(2017, 1, 1)
DEFAULT_RANGE_START = dt.date(2023, 1, 3)
BTC_TICKER = "BTC-USD"
BOND_TICKER = "TLT"
MARKET_TICKERS = {"SPY (S&P 500)": "SPY", "URTH (MSCI World)": "URTH"}


@st.cache_data(ttl=3600)
def load_close_prices(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    tickers = [BTC_TICKER, "SPY", "URTH", BOND_TICKER]
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date + dt.timedelta(days=1),
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError("No data returned from yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw.copy()

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    missing = [ticker for ticker in tickers if ticker not in close.columns]
    if missing:
        raise ValueError(f"Missing expected ticker columns: {', '.join(missing)}")

    close = close[tickers].dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index()


def build_portfolio_returns(
    asset_returns: pd.DataFrame,
    market_ticker: str,
    btc_weight: float,
    market_weight: float,
    bond_weight: float,
) -> pd.DataFrame:
    btc = asset_returns[BTC_TICKER]
    market = asset_returns[market_ticker]
    bond = asset_returns[BOND_TICKER]

    output = pd.DataFrame(index=asset_returns.index)
    output["Market (100%)"] = market
    output["BTC (100%)"] = btc
    output["Bond (100%)"] = bond
    output["2% BTC / 98% Market"] = 0.02 * btc + 0.98 * market
    output["5% BTC / 95% Market"] = 0.05 * btc + 0.95 * market
    output["10% BTC / 90% Market"] = 0.10 * btc + 0.90 * market
    output["5% BTC / 60% Market / 35% Bond"] = 0.05 * btc + 0.60 * market + 0.35 * bond
    output["Custom Portfolio"] = btc_weight * btc + market_weight * market + bond_weight * bond
    return output


def normalize_weights(btc_pct: float, market_pct: float, bond_pct: float) -> Tuple[float, float, float]:
    raw = np.array([btc_pct, market_pct, bond_pct], dtype=float)
    total = raw.sum()
    if total <= 0:
        return 0.0, 1.0, 0.0
    normalized = raw / total
    return float(normalized[0]), float(normalized[1]), float(normalized[2])


st.set_page_config(page_title="Portfolio Returns Dashboard", page_icon="📈", layout="wide")
st.title("Portfolio Allocation Returns Dashboard")
st.caption(
    "Compare portfolio return series with editable date range and custom BTC/market/bond weights."
)

with st.spinner("Loading BTC, market, and bond price history..."):
    try:
        close_prices = load_close_prices(DOWNLOAD_START, dt.date.today())
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

asset_returns = close_prices.pct_change().dropna()
min_date = asset_returns.index.min().date()
max_date = asset_returns.index.max().date()
default_start = DEFAULT_RANGE_START if DEFAULT_RANGE_START >= min_date else min_date

st.sidebar.header("Dashboard Controls")
market_label = st.sidebar.selectbox("Market index", options=list(MARKET_TICKERS.keys()))
market_ticker = MARKET_TICKERS[market_label]

st.sidebar.subheader("Custom Portfolio Allocation (%)")
btc_weight_pct = st.sidebar.slider("BTC", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
market_weight_pct = st.sidebar.slider(
    "Market index", min_value=0.0, max_value=100.0, value=50.0, step=1.0
)
bond_weight_pct = st.sidebar.slider(
    "Bond index (TLT)", min_value=0.0, max_value=100.0, value=40.0, step=1.0
)

raw_total = btc_weight_pct + market_weight_pct + bond_weight_pct
btc_weight, market_weight, bond_weight = normalize_weights(
    btc_weight_pct, market_weight_pct, bond_weight_pct
)
if not np.isclose(raw_total, 100.0):
    st.sidebar.info(f"Input weights sum to {raw_total:.1f}%. They are normalized to 100% for the custom portfolio.")

st.sidebar.markdown(
    (
        f"Effective weights: **BTC {btc_weight:.1%}** / "
        f"**Market {market_weight:.1%}** / **Bond {bond_weight:.1%}**"
    )
)

date_range = st.sidebar.date_input(
    "Plot date range",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
)

if not isinstance(date_range, tuple) or len(date_range) != 2:
    st.warning("Please select both a start date and an end date.")
    st.stop()

start_date, end_date = date_range
if start_date > end_date:
    st.error("Start date must be earlier than end date.")
    st.stop()

all_portfolios = build_portfolio_returns(
    asset_returns=asset_returns,
    market_ticker=market_ticker,
    btc_weight=btc_weight,
    market_weight=market_weight,
    bond_weight=bond_weight,
)
portfolio_returns = all_portfolios.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]
if portfolio_returns.empty:
    st.error("No return data available for the selected date range.")
    st.stop()

st.subheader("Return Series Plot")
plot_mode = st.radio(
    "Display mode",
    options=["Cumulative growth of $1", "Daily returns"],
    horizontal=True,
)

selected_series = st.multiselect(
    "Toggle return series",
    options=portfolio_returns.columns.tolist(),
    default=portfolio_returns.columns.tolist(),
)

if not selected_series:
    st.warning("Select at least one series to render the chart.")
    st.stop()

plot_df = portfolio_returns[selected_series].copy()
if plot_mode == "Cumulative growth of $1":
    plot_df = (1 + plot_df).cumprod()
    y_axis_label = "Growth of $1"
    plot_title = f"Cumulative Returns ({start_date} to {end_date})"
else:
    y_axis_label = "Daily Return"
    plot_title = f"Daily Portfolio Returns ({start_date} to {end_date})"

fig, ax = plt.subplots(figsize=(13, 7))
for col in plot_df.columns:
    ax.plot(plot_df.index, plot_df[col], label=col, linewidth=2)

ax.set_title(plot_title)
ax.set_xlabel("Date")
ax.set_ylabel(y_axis_label)
ax.grid(alpha=0.25)
ax.legend()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("Snapshot Metrics")
if plot_mode == "Cumulative growth of $1":
    total_return = ((1 + portfolio_returns[selected_series]).cumprod().iloc[-1] - 1.0) * 100
    summary = pd.DataFrame(
        {
            "Total return (%)": total_return,
            "Annualized volatility (%)": portfolio_returns[selected_series].std() * np.sqrt(252) * 100,
        }
    )
else:
    summary = pd.DataFrame(
        {
            "Mean daily return (%)": portfolio_returns[selected_series].mean() * 100,
            "Daily volatility (%)": portfolio_returns[selected_series].std() * 100,
        }
    )

st.dataframe(summary.style.format("{:.2f}"), use_container_width=True)
