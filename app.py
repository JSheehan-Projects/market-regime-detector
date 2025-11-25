import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from hmmlearn import hmm

# --- Configuration ---
st.set_page_config(page_title="Market Regime Detector (HMM)", layout="wide")

# --- 1. Sidebar & Inputs ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
window_size = st.sidebar.slider("Volatility Lookback (Days)", 5, 60, 20)

st.title(f"📊 Market Regime Detector (HMM): {ticker}")
st.markdown("""
**Methodology:** Uses a **Gaussian Hidden Markov Model (HMM)** (Bishop Ch. 13).
Unlike simple clustering, this model assumes the market has "memory"—the regime today depends on the regime yesterday.
""")

# --- 2. Data Fetching (Robust) ---
@st.cache_data
def get_data(ticker, start):
    try:
        data = yf.download(ticker, start=start)
    except Exception as e:
        return None
    
    if data.empty:
        return None

    # Fix MultiIndex columns (yfinance update)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Fallback for price column
    if 'Adj Close' not in data.columns and 'Close' in data.columns:
        data['Adj Close'] = data['Close']

    if 'Adj Close' not in data.columns:
        return None

    # Calculate Returns & Volatility
    data['Log_Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data['Volatility'] = data['Log_Returns'].rolling(window=window_size).std() * np.sqrt(252)
    
    data.dropna(inplace=True)
    return data

data = get_data(ticker, start_date)

if data is None:
    st.error(f"No data found for **{ticker}**. Please check the symbol.")
    st.stop()

# --- 3. Model Training (The "Quant" Upgrade) ---
# We use Returns and Volatility as observed features
X = data[['Log_Returns', 'Volatility']].values

# Initialize HMM (Bishop Ch. 13)
# n_components=3 -> (Low Vol, Neutral, High Vol)
# covariance_type="full" -> Models correlation between Returns and Volatility
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
model.fit(X)

# Predict the hidden states (The "Regimes")
regimes = model.predict(X)

# --- 4. Enforce Order (Crucial for Visualization) ---
# HMM labels states randomly (e.g., State 0 might be "Crisis").
# We sort them by Volatility so State 0 is ALWAYS "Low Vol/Bull".
stats_unsorted = []
for i in range(3):
    # Calculate mean volatility for this state
    stats_unsorted.append(X[regimes == i, 1].mean())

# Get the order (e.g., [2, 0, 1] means State 2 is lowest vol)
order = np.argsort(stats_unsorted)
mapping = {old_label: new_label for new_label, old_label in enumerate(order)}
ordered_regimes = np.array([mapping[label] for label in regimes])

data['Regime'] = ordered_regimes

# --- 5. Visualization (Regime Chart) ---
fig = go.Figure()
colors = ['green', 'orange', 'red']
regime_names = ['Low Vol (Bull)', 'Neutral', 'High Vol (Bear)']

for regime_id in range(3):
    mask = data['Regime'] == regime_id
    fig.add_trace(go.Scatter(
        x=data.index[mask], y=data['Adj Close'][mask],
        mode='markers', name=regime_names[regime_id],
        marker=dict(color=colors[regime_id], size=4)
    ))

fig.add_trace(go.Scatter(
    x=data.index, y=data['Adj Close'],
    mode='lines', line=dict(color='gray', width=1),
    opacity=0.3, showlegend=False
))

fig.update_layout(title=f"Price Regimes: {ticker} (HMM Sorted)", template="plotly_dark", hovermode="x")
st.plotly_chart(fig, use_container_width=True)

# --- 6. The Transition Matrix (New Feature) ---
st.subheader("🔁 Regime Transition Matrix")
st.markdown("This matrix shows the probability of switching from one state to another (or staying in the same state).")

# Get the transition matrix from the model and reorder it to match our sorted regimes
trans_mat = model.transmat_
# Reorder rows and columns to match (0=Low, 1=Neutral, 2=High)
trans_mat_sorted = trans_mat[order][:, order]

fig_mat = px.imshow(
    trans_mat_sorted,
    labels=dict(x="To Regime", y="From Regime", color="Probability"),
    x=regime_names,
    y=regime_names,
    text_auto='.2f',
    color_continuous_scale='Blues'
)
fig_mat.update_layout(template="plotly_dark")
st.plotly_chart(fig_mat, use_container_width=True)

# --- 7. Stats Table ---
st.subheader("Regime Statistics")
stats = data.groupby('Regime')[['Log_Returns', 'Volatility']].mean()
stats.index = regime_names
stats.columns = ['Mean Daily Return', 'Mean Annualized Vol']
st.table(stats)