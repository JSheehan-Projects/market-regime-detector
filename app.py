import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture

# --- Configuration ---
st.set_page_config(page_title="Market Regime Detector", layout="wide")

# --- 1. Sidebar & Inputs ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
window_size = st.sidebar.slider("Volatility Lookback (Days)", 5, 60, 20)

st.title(f"📊 Market Regime Detection: {ticker}")
st.markdown("""
**Methodology:** Uses a **Gaussian Mixture Model (GMM)** (Bishop, Ch. 9) to cluster trading days into latent 'Regimes' based on Returns and Volatility.
This allows us to visualize when the market shifts from 'Calm' to 'Volatile' states dynamically.
""")

# --- 2. Data Fetching & Processing ---
@st.cache_data
def get_data(ticker, start):
    data = yf.download(ticker, start=start)
    
    if data.empty:
        return None

    # --- THE FIX: Flatten MultiIndex columns ---
    # If yfinance returns columns like ('Adj Close', 'SPY'), this flattens them to just 'Adj Close'
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Some versions of yfinance might label it 'Close' if auto_adjust=True is on
    # This block ensures we use the best available price column
    if 'Adj Close' not in data.columns and 'Close' in data.columns:
        data['Adj Close'] = data['Close']

    # Calculate Log Returns
    data['Log_Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    
    # Calculate Rolling Volatility (Annualized)
    data['Volatility'] = data['Log_Returns'].rolling(window=window_size).std() * np.sqrt(252)
    
    data.dropna(inplace=True)
    return data

data = get_data(ticker, start_date)

if data is None:
    st.error("No data found. Please check the ticker.")
    st.stop()

# --- 3. Model Training (The "Quant" Part) ---
# We use Returns and Volatility as features to cluster the days
X = data[['Log_Returns', 'Volatility']].values

# Fit the GMM (Expectation-Maximization)
# We assume 3 regimes: Low Vol, Medium Vol, High Vol
model = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
model.fit(X)

# Predict the latent state for each day
regimes = model.predict(X)

# --- 4. Enforce Order (0=Low Vol, 2=High Vol) ---
# The model labels clusters randomly (e.g., State 0 might be High Vol).
# We sort them so State 0 is always the lowest volatility state.
vol_means = [X[regimes == i, 1].mean() for i in range(3)]
order = np.argsort(vol_means) # Indices that would sort the array
mapping = {old_label: new_label for new_label, old_label in enumerate(order)}
ordered_regimes = np.array([mapping[label] for label in regimes])

data['Regime'] = ordered_regimes

# --- 5. Visualization (Plotly) ---
# Create the main price chart colored by Regime
fig = go.Figure()

colors = ['green', 'orange', 'red']
regime_names = ['Low Vol / Bull', 'Neutral / Transition', 'High Vol / Bear']

# We plot segments effectively by grouping
for regime_id in range(3):
    # Mask for the current regime
    mask = data['Regime'] == regime_id
    # We plot markers to handle non-contiguous segments cleanly
    fig.add_trace(go.Scatter(
        x=data.index[mask],
        y=data['Adj Close'][mask],
        mode='markers', # Markers are safer for scattered regimes than lines
        name=regime_names[regime_id],
        marker=dict(color=colors[regime_id], size=4)
    ))

# Add a line for continuity (optional, translucent)
fig.add_trace(go.Scatter(
    x=data.index, y=data['Adj Close'],
    mode='lines', line=dict(color='gray', width=1),
    opacity=0.3, showlegend=False
))

fig.update_layout(title=f"Price Regimes for {ticker}", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# --- 6. Stats Table ---
st.subheader("Regime Statistics")
stats = data.groupby('Regime')[['Log_Returns', 'Volatility']].mean()
stats.index = regime_names
stats.columns = ['Mean Daily Return', 'Mean Annualized Vol']
st.table(stats)

# --- 7. Bishop Context ---
with st.expander("See the Math (Reference: Bishop Ch. 9)"):
    st.latex(r"""
    p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)
    """)
    st.write("""
    We model the market features $x$ (Returns, Volatility) as a mixture of $K=3$ Gaussian distributions.
    The model learns the parameters $\mu$ (means), $\Sigma$ (covariances), and $\pi$ (mixing coefficients) 
    using the **Expectation-Maximization (EM)** algorithm to maximize the likelihood of the observed data.
    """)