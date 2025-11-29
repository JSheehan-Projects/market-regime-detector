import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm

# --- Page Config ---
st.set_page_config(
    page_title="Machine Learning Regime Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Sidebar Configuration ---
st.sidebar.header("Model Settings")

# Ticker Input
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
window_size = st.sidebar.slider(
    "Volatility Window (Days)", 
    min_value=5, 
    max_value=60, 
    value=20,
    help="Each datapoint calculates the annualised volatility based on the returns of this many previous days (Rolling Window)."
)
st.sidebar.markdown("---")
st.sidebar.header("Algorithm Selection")
model_type = st.sidebar.radio(
    "Choose Model:",
    ["Hidden Markov Model (HMM)", "Gaussian Mixture Model (GMM)"]
)
st.sidebar.markdown("---")
use_cblind = st.sidebar.checkbox("Colourblind Mode", value=True)

# Dynamic Description based on selection
if model_type == "Hidden Markov Model (HMM)":
    st.title(f"📊 Market Regimes (HMM): {ticker}")
    st.markdown("""
    **Technique:** Bishop Chapter 13 (Sequential Data).  
    **Concept:** Clusters days based on their Returns/Volatility profile, while also assuming markets have "memory." The probability of being in a certain market state today depends on yesterday's market state.  
    **Best For:** Capturing sustained trends and filtering out daily noise.
    """)
else:
    st.title(f"📊 Market Regimes (GMM): {ticker}")
    st.markdown("""
    **Technique:** Bishop Chapter 9 (Mixture Models).  
    **Concept:** Treats every day as an independent event. Clusters days based purely on their Returns/Volatility profile.  
    **Best For:** Understanding the distribution of returns without assuming time-dependence.
    """)

# --- 2. Data Engine ---
@st.cache_data
def get_data(ticker, start, window):
    try:
        data = yf.download(ticker, start=start)
    except Exception:
        return None
    
    if data.empty:
        return None

    # Flatten MultiIndex if necessary
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Price Column Fallback
    if 'Adj Close' not in data.columns and 'Close' in data.columns:
        data['Adj Close'] = data['Close']
    
    if 'Adj Close' not in data.columns:
        return None

    # Feature Engineering
    data['Log_Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data['Volatility'] = data['Log_Returns'].rolling(window=window).std() * np.sqrt(252)
    
    data.dropna(inplace=True)
    return data

data = get_data(ticker, start_date, window_size)

if data is None:
    st.error(f"Error: Could not fetch data for {ticker}. Check symbol or internet connection.")
    st.stop()

# --- 3. Modeling Engine ---
X = data[['Log_Returns', 'Volatility']].values

# We need to store the model to extract params later
model = None
regimes = None

if model_type == "Hidden Markov Model (HMM)":
    # HMM Training
    model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X)
    regimes = model.predict(X)

else:
    # GMM Training
    model = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
    model.fit(X)
    regimes = model.predict(X)

# --- 4. Regime Sorting  ---
# We sort so that Regime 0 is always Low Vol, Regime 2 is High Vol
# (even though the regimes are multidimentional, we can only really sort by one average)
stats_unsorted = []
for i in range(3):
    # Mean Volatility for this cluster
    stats_unsorted.append(X[regimes == i, 1].mean())

order = np.argsort(stats_unsorted)
mapping = {old_label: new_label for new_label, old_label in enumerate(order)}
ordered_regimes = np.array([mapping[label] for label in regimes])
data['Regime'] = ordered_regimes

# --- 5. Visualisation ---
if use_cblind:
    # Colourblind Safe Palette (Blue vs Orange)
    # Low Vol: Blue, Neutral: Grey, High Vol: Orange
    colors = ['#377eb8', '#999999', '#ff7f0e'] 
else:
    # Standard Finance Palette (Green vs Red)
    # Low Vol: Green, Neutral: Grey, High Vol: Red
    # EDIT THESE HEX CODES TO CHANGE YOUR STANDARD COLOURS
    colors = ['#2ca02c', '#7f7f7f', '#d62728']

regime_names = ['Low Volatility', 'Neutral Volatility', 'High Volatility']

# Main Chart
fig = go.Figure()

# Plot price line (dimmed)
fig.add_trace(go.Scatter(
    x=data.index, y=data['Adj Close'],
    mode='lines', line=dict(color='gray', width=1),
    opacity=0.3, showlegend=False
))

# Plot Regime Markers
for regime_id in range(3):
    mask = data['Regime'] == regime_id
    fig.add_trace(go.Scatter(
        x=data.index[mask], y=data['Adj Close'][mask],
        mode='markers', name=regime_names[regime_id],
        marker=dict(color=colors[regime_id], size=4)
    ))

fig.update_layout(
    title=f"Price Regimes ({model_type})",
    template="plotly_dark",
    hovermode="x",
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# --- 6. Model Specific Insights ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Regime Statistics")
    stats = data.groupby('Regime')[['Log_Returns', 'Volatility']].mean()
    stats.index = regime_names
    stats.columns = ['Avg Daily Return', 'Avg Annual Vol']
    st.table(stats.style.format("{:.4f}"))

with col2:
    if model_type == "Hidden Markov Model (HMM)":
        st.subheader("Transition Matrix")
        st.write("Probability of switching from Row state to Column state.")
        
        # Extract and Sort Transition Matrix
        trans_mat = model.transmat_
        trans_mat_sorted = trans_mat[order][:, order]
        
        fig_mat = px.imshow(
            trans_mat_sorted,
            labels=dict(x="To State", y="From State", color="Prob"),
            x=regime_names, y=regime_names,
            text_auto='.2f', color_continuous_scale='Blues'
        )
        fig_mat.update_traces(
            hovertemplate="From State: %{y}<br>To State: %{x}<br>Probability: %{z:.4f}<extra></extra>"
        )
        st.plotly_chart(fig_mat, use_container_width=True)
        
    else:
        st.subheader("GMM Component Weights")
        st.write("Overall proportion of time spent in each regime.")
        
        # Extract and Sort Weights
        weights = model.weights_
        weights_sorted = weights[order]
        
        fig_pie = px.pie(
            values=weights_sorted, 
            names=regime_names,
            color=regime_names,
            color_discrete_map={
                'Low Vol': '#2ca02c', 
                'Neutral': '#ff7f0e', 
                'High Vol': '#d62728'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# --- 7. Interview Cheatsheet (Hidden Expander) ---
with st.expander("ℹ️  Model Comparison"):
    st.markdown("""
    **Why comparing GMM and HMM matters:**
    
    1.  **Independent vs. Sequential:** 
        * **GMM** asks: *"Does today look like a crash?"* 
        * **HMM** asks: *"While also taking into account yesterday's situation, does today look like a crash?"*
        
    2.  **Noise Filtering:** 
        * Toggle to **GMM**: You will see the regimes "flicker" rapidly during transition periods.  
        * Toggle to **HMM**: The regimes are stickier. The Transition Matrix imposes a "penalty" for switching states too quickly, smoothing the signal.
        
    3.  **The Maths:** 
        * GMM uses **EM Algorithm** to maximise likelihood of independent points.  
        * HMM uses **Baum-Welch (EM)** for parameters and **Viterbi** for state decoding.
    """)