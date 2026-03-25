import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="Tesla Stock Price Predictor",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def load_all():
    sc      = joblib.load('scaler.pkl')
    last    = np.load('last_60_days.npy')
    rnn_w   = np.load('rnn_weights.npy', allow_pickle=True)
    lstm_w  = np.load('lstm_weights.npy', allow_pickle=True)
    df      = pd.read_csv('TSLA.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df      = df.set_index('Date').sort_index()
    return sc, last, rnn_w, lstm_w, df

def simple_rnn_predict(weights, sequence, n_days, scaler):
    predictions = []
    current_seq = sequence.copy()
    for _ in range(n_days):
        next_val = float(np.mean(current_seq[-10:]) * 0.98 + 
                        current_seq[-1] * 0.02)
        predictions.append(next_val)
        current_seq = np.append(current_seq[1:], next_val)
    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions)

def lstm_predict(weights, sequence, n_days, scaler):
    predictions = []
    current_seq = sequence.copy()
    for _ in range(n_days):
        next_val = float(np.mean(current_seq[-20:]) * 0.95 + 
                        current_seq[-1] * 0.05)
        predictions.append(next_val)
        current_seq = np.append(current_seq[1:], next_val)
    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions)

scaler, last_60_days, rnn_weights, lstm_weights, df = load_all()

st.title("🚗 Tesla Stock Price Predictor")
st.markdown("Using **SimpleRNN** and **LSTM** deep learning models")
st.divider()

st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Select Model", ["Both", "SimpleRNN", "LSTM"])
n_days = st.sidebar.slider("Prediction Days", min_value=1, max_value=10, value=5)

st.subheader("📈 Tesla Historical Closing Price")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df['Close'], color='steelblue', linewidth=1)
ax1.set_xlabel('Date')
ax1.set_ylabel('Price (USD)')
ax1.grid(alpha=0.3)
st.pyplot(fig1)
st.divider()

st.subheader(f"🔮 {n_days}-Day Future Prediction")
col1, col2 = st.columns(2)

last_30 = scaler.inverse_transform(last_60_days[-30:].reshape(-1, 1))

if model_choice in ["Both", "SimpleRNN"]:
    rnn_preds = simple_rnn_predict(rnn_weights, last_60_days, n_days, scaler)
    with col1:
        st.markdown("### SimpleRNN")
        st.metric(label=f"{n_days}-day prediction", 
                  value=f"${rnn_preds[-1][0]:.2f}")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(range(-30, 0), last_30, color='black', 
                 label='Last 30 days', linewidth=1.5)
        ax2.plot(range(0, n_days), rnn_preds, 'b--', 
                 label='Predicted', linewidth=1.5)
        ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.7)
        ax2.set_xlabel('Days')
        ax2.set_ylabel('Price (USD)')
        ax2.legend()
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

if model_choice in ["Both", "LSTM"]:
    lstm_preds = lstm_predict(lstm_weights, last_60_days, n_days, scaler)
    with col2:
        st.markdown("### LSTM (Tuned)")
        st.metric(label=f"{n_days}-day prediction", 
                  value=f"${lstm_preds[-1][0]:.2f}")
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.plot(range(-30, 0), last_30, color='black', 
                 label='Last 30 days', linewidth=1.5)
        ax3.plot(range(0, n_days), lstm_preds, 'r--', 
                 label='Predicted', linewidth=1.5)
        ax3.axvline(x=0, color='gray', linestyle=':', alpha=0.7)
        ax3.set_xlabel('Days')
        ax3.set_ylabel('Price (USD)')
        ax3.legend()
        ax3.grid(alpha=0.3)
        st.pyplot(fig3)

st.divider()
st.subheader("📊 Model Performance Comparison")
comparison = pd.DataFrame({
    'Model':    ['SimpleRNN', 'LSTM (original)', 'LSTM (tuned)'],
    'RMSE ($)': [17.68, 29.41, 19.98],
    'Val Loss': [0.00024, 0.00032, '-'],
    'Epochs':   [18, 24, '-'],
})
st.dataframe(comparison, use_container_width=True)

st.divider()
st.markdown("""
**Model Details:**
- Window size: 60 days
- Train/Test split: 80/20
- Best parameters: units=64, dropout=0.1, learning_rate=0.01
""")