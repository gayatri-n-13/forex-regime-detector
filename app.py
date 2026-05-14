import streamlit as st
import plotly.express as px

from data_loader import load_data
from feature_engineering import create_features
from model import detect_regimes

st.title("Forex Market Regime Detection")

# Load data
data = load_data()

# Create features
data = create_features(data)

# Detect regimes
data = detect_regimes(data)

st.write(data.tail())

# Plot chart
fig = px.scatter(
    x=data.index,
    y=data[('Close', 'EURUSD=X')],
    color=data['Regime'],
    title='Forex Market Regimes'
)

st.plotly_chart(fig)