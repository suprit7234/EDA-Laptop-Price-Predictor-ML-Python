import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page config
st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

# Absolute paths (no FileNotFoundError)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPE_PATH = os.path.join(BASE_DIR, "pipe.pkl")
DF_PATH = os.path.join(BASE_DIR, "df.pkl")

# Load model and dataframe (cached)
@st.cache_resource
def load_artifacts():
    with open(PIPE_PATH, "rb") as f:
        pipe = pickle.load(f)
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)
    return pipe, df

pipe, df = load_artifacts()

# App UI
st.title("💻 Laptop Price Predictor")

# Inputs
company = st.selectbox('Brand', sorted(df['Company'].dropna().unique()))
laptop_type = st.selectbox('Type', sorted(df['TypeName'].dropna().unique()))
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop (kg)', min_value=0.5, max_value=5.0, value=1.5, step=0.1)

touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS Display', ['No', 'Yes'])

screen_size = st.slider('Screen size (in inches)', 10.0, 18.0, 13.3)

resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080','1366x768','1600x900','3840x2160',
     '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440']
)

cpu = st.selectbox('CPU Brand', sorted(df['Cpu brand'].dropna().unique()))
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU Brand', sorted(df['Gpu brand'].dropna().unique()))
os_name = st.selectbox('Operating System', sorted(df['os'].dropna().unique()))

# Prediction
if st.button('🔮 Predict Price'):
    # Binary encoding
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    # PPI calculation
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size

    # Create DataFrame query (MUST match training columns)
    query = pd.DataFrame([{
        'Company': company,
        'TypeName': laptop_type,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen_val,
        'Ips': ips_val,
        'Inches': screen_size,
        'ppi': ppi,
        'Cpu brand': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu brand': gpu,
        'os': os_name
    }])

    # Align columns exactly as training data (extra safety)
    X_cols = df.drop("Price", axis=1).columns
    query = query.reindex(columns=X_cols)

    # Predict (model trained on log(price))
    pred_log_price = pipe.predict(query)[0]
    price = int(np.expm1(pred_log_price))

    st.success(f"💰 Estimated Price: ₹ {price:,}")
    st.caption("Note: This is an ML estimate; actual market price may vary.")



#streamlit run Desktop\Laptop-Pred-ML\Lap-app.py