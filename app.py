# app.py
import sys, os

sys.path.append(os.path.abspath("."))  # da Python može naći src/model.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
from sklearn.metrics import mean_absolute_error
import plotly.express as px

mpl.rcParams.update(
    {
        "axes.titlesize": 5,
        "axes.labelsize": 5,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 10,
    }
)

from src.model import train_and_forecast_arima
from sklearn.metrics import mean_squared_error


st.set_page_config(page_title="Predviđanje potrošnje", layout="wide")
st.title("📈 Predviđanje potrošnje (ARIMA)")

# ─── 1) Create Tabs ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["1️⃣ Upload & Preview", "2️⃣ Model Parameters", "3️⃣ Forecast & Metrics"]
)

# ─── 2) Tab 1: Upload & Preview ─────────────────────────────────
with tab1:
    uploaded_file = st.file_uploader(
        "Upload CSV s kolonama `Datum` i `Potrošnja`", type=["csv"]
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, skipinitialspace=True)
            df.columns = (
                df.columns.str.strip()
                .str.capitalize()
                .str.replace("Potrosnja", "Potrošnja")
            )
            df["Datum"] = pd.to_datetime(df["Datum"], format="%Y-%m-%d", exact=True)
            df.set_index("Datum", inplace=True)
        except Exception as e:
            st.error(f"Greška pri čitanju CSV-a: {e}")
            st.stop()

        st.subheader("Povijesni podaci — Tablica")
        st.dataframe(df)
        st.subheader("Povijesni podaci — Graf")
        st.line_chart(df["Potrošnja"])

# ─── 3) Tab 2: Model Parameters ────────────────────────────────
with tab2:
    st.subheader("Odabir parametara modela")
    # reuse the same `uploaded_file` check so train/valid exist
    if uploaded_file is None:
        st.info("Prvo uploadajte CSV u kartici 1.")
        st.stop()
    periods = st.number_input("Broj dana za predviđanje", 1, 60, 7)
    p = st.number_input("p (AR order)", 0, 5, 1)
    d = st.number_input("d (I order)", 0, 2, 1)
    q = st.number_input("q (MA order)", 0, 5, 1)
    order = (p, d, q)
    submit = st.button("Submit")

    # split *after* valid CSV
    n = len(df)
    split = int(n * 0.8)
    train = df.iloc[:split].copy()
    valid = df.iloc[split:].copy()

# ─── 4) Tab 3: Forecast & Metrics ──────────────────────────────
with tab3:
    if not submit:
        st.info('Postavite parametre u kartici 2 i kliknite "Submit"')
        st.stop()

    # parameter‐size guard
    total_params = p + d + q + 1
    if total_params >= len(train) / 2:
        st.error(
            f"Parametri (p+d+q+1={total_params}) su preveliki za {len(train)} točaka."
        )
        st.stop()

    # 1) Train & forecast
    with st.spinner("Treniram ARIMA i predviđam..."):
        try:
            res = train_and_forecast_arima(
                train, order=order, periods=periods, valid=valid["Potrošnja"]
            )
        except np.linalg.LinAlgError:
            st.error(f"ARIMA{order} ne može konvergirati. Smanjite p, d ili q.")
            st.stop()

    forecast = res["forecast"]
    aic = res["aic"]
    rmse = res.get("rmse", np.nan)
    mae = mean_absolute_error(valid["Potrošnja"], forecast[: len(valid)])
    mape = (
        abs((valid["Potrošnja"] - forecast[: len(valid)]) / valid["Potrošnja"])
    ).mean() * 100

    # 2) Interactive Plot with Plotly
    import plotly.express as px

    future_idx = pd.date_range(
        start=df.index.max() + pd.Timedelta(days=1), periods=periods, freq="D"
    )
    df_plot = (
        pd.concat(
            [
                train["Potrošnja"].rename("Train"),
                valid["Potrošnja"].rename("Valid"),
                pd.Series(forecast.values, index=future_idx, name="Forecast"),
            ],
            axis=1,
        )
        .reset_index()
        .melt(id_vars="index", var_name="Serija", value_name="Potrošnja")
        .rename(columns={"index": "Datum"})
    )
    fig = px.line(
        df_plot,
        x="Datum",
        y="Potrošnja",
        color="Serija",
        title="Potrošnja: povijest i ARIMA predviđanje",
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3) Download button for forecast CSV
    csv = (
        forecast.reset_index()
        .rename(columns={0: "Potrošnja"})
        .to_csv(index=False)
        .encode("utf-8")
    )

    # 4) Metrics
    st.subheader("Metrike")
    st.markdown(f"**AIC:** {aic:.2f}  •  **RMSE:** {rmse:.2f}")
    st.markdown(f"**MAE:** {mae:.2f}  •  **MAPE:** {mape:.1f}%")

    # 5) Tablični pregled predikcija
    st.subheader(f"Predikcije (prvih {periods} dana)")
    st.dataframe(
        forecast.reset_index().rename(columns={"index": "Datum", 0: "Potrošnja"})
    )
