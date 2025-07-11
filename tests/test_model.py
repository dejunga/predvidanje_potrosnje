import pandas as pd
import numpy as np
from src.model import train_and_forecast_arima


def test_constant_series_forecast():
    # Create 30 days of constant consumption
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    df = pd.DataFrame({"Potrošnja": 100}, index=dates)
    res = train_and_forecast_arima(df, order=(1, 1, 1), periods=5)
    # All forecasts should equal 100
    assert np.allclose(res["forecast"].values, 100)
    # AIC should be a finite number
    assert np.isfinite(res["aic"])
