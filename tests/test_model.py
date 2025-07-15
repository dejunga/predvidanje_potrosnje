import pandas as pd
import numpy as np
import pytest
from src.model import train_and_forecast_arima


def test_constant_series_forecast():
    """Test forecasting on constant data returns constant values."""
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    df = pd.DataFrame({"Potrošnja": 100}, index=dates)
    res = train_and_forecast_arima(df, order=(1, 1, 1), periods=5)
    
    assert np.allclose(res["forecast"].values, 100, atol=1e-2)
    assert np.isfinite(res["aic"])
    assert len(res["forecast"]) == 5


def test_invalid_periods_raises_error():
    """Test that periods < 1 raises ValueError."""
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    df = pd.DataFrame({"Potrošnja": 100}, index=dates)
    
    with pytest.raises(ValueError, match="`periods` must be >= 1"):
        train_and_forecast_arima(df, periods=0)
    
    with pytest.raises(ValueError, match="`periods` must be >= 1"):
        train_and_forecast_arima(df, periods=-1)


def test_rmse_calculation_with_validation():
    """Test RMSE calculation when validation data is provided."""
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    df = pd.DataFrame({"Potrošnja": 100}, index=dates)
    
    # Create validation data for next 5 days
    valid_dates = pd.date_range("2025-01-31", periods=5, freq="D")
    valid_data = pd.Series([101, 102, 99, 100, 98], index=valid_dates)
    
    res = train_and_forecast_arima(df, periods=5, valid=valid_data)
    
    assert "rmse" in res
    assert np.isfinite(res["rmse"])
    assert res["rmse"] > 0


def test_trending_data():
    """Test model on trending consumption data."""
    dates = pd.date_range("2025-01-01", periods=50, freq="D")
    # Create upward trending data
    trend = np.linspace(100, 150, 50)
    noise = np.random.normal(0, 5, 50)
    np.random.seed(42)  # For reproducibility
    consumption = trend + noise
    
    df = pd.DataFrame({"Potrošnja": consumption}, index=dates)
    res = train_and_forecast_arima(df, order=(1, 1, 1), periods=7)
    
    assert len(res["forecast"]) == 7
    assert np.isfinite(res["aic"])
    # Forecast should continue the trend (values should be > 140)
    assert res["forecast"].iloc[-1] > 140


def test_seasonal_data():
    """Test model on seasonal consumption data."""
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    # Create seasonal pattern (weekly cycle)
    seasonal = 100 + 20 * np.sin(2 * np.pi * np.arange(60) / 7)
    
    df = pd.DataFrame({"Potrošnja": seasonal}, index=dates)
    res = train_and_forecast_arima(df, order=(2, 1, 2), periods=7)
    
    assert len(res["forecast"]) == 7
    assert np.isfinite(res["aic"])


def test_date_index_continuity():
    """Test that forecast dates are correctly generated."""
    start_date = "2025-01-15"
    dates = pd.date_range(start_date, periods=30, freq="D")
    df = pd.DataFrame({"Potrošnja": 100}, index=dates)
    
    res = train_and_forecast_arima(df, periods=5)
    
    # Check forecast starts day after last training date
    expected_start = pd.to_datetime("2025-02-14")
    assert res["forecast"].index[0] == expected_start
    
    # Check forecast dates are consecutive
    forecast_dates = res["forecast"].index
    expected_dates = pd.date_range(expected_start, periods=5, freq="D")
    pd.testing.assert_index_equal(forecast_dates, expected_dates)


def test_different_arima_orders():
    """Test different ARIMA orders work correctly."""
    dates = pd.date_range("2025-01-01", periods=40, freq="D")
    # Create AR(1) process
    np.random.seed(42)
    data = [100]
    for i in range(39):
        data.append(0.8 * data[-1] + np.random.normal(0, 10))
    
    df = pd.DataFrame({"Potrošnja": data}, index=dates)
    
    # Test different orders
    orders = [(1, 0, 0), (0, 1, 1), (2, 1, 1)]
    for order in orders:
        res = train_and_forecast_arima(df, order=order, periods=3)
        assert len(res["forecast"]) == 3
        assert np.isfinite(res["aic"])


def test_different_forecast_periods():
    """Test different forecast period lengths."""
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    df = pd.DataFrame({"Potrošnja": 100}, index=dates)
    
    periods_list = [1, 3, 7, 14]
    for periods in periods_list:
        res = train_and_forecast_arima(df, periods=periods)
        assert len(res["forecast"]) == periods
        assert np.isfinite(res["aic"])


def test_short_series():
    """Test model behavior with minimal data."""
    # ARIMA needs at least p+d+q+1 observations
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    df = pd.DataFrame({"Potrošnja": 100}, index=dates)
    
    # Use simple ARIMA(0,1,0) which needs fewer observations
    res = train_and_forecast_arima(df, order=(0, 1, 0), periods=2)
    
    assert len(res["forecast"]) == 2
    assert np.isfinite(res["aic"])


def test_missing_validation_no_rmse():
    """Test that RMSE is not calculated when no validation data provided."""
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    df = pd.DataFrame({"Potrošnja": 100}, index=dates)
    
    res = train_and_forecast_arima(df, periods=5, valid=None)
    
    assert "rmse" not in res
    assert "forecast" in res
    assert "aic" in res


def test_validation_alignment():
    """Test validation data alignment with forecast dates."""
    dates = pd.date_range("2025-01-01", periods=20, freq="D")
    df = pd.DataFrame({"Potrošnja": 100}, index=dates)
    
    # Create validation with some overlap with forecast dates
    # Training ends 2025-01-20, forecast starts 2025-01-21
    valid_dates = pd.date_range("2025-01-22", periods=3, freq="D")
    valid_data = pd.Series([105, 98, 102], index=valid_dates)
    
    res = train_and_forecast_arima(df, periods=5, valid=valid_data)
    
    # Should calculate RMSE for overlapping dates
    assert "rmse" in res
    assert np.isfinite(res["rmse"])


def test_validation_no_overlap():
    """Test validation data with no overlap returns no RMSE."""
    dates = pd.date_range("2025-01-01", periods=20, freq="D")
    df = pd.DataFrame({"Potrošnja": 100}, index=dates)
    
    # Create validation with no overlap (forecast starts 2025-01-21)
    valid_dates = pd.date_range("2025-01-30", periods=3, freq="D")
    valid_data = pd.Series([105, 98, 102], index=valid_dates)
    
    res = train_and_forecast_arima(df, periods=5, valid=valid_data)
    
    # Should not calculate RMSE when no overlap
    assert "rmse" not in res