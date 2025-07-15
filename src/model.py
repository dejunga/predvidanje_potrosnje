import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from typing import Optional, Tuple, Dict


def train_and_forecast_arima(
    df: pd.DataFrame,
    order: Tuple[int, int, int] = (1, 1, 1),
    periods: int = 7,
    valid: Optional[pd.Series] = None,
) -> Dict[str, object]:
    """
    Fit ARIMA on df['Potrošnja'] (DatetimeIndex) and forecast next `periods` days.
    If `valid` is provided (must align with forecast dates), compute RMSE.
    Returns a dict with:
      - 'forecast': pd.Series (indexed by forecast dates)
      - 'aic': float
      - 'rmse': float (if valid provided)
    """
    if periods < 1:
        raise ValueError("`periods` must be >= 1")

    # Fit model
    model = sm.tsa.ARIMA(df["Potrošnja"], order=order).fit()

    # Forecast
    pred = model.get_forecast(steps=periods)
    forecast = pred.predicted_mean
    # Attach proper date index
    last_date = pd.to_datetime(df.index.max())
    start_date = last_date + pd.Timedelta(days=1)
    forecast.index = pd.date_range(start=start_date, periods=periods, freq="D")

    result = {"forecast": forecast, "aic": model.aic}

    # Optional RMSE
    if valid is not None:
        # align valid to forecast index
        valid_aligned = valid.reindex(forecast.index)
        # Remove NaN values for RMSE calculation
        mask = ~valid_aligned.isna()
        if mask.sum() > 0:  # Only calculate if we have overlapping data
            rmse = np.sqrt(mean_squared_error(valid_aligned[mask], forecast[mask]))
            result["rmse"] = rmse

    return result
