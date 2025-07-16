"""Integration tests for the consumption forecast application."""
import os
import pandas as pd
import numpy as np
import pytest
from src.model import train_and_forecast_arima


@pytest.fixture
def sample_data():
    """Generate sample consumption data for testing."""
    dates = pd.date_range("2023-01-01", periods=90, freq="D")
    # Create realistic consumption pattern
    base_consumption = 200
    weekly_pattern = 50 * np.sin(2 * np.pi * np.arange(90) / 7)
    trend = np.linspace(0, 50, 90)
    noise = np.random.normal(0, 20, 90)
    
    consumption = base_consumption + weekly_pattern + trend + noise
    consumption = np.maximum(consumption, 0)  # Ensure non-negative
    
    return pd.DataFrame({"Potrošnja": consumption}, index=dates)


@pytest.fixture
def train_valid_split(sample_data):
    """Split sample data into training and validation sets."""
    split_idx = int(len(sample_data) * 0.8)
    train = sample_data.iloc[:split_idx]
    valid = sample_data.iloc[split_idx:]
    return train, valid


class TestFullWorkflow:
    """Test the complete forecast workflow."""
    
    def test_end_to_end_forecast(self, train_valid_split):
        """Test complete forecast workflow with train/validation split."""
        train, valid = train_valid_split
        
        # Train model and generate forecast
        result = train_and_forecast_arima(
            train, 
            order=(1, 1, 1), 
            periods=7,
            valid=valid["Potrošnja"]
        )
        
        # Verify results
        assert "forecast" in result
        assert "aic" in result
        assert "rmse" in result
        
        assert len(result["forecast"]) == 7
        assert np.isfinite(result["aic"])
        assert np.isfinite(result["rmse"])
        assert result["rmse"] > 0
        
        # Verify forecast dates are correct
        expected_start = train.index[-1] + pd.Timedelta(days=1)
        assert result["forecast"].index[0] == expected_start
    
    def test_auto_arima_simulation(self, sample_data):
        """Test auto-ARIMA parameter selection (simplified version)."""
        train_size = int(len(sample_data) * 0.8)
        train = sample_data.iloc[:train_size]
        
        best_aic = float('inf')
        best_order = None
        best_result = None
        
        # Test different ARIMA orders
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        result = train_and_forecast_arima(
                            train, 
                            order=(p, d, q), 
                            periods=7
                        )
                        if result['aic'] < best_aic:
                            best_aic = result['aic']
                            best_order = (p, d, q)
                            best_result = result
                    except:
                        continue
        
        assert best_result is not None
        assert best_order is not None
        assert np.isfinite(best_aic)
    
    def test_multiple_forecast_horizons(self, sample_data):
        """Test forecasting with different horizons."""
        train_size = int(len(sample_data) * 0.8)
        train = sample_data.iloc[:train_size]
        
        horizons = [1, 3, 7, 14, 30]
        
        for horizon in horizons:
            result = train_and_forecast_arima(
                train, 
                order=(1, 1, 1), 
                periods=horizon
            )
            
            assert len(result["forecast"]) == horizon
            assert np.isfinite(result["aic"])
            
            # Check forecast values are reasonable
            assert all(result["forecast"] > 0)  # Non-negative consumption
            assert all(result["forecast"] < 2000)  # Reasonable upper bound


class TestDataQuality:
    """Test data quality and edge cases."""
    
    def test_missing_values_handling(self):
        """Test handling of missing values in input data."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        consumption = np.random.normal(200, 50, 50)
        consumption[10:15] = np.nan  # Add missing values
        
        df = pd.DataFrame({"Potrošnja": consumption}, index=dates)
        
        # Remove NaN values before training
        df_clean = df.dropna()
        
        result = train_and_forecast_arima(
            df_clean, 
            order=(1, 1, 1), 
            periods=5
        )
        
        assert len(result["forecast"]) == 5
        assert np.isfinite(result["aic"])
    
    def test_zero_values_handling(self):
        """Test handling of zero consumption values."""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        consumption = np.random.normal(200, 50, 30)
        consumption[5:10] = 0  # Add zero values
        
        df = pd.DataFrame({"Potrošnja": consumption}, index=dates)
        
        result = train_and_forecast_arima(
            df, 
            order=(1, 1, 1), 
            periods=5
        )
        
        assert len(result["forecast"]) == 5
        assert np.isfinite(result["aic"])
    
    def test_extreme_values_handling(self):
        """Test handling of extreme consumption values."""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        consumption = np.random.normal(200, 50, 30)
        consumption[15] = 5000  # Add extreme value
        
        df = pd.DataFrame({"Potrošnja": consumption}, index=dates)
        
        result = train_and_forecast_arima(
            df, 
            order=(1, 1, 1), 
            periods=5
        )
        
        assert len(result["forecast"]) == 5
        assert np.isfinite(result["aic"])


class TestPerformanceMetrics:
    """Test performance evaluation metrics."""
    
    def test_rmse_calculation(self, sample_data):
        """Test RMSE calculation accuracy."""
        train_size = int(len(sample_data) * 0.8)
        train = sample_data.iloc[:train_size]
        valid = sample_data.iloc[train_size:]
        
        result = train_and_forecast_arima(
            train, 
            order=(1, 1, 1), 
            periods=len(valid),
            valid=valid["Potrošnja"]
        )
        
        # Manual RMSE calculation for verification
        forecast_aligned = result["forecast"].reindex(valid.index)
        mask = ~forecast_aligned.isna()
        manual_rmse = np.sqrt(np.mean((valid.loc[mask, "Potrošnja"] - forecast_aligned[mask]) ** 2))
        
        assert np.isclose(result["rmse"], manual_rmse, rtol=1e-5)
    
    def test_aic_consistency(self, sample_data):
        """Test AIC consistency across runs."""
        train_size = int(len(sample_data) * 0.8)
        train = sample_data.iloc[:train_size]
        
        # Run same model multiple times
        results = []
        for _ in range(3):
            result = train_and_forecast_arima(
                train, 
                order=(1, 1, 1), 
                periods=7
            )
            results.append(result["aic"])
        
        # AIC should be identical for same model and data
        assert all(np.isclose(results[0], aic, rtol=1e-10) for aic in results)


class TestSampleDatasets:
    """Test with actual sample datasets if available."""
    
    def test_sample_datasets(self):
        """Test with available sample datasets."""
        sample_files = [
            "sample_growing_consumption.csv",
            "sample_retail_consumption.csv",
            "sample_industrial_consumption.csv"
        ]
        
        for filename in sample_files:
            filepath = f"data/{filename}"
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df.columns = df.columns.str.strip().str.capitalize().str.replace("Potrosnja", "Potrošnja")
                df["Datum"] = pd.to_datetime(df["Datum"])
                df.set_index("Datum", inplace=True)
                
                # Test forecasting
                result = train_and_forecast_arima(
                    df, 
                    order=(1, 1, 1), 
                    periods=7
                )
                
                assert len(result["forecast"]) == 7
                assert np.isfinite(result["aic"])
                assert all(result["forecast"] >= 0)  # Non-negative consumption


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({"Potrošnja": [100, 101, 99, 102, 98]}, index=dates)
        
        # Should work with simple model
        result = train_and_forecast_arima(df, order=(0, 1, 0), periods=3)
        assert len(result["forecast"]) == 3
        
        # Complex model might fail with insufficient data
        with pytest.raises(Exception):
            train_and_forecast_arima(df, order=(3, 2, 3), periods=3)
    
    def test_invalid_order_parameters(self):
        """Test invalid ARIMA order parameters."""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        df = pd.DataFrame({"Potrošnja": 100}, index=dates)
        
        # Negative parameters should raise error
        with pytest.raises(Exception):
            train_and_forecast_arima(df, order=(-1, 1, 1), periods=5)