# src/training/models/baseline_models.py

"""
Baseline Forecasting Models for Cryptocurrency Price Prediction

Simple statistical models for benchmarking deep learning models:
- Naive forecast (last value persistence) - simplest baseline
- Moving Average (MA) - smoothed historical average
- Exponential Moving Average (EMA) - weighted recent values
- Drift forecast - linear trend extrapolation
- Seasonal Naive - captures periodic patterns

These models provide important benchmarks to validate that complex models
actually improve over simple statistical methods.
"""

import logging
from typing import Optional, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class BaselineModels:
    """Collection of baseline forecasting models"""
    
    def __init__(self):
        """Initialize baseline models"""
        self.logger = self._setup_logger()
        self.logger.info("BaselineModels initialized")
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def naive_forecast(
        self,
        prices: pd.Series,
        horizon: int = 1
    ) -> np.ndarray:
        """
        Naive forecast: persistence model (last value)
        Baseline benchmark - simplest possible forecast
        
        Args:
            prices: Historical prices
            horizon: Forecast horizon
            
        Returns:
            Array of predictions (all equal to last price)
        """
        if len(prices) == 0:
            return np.array([])
        
        last_value = prices.iloc[-1]
        predictions = np.full(horizon, last_value)
        
        self.logger.debug(f"Naive forecast: {last_value:.2f} × {horizon}")
        return predictions
    
    def moving_average_forecast(
        self,
        prices: pd.Series,
        window: int = 20,
        horizon: int = 1
    ) -> np.ndarray:
        """
        Simple Moving Average forecast
        
        Args:
            prices: Historical prices
            window: MA window size
            horizon: Forecast horizon
            
        Returns:
            Array of predictions (all equal to MA)
        """
        if len(prices) < window:
            self.logger.warning(f"Insufficient data for MA({window}), using naive")
            return self.naive_forecast(prices, horizon)
        
        ma = prices.rolling(window=window).mean().iloc[-1]
        predictions = np.full(horizon, ma)
        
        self.logger.debug(f"MA({window}) forecast: {ma:.2f}")
        return predictions
    
    def exponential_moving_average_forecast(
        self,
        prices: pd.Series,
        span: int = 20,
        horizon: int = 1
    ) -> np.ndarray:
        """
        Exponential Moving Average forecast
        Gives more weight to recent observations
        
        Args:
            prices: Historical prices
            span: EMA span
            horizon: Forecast horizon
            
        Returns:
            Array of predictions (all equal to EMA)
        """
        if len(prices) < 2:
            return self.naive_forecast(prices, horizon)
        
        ema = prices.ewm(span=span, adjust=False).mean().iloc[-1]
        predictions = np.full(horizon, ema)
        
        self.logger.debug(f"EMA({span}) forecast: {ema:.2f}")
        return predictions
    
    def drift_forecast(
        self,
        prices: pd.Series,
        horizon: int = 1
    ) -> np.ndarray:
        """
        Drift forecast: extrapolate linear trend
        Captures overall directional movement
        
        Args:
            prices: Historical prices
            horizon: Forecast horizon
            
        Returns:
            Array of predictions following trend
        """
        if len(prices) < 2:
            return self.naive_forecast(prices, horizon)
        
        # Average change per period
        drift = (prices.iloc[-1] - prices.iloc[0]) / (len(prices) - 1)
        
        # Extrapolate
        last_value = prices.iloc[-1]
        predictions = np.array([last_value + (drift * h) for h in range(1, horizon + 1)])
        
        self.logger.debug(f"Drift forecast: trend={drift:.2f}/day")
        return predictions
    
    def seasonal_naive_forecast(
        self,
        prices: pd.Series,
        season_length: int = 7,
        horizon: int = 1
    ) -> np.ndarray:
        """
        Seasonal naive: use value from same day in previous period
        Useful for data with weekly/monthly patterns
        
        Args:
            prices: Historical prices
            season_length: Length of seasonal period (default: 7 for weekly)
            horizon: Forecast horizon
            
        Returns:
            Array of predictions
        """
        if len(prices) < season_length:
            return self.naive_forecast(prices, horizon)
        
        predictions = []
        for h in range(horizon):
            lookback_idx = -season_length - h
            if abs(lookback_idx) <= len(prices):
                predictions.append(prices.iloc[lookback_idx])
            else:
                predictions.append(prices.iloc[-1])
        
        return np.array(predictions)
    
    def evaluate(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray
    ) -> dict:
        """
        Evaluate predictions against actuals
        
        Args:
            actuals: True values
            predictions: Predicted values
            
        Returns:
            Dict of metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100 if np.all(actuals != 0) else 0
        r2 = r2_score(actuals, predictions)
        
        # Directional accuracy
        if len(actuals) > 1:
            actual_direction = np.sign(np.diff(actuals, prepend=actuals[0]))
            pred_direction = np.sign(np.diff(predictions, prepend=predictions[0]))
            dir_acc = np.mean(actual_direction == pred_direction)
        else:
            dir_acc = 0.5
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2_score': float(r2),
            'directional_accuracy': float(dir_acc)
        }
        
        return metrics
    
    def get_all_forecasts(
        self,
        prices: pd.Series,
        horizon: int = 7
    ) -> dict:
        """
        Generate forecasts from all baseline models
        
        Args:
            prices: Historical prices
            horizon: Forecast horizon
            
        Returns:
            Dict mapping model name to predictions
        """
        forecasts = {
            'Naive': self.naive_forecast(prices, horizon),
            'MA(20)': self.moving_average_forecast(prices, window=20, horizon=horizon),
            'EMA(20)': self.exponential_moving_average_forecast(prices, span=20, horizon=horizon),
            'Drift': self.drift_forecast(prices, horizon),
            'Seasonal': self.seasonal_naive_forecast(prices, season_length=7, horizon=horizon)
        }
        
        self.logger.info(f"Generated {len(forecasts)} baseline forecasts")
        return forecasts
