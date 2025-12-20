# src/training/arima_predictor.py

"""
ARIMA model wrapper for time series forecasting.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class ARIMAPredictor:
    """
    ARIMA model for cryptocurrency price prediction.
    Uses auto-ARIMA to find optimal parameters.
    """
    
    def __init__(self, seasonal: bool = False, m: int = 7):
        """
        Args:
            seasonal: Whether to use seasonal ARIMA
            m: Seasonal period (e.g., 7 for weekly seasonality)
        """
        self.seasonal = seasonal
        self.m = m
        self.model = None
        self.fitted = False
        self.name = "ARIMA"
        
        logger.info(f"Initialized ARIMA predictor (seasonal={seasonal}, m={m})")
    
    def fit(self, prices: pd.Series, max_p: int = 5, max_q: int = 5) -> None:
        """
        Fit ARIMA model on price series using auto-ARIMA.
        
        Args:
            prices: Time series of prices
            max_p: Maximum AR order to test
            max_q: Maximum MA order to test
        """
        try:
            from pmdarima import auto_arima
            
            logger.info("Fitting ARIMA model...")
            logger.info(f"Price series length: {len(prices)}, Min: {prices.min():.2f}, Max: {prices.max():.2f}")
            
            self.model = auto_arima(
                prices,
                seasonal=self.seasonal,
                m=self.m if self.seasonal else 1,
                max_p=max_p,
                max_q=max_q,
                max_d=2,
                start_p=1,
                start_q=1,
                suppress_warnings=True,
                stepwise=True,
                error_action='ignore',
                trace=False
            )
            
            self.fitted = True
            logger.info(f"ARIMA model fitted successfully: {self.model.order}")
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA: {e}")
            self.fitted = False
    
    def predict(self, n_periods: int = 1) -> np.ndarray:
        """
        Forecast future prices.
        
        Args:
            n_periods: Number of periods to forecast
        
        Returns:
            Array of predictions
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        try:
            predictions = self.model.predict(n_periods=n_periods)
            logger.info(f"ARIMA generated {n_periods} predictions")
            return predictions
        
        except Exception as e:
            logger.error(f"Error in ARIMA prediction: {e}")
            return np.array([])
    
    def predict_from_sequences(
        self,
        X: np.ndarray,
        close_idx: int = -1
    ) -> np.ndarray:
        """
        Generate predictions for sequences (to match baseline interface).
        
        For each sequence, fit ARIMA on that sequence and predict next value.
        This is slower but necessary for fair comparison.
        
        Args:
            X: Input sequences of shape (samples, timesteps, features)
            close_idx: Index of close price in features
        
        Returns:
            Predictions array of shape (samples,)
        """
        predictions = []
        
        for i, sample in enumerate(X):
            prices = pd.Series(sample[:, close_idx])
            
            try:
                # Fit ARIMA on this sequence
                from pmdarima import auto_arima
                
                model = auto_arima(
                    prices,
                    seasonal=False,
                    max_p=3,
                    max_q=3,
                    max_d=1,
                    suppress_warnings=True,
                    stepwise=True,
                    error_action='ignore'
                )
                
                # Predict next value
                pred = model.predict(n_periods=1)[0]
                predictions.append(pred)
                
            except Exception as e:
                # If ARIMA fails, fall back to naive prediction
                logger.warning(f"ARIMA failed for sample {i}, using naive fallback: {e}")
                predictions.append(prices.iloc[-1])
        
        predictions = np.array(predictions)
        logger.info(f"ARIMA generated {len(predictions)} predictions from sequences")
        return predictions
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # Directional accuracy
        y_true_direction = np.sign(np.diff(y_true, prepend=y_true[0]))
        y_pred_direction = np.sign(np.diff(y_pred, prepend=y_pred[0]))
        dir_acc = np.mean(y_true_direction == y_pred_direction)
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'directional_accuracy': float(dir_acc)
        }
