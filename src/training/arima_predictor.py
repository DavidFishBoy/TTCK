

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class ARIMAPredictor:
    
    def __init__(self, seasonal: bool = False, m: int = 7):
        self.seasonal = seasonal
        self.m = m
        self.model = None
        self.fitted = False
        self.name = "ARIMA"
        
        logger.info(f"Initialized ARIMA predictor (seasonal={seasonal}, m={m})")
    
    def fit(self, prices: pd.Series, max_p: int = 5, max_q: int = 5) -> None:
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
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        try:
            predictions = self.model.predict(n_periods=n_periods)
            logger.info(f"ARIMA generated {n_periods} predictions")
            return predictions
        
        except Exception as e:
            logger.error(f"Error in ARIMA prediction: {e}")
            return np.array([])
    
    def predict_log_return_future(self, prices: np.ndarray, horizon: int = 5) -> np.ndarray:
        log_returns = np.diff(np.log(prices + 1e-8))
        
        ar_coef = 0.6
        
        predicted_returns = []
        last_return = log_returns[-1]
        
        for _ in range(horizon):
            next_return = ar_coef * last_return
            predicted_returns.append(next_return)
            last_return = next_return
        
        return np.array(predicted_returns)
    
    def predict_future_prices(self, prices: np.ndarray, horizon: int = 5) -> np.ndarray:
        predicted_returns = self.predict_log_return_future(prices, horizon)
        last_price = prices[-1]
        
        future_prices = []
        current_log_price = np.log(last_price)
        
        for r in predicted_returns:
            current_log_price += r
            future_prices.append(np.exp(current_log_price))
        
        return np.array(future_prices)
    
    def predict_from_sequences(
        self,
        X: np.ndarray,
        close_idx: int = -1
    ) -> np.ndarray:
        predictions = []
        
        for i, sample in enumerate(X):
            prices = pd.Series(sample[:, close_idx])
            
            try:
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
                
                pred = model.predict(n_periods=1)[0]
                predictions.append(pred)
                
            except Exception as e:
                logger.warning(f"ARIMA failed for sample {i}, using naive fallback: {e}")
                predictions.append(prices.iloc[-1])
        
        predictions = np.array(predictions)
        logger.info(f"ARIMA generated {len(predictions)} predictions from sequences")
        return predictions
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        y_true_direction = np.sign(np.diff(y_true, prepend=y_true[0]))
        y_pred_direction = np.sign(np.diff(y_pred, prepend=y_pred[0]))
        dir_acc = np.mean(y_true_direction == y_pred_direction)
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'directional_accuracy': float(dir_acc)
        }
    
    @staticmethod
    def evaluate_log_return(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true_returns = np.diff(np.log(y_true + 1e-8))
        y_pred_returns = np.diff(np.log(y_pred + 1e-8))
        
        mae = np.mean(np.abs(y_true_returns - y_pred_returns))
        rmse = np.sqrt(np.mean((y_true_returns - y_pred_returns) ** 2))
        
        y_true_dir = np.sign(y_true_returns)
        y_pred_dir = np.sign(y_pred_returns)
        dir_acc = np.mean(y_true_dir == y_pred_dir)
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'directional_accuracy': float(dir_acc)
        }
