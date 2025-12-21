

import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class NaiveModel:
    
    def __init__(self):
        self.name = "Naive"
        logger.info("Initialized Naive baseline model")
    
    def predict(self, X: np.ndarray, close_idx: int = -1) -> np.ndarray:
        predictions = X[:, -1, close_idx]
        logger.info(f"Naive model generated {len(predictions)} predictions")
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

class MovingAverageModel:
    
    def __init__(self, window: int = 5):
        self.window = window
        self.name = f"MA({window})"
        logger.info(f"Initialized Moving Average baseline model (window={window})")
    
    def predict(self, X: np.ndarray, close_idx: int = -1) -> np.ndarray:
        recent_prices = X[:, -self.window:, close_idx]
        
        predictions = np.mean(recent_prices, axis=1)
        
        logger.info(f"MA({self.window}) generated {len(predictions)} predictions")
        return predictions
    
    def predict_log_return_future(self, prices: np.ndarray, horizon: int = 5) -> np.ndarray:
        log_returns = np.diff(np.log(prices + 1e-8))
        
        predicted_return = np.mean(log_returns[-self.window:])
        
        return np.full(horizon, predicted_return)
    
    def predict_future_prices(self, prices: np.ndarray, horizon: int = 5) -> np.ndarray:
        predicted_returns = self.predict_log_return_future(prices, horizon)
        last_price = prices[-1]
        
        future_prices = []
        current_log_price = np.log(last_price)
        
        for r in predicted_returns:
            current_log_price += r
            future_prices.append(np.exp(current_log_price))
        
        return np.array(future_prices)
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return NaiveModel.evaluate(y_true, y_pred)
    
    @staticmethod
    def evaluate_log_return(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return NaiveModel.evaluate_log_return(y_true, y_pred)

class ExponentialMovingAverageModel:
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.name = f"EMA(α={alpha})"
        logger.info(f"Initialized EMA baseline model (alpha={alpha})")
    
    def predict(self, X: np.ndarray, close_idx: int = -1) -> np.ndarray:
        predictions = []
        
        for sample in X:
            prices = sample[:, close_idx]
            
            ema = prices[0]
            for price in prices[1:]:
                ema = self.alpha * price + (1 - self.alpha) * ema
            
            predictions.append(ema)
        
        predictions = np.array(predictions)
        logger.info(f"EMA generated {len(predictions)} predictions")
        return predictions
    
    def predict_log_return_future(self, prices: np.ndarray, horizon: int = 5) -> np.ndarray:
        log_returns = np.diff(np.log(prices + 1e-8))
        
        ema = log_returns[0]
        for r in log_returns[1:]:
            ema = self.alpha * r + (1 - self.alpha) * ema
        
        return np.full(horizon, ema)
    
    def predict_future_prices(self, prices: np.ndarray, horizon: int = 5) -> np.ndarray:
        predicted_returns = self.predict_log_return_future(prices, horizon)
        last_price = prices[-1]
        
        future_prices = []
        current_log_price = np.log(last_price)
        
        for r in predicted_returns:
            current_log_price += r
            future_prices.append(np.exp(current_log_price))
        
        return np.array(future_prices)
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return NaiveModel.evaluate(y_true, y_pred)
    
    @staticmethod
    def evaluate_log_return(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return NaiveModel.evaluate_log_return(y_true, y_pred)

def get_all_baseline_models():
    return [
        NaiveModel(),
        MovingAverageModel(window=5),
        MovingAverageModel(window=10),
        MovingAverageModel(window=20),
        ExponentialMovingAverageModel(alpha=0.3),
    ]
