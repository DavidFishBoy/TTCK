# src/training/baseline_models.py

"""
Baseline models for benchmarking.
These models don't require training - they use simple rules.
"""

import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class NaiveModel:
    """
    Naive persistence model.
    Predicts tomorrow's price = today's price.
    """
    
    def __init__(self):
        self.name = "Naive"
        logger.info("Initialized Naive baseline model")
    
    def predict(self, X: np.ndarray, close_idx: int = -1) -> np.ndarray:
        """
        Predict using last known price.
        
        Args:
            X: Input sequences of shape (samples, timesteps, features)
            close_idx: Index of close price in features dimension
        
        Returns:
            Predictions array of shape (samples,)
        """
        # Use the last timestep's close price as prediction
        predictions = X[:, -1, close_idx]
        logger.info(f"Naive model generated {len(predictions)} predictions")
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


class MovingAverageModel:
    """
    Moving Average model.
    Predicts using simple moving average of recent prices.
    """
    
    def __init__(self, window: int = 5):
        """
        Args:
            window: Number of recent prices to average
        """
        self.window = window
        self.name = f"MA({window})"
        logger.info(f"Initialized Moving Average baseline model (window={window})")
    
    def predict(self, X: np.ndarray, close_idx: int = -1) -> np.ndarray:
        """
        Predict using moving average of last `window` prices.
        
        Args:
            X: Input sequences of shape (samples, timesteps, features)
            close_idx: Index of close price in features dimension
        
        Returns:
            Predictions array of shape (samples,)
        """
        # Get the last `window` close prices for each sample
        recent_prices = X[:, -self.window:, close_idx]
        
        # Calculate mean
        predictions = np.mean(recent_prices, axis=1)
        
        logger.info(f"MA({self.window}) generated {len(predictions)} predictions")
        return predictions
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        return NaiveModel.evaluate(y_true, y_pred)


class ExponentialMovingAverageModel:
    """
    Exponential Moving Average model.
    Gives more weight to recent prices.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: Smoothing factor (0 < alpha <= 1)
                  Higher alpha = more weight on recent values
        """
        self.alpha = alpha
        self.name = f"EMA(α={alpha})"
        logger.info(f"Initialized EMA baseline model (alpha={alpha})")
    
    def predict(self, X: np.ndarray, close_idx: int = -1) -> np.ndarray:
        """
        Predict using exponential moving average.
        
        Args:
            X: Input sequences of shape (samples, timesteps, features)
            close_idx: Index of close price in features dimension
        
        Returns:
            Predictions array of shape (samples,)
        """
        predictions = []
        
        for sample in X:
            prices = sample[:, close_idx]
            
            # Calculate EMA
            ema = prices[0]
            for price in prices[1:]:
                ema = self.alpha * price + (1 - self.alpha) * ema
            
            predictions.append(ema)
        
        predictions = np.array(predictions)
        logger.info(f"EMA generated {len(predictions)} predictions")
        return predictions
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        return NaiveModel.evaluate(y_true, y_pred)


def get_all_baseline_models():
    """Get all baseline models for comparison."""
    return [
        NaiveModel(),
        MovingAverageModel(window=5),
        MovingAverageModel(window=10),
        MovingAverageModel(window=20),
        ExponentialMovingAverageModel(alpha=0.3),
    ]
