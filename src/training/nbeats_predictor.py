# src/training/nbeats_predictor.py
"""
N-BEATS Predictor for cryptocurrency forecasting.
Handles data preparation, training, prediction, and evaluation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.training.model.nbeats_model import NBEATSModel

logger = logging.getLogger(__name__)


class NBEATSPredictor:
    """
    N-BEATS predictor for cryptocurrency forecasting.
    
    Key features:
    - Global model: trains on all coins simultaneously
    - Predicts log returns, not prices
    - Multi-horizon: 5-day forecast by default
    """
    
    def __init__(
        self,
        horizon: int = 5,
        input_size: int = 90,
        learning_rate: float = 1e-3,
        max_steps: int = 2000,
        num_stacks: int = 3,
        random_seed: int = 42
    ):
        """
        Initialize N-BEATS predictor.
        
        Args:
            horizon: Number of days to forecast (default: 5)
            input_size: Lookback window size (default: 90)
            learning_rate: Learning rate for training
            max_steps: Maximum training steps
            num_stacks: Number of NBEATS stacks
            random_seed: Random seed for reproducibility
        """
        self.horizon = horizon
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.num_stacks = num_stacks
        self.random_seed = random_seed
        
        self._model: Optional[NBEATSModel] = None
        self._is_fitted = False
        
        logger.info(
            f"NBEATSPredictor initialized: horizon={horizon}, input_size={input_size}, "
            f"learning_rate={learning_rate}, max_steps={max_steps}"
        )
    
    def _init_model(self):
        """Initialize the underlying N-BEATS model."""
        self._model = NBEATSModel(
            horizon=self.horizon,
            input_size=self.input_size,
            learning_rate=self.learning_rate,
            max_steps=self.max_steps,
            num_stacks=self.num_stacks,
            random_seed=self.random_seed
        )
        self._model.build()
    
    def prepare_long_format(
        self,
        data_dict: Dict[str, pd.DataFrame],
        target_col: str = "close"
    ) -> pd.DataFrame:
        """
        Convert per-coin DataFrames to NeuralForecast long format.
        
        Expected input format (per coin):
            - DataFrame with 'timestamp' or datetime index
            - 'close' column for price data
        
        Output format (long format):
            unique_id | ds        | y
            BTC       | 2024-01-01| 0.023  (log return)
            BTC       | 2024-01-02| -0.015
            ETH       | 2024-01-01| 0.031
            ...
        
        Args:
            data_dict: Dictionary mapping coin names to their DataFrames
            target_col: Column name for price data (default: 'close')
        
        Returns:
            DataFrame in long format ready for NeuralForecast
        """
        all_dfs = []
        
        for coin_name, df in data_dict.items():
            if df is None or df.empty:
                logger.warning(f"Skipping {coin_name}: empty DataFrame")
                continue
            
            # Create a copy to avoid modifying original
            coin_df = df.copy()
            
            # Handle timestamp column
            if 'timestamp' in coin_df.columns:
                coin_df['ds'] = pd.to_datetime(coin_df['timestamp'])
            elif coin_df.index.name == 'timestamp' or isinstance(coin_df.index, pd.DatetimeIndex):
                coin_df['ds'] = pd.to_datetime(coin_df.index)
            else:
                # Try to find a date column
                for col in ['date', 'time', 'datetime']:
                    if col in coin_df.columns:
                        coin_df['ds'] = pd.to_datetime(coin_df[col])
                        break
                else:
                    logger.error(f"No timestamp column found for {coin_name}")
                    continue
            
            # Calculate log returns
            if target_col not in coin_df.columns:
                logger.error(f"Column '{target_col}' not found in {coin_name}")
                continue
            
            coin_df['log_price'] = np.log(coin_df[target_col].astype(float))
            coin_df['y'] = coin_df['log_price'].diff()
            
            # Remove NaN from diff
            coin_df = coin_df.dropna(subset=['y'])
            
            # Create unique_id (coin symbol in uppercase)
            coin_symbol = coin_name.upper()[:3] if len(coin_name) > 3 else coin_name.upper()
            
            # Select only required columns
            long_df = pd.DataFrame({
                'unique_id': coin_symbol,
                'ds': coin_df['ds'],
                'y': coin_df['y']
            })
            
            all_dfs.append(long_df)
            logger.info(f"Prepared {len(long_df)} samples for {coin_name} ({coin_symbol})")
        
        if not all_dfs:
            raise ValueError("No valid data to prepare")
        
        # Concatenate all coin data
        result = pd.concat(all_dfs, ignore_index=True)
        result = result.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        
        logger.info(f"Long format data prepared: {len(result)} total samples, "
                    f"{result['unique_id'].nunique()} coins")
        
        return result
    
    def train(
        self,
        df_long: pd.DataFrame,
        val_size: Optional[int] = None
    ) -> Dict:
        """
        Train global N-BEATS model on long-format data.
        
        Args:
            df_long: DataFrame in long format (unique_id, ds, y)
            val_size: Optional validation set size (days to hold out per series)
        
        Returns:
            Dictionary with training info
        """
        if self._model is None:
            self._init_model()
        
        # Validate input format
        required_cols = ['unique_id', 'ds', 'y']
        for col in required_cols:
            if col not in df_long.columns:
                raise ValueError(f"Missing required column: {col}")
        
        logger.info(f"Starting N-BEATS training on {len(df_long)} samples...")
        start_time = datetime.now()
        
        # Train the model
        nf = self._model.neural_forecast
        if val_size:
            nf.fit(df=df_long, val_size=val_size)
        else:
            nf.fit(df=df_long)
        
        self._is_fitted = True
        training_time = (datetime.now() - start_time).total_seconds()
        
        training_info = {
            'training_time_seconds': training_time,
            'n_samples': len(df_long),
            'n_coins': df_long['unique_id'].nunique(),
            'horizon': self.horizon,
            'input_size': self.input_size,
            'max_steps': self.max_steps
        }
        
        logger.info(f"N-BEATS training completed in {training_time:.2f}s")
        return training_info
    
    def predict(self, df_long: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate predictions for the next `horizon` days.
        
        Args:
            df_long: Optional new data for prediction (if None, uses training data)
        
        Returns:
            DataFrame with predictions:
                unique_id | ds   | NBEATS
                BTC       | t+1  | 0.012
                BTC       | t+2  | -0.005
                ...
        """
        if not self._is_fitted:
            raise ValueError("Model must be trained before prediction. Call train() first.")
        
        nf = self._model.neural_forecast
        if df_long is not None:
            predictions = nf.predict(df=df_long)
        else:
            predictions = nf.predict()
        
        logger.info(f"Generated predictions: {len(predictions)} rows")
        return predictions
    
    def predict_returns_to_prices(
        self,
        predictions: pd.DataFrame,
        last_prices: Dict[str, float],
        return_col: str = 'NBEATS'
    ) -> Dict[str, List[float]]:
        """
        Convert predicted log returns to actual price forecasts.
        
        Args:
            predictions: DataFrame from predict() with log return predictions
            last_prices: Dictionary mapping coin symbols to their last known prices
            return_col: Column name containing return predictions
        
        Returns:
            Dictionary mapping coin symbols to list of predicted prices
        """
        price_forecasts = {}
        
        for coin_id in predictions['unique_id'].unique():
            coin_preds = predictions[predictions['unique_id'] == coin_id][return_col].values
            
            if coin_id not in last_prices:
                logger.warning(f"No last price for {coin_id}, skipping")
                continue
            
            # Convert returns to prices
            prices = []
            current_log_price = np.log(last_prices[coin_id])
            
            for r in coin_preds:
                current_log_price += r
                prices.append(np.exp(current_log_price))
            
            price_forecasts[coin_id] = prices
        
        return price_forecasts
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate predictions using standard metrics.
        
        Args:
            y_true: Actual values (log returns or prices)
            y_pred: Predicted values
        
        Returns:
            Dictionary with MAE, RMSE, and Directional Accuracy
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        
        # Directional accuracy
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true, prepend=y_true[0]))
            pred_direction = np.sign(np.diff(y_pred, prepend=y_pred[0]))
            dir_acc = float(np.mean(true_direction == pred_direction))
        else:
            dir_acc = float(np.sign(y_true[0]) == np.sign(y_pred[0]))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': dir_acc
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the trained model.
        
        Args:
            path: Directory path to save the model
        """
        if not self._is_fitted:
            raise ValueError("Cannot save untrained model")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save NeuralForecast model
        nf = self._model.neural_forecast
        nf.save(path=str(path), model_index=None, overwrite=True, save_dataset=True)
        
        # Save hyperparameters
        params = {
            'horizon': self.horizon,
            'input_size': self.input_size,
            'learning_rate': self.learning_rate,
            'max_steps': self.max_steps,
            'num_stacks': self.num_stacks,
            'random_seed': self.random_seed
        }
        with open(path / 'params.json', 'w') as f:
            json.dump(params, f, indent=2)
        
        logger.info(f"N-BEATS model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'NBEATSPredictor':
        """
        Load a trained model.
        
        Args:
            path: Directory path containing the saved model
        
        Returns:
            Loaded NBEATSPredictor instance
        """
        try:
            from neuralforecast import NeuralForecast
        except ImportError:
            raise ImportError("neuralforecast is required. Install with: pip install neuralforecast")
        
        path = Path(path)
        
        # Load hyperparameters
        with open(path / 'params.json', 'r') as f:
            params = json.load(f)
        
        # Create instance with saved parameters
        predictor = cls(**params)
        
        # Initialize model and load NeuralForecast
        predictor._model = NBEATSModel(**params)
        predictor._model._nf = NeuralForecast.load(path=str(path))
        predictor._model._is_initialized = True
        predictor._is_fitted = True
        
        logger.info(f"N-BEATS model loaded from {path}")
        return predictor
