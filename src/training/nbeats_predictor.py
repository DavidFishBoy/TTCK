
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
    
    def __init__(
        self,
        horizon: int = 5,
        input_size: int = 90,
        learning_rate: float = 1e-3,
        max_steps: int = 2000,
        num_stacks: int = 3,
        random_seed: int = 42
    ):
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
        all_dfs = []
        
        for coin_name, df in data_dict.items():
            if df is None or df.empty:
                logger.warning(f"Skipping {coin_name}: empty DataFrame")
                continue
            
            coin_df = df.copy()
            
            if 'timestamp' in coin_df.columns:
                coin_df['ds'] = pd.to_datetime(coin_df['timestamp'])
            elif coin_df.index.name == 'timestamp' or isinstance(coin_df.index, pd.DatetimeIndex):
                coin_df['ds'] = pd.to_datetime(coin_df.index)
            else:
                for col in ['date', 'time', 'datetime']:
                    if col in coin_df.columns:
                        coin_df['ds'] = pd.to_datetime(coin_df[col])
                        break
                else:
                    logger.error(f"No timestamp column found for {coin_name}")
                    continue
            
            if target_col not in coin_df.columns:
                logger.error(f"Column '{target_col}' not found in {coin_name}")
                continue
            
            coin_df['log_price'] = np.log(coin_df[target_col].astype(float))
            coin_df['y'] = coin_df['log_price'].diff()
            
            coin_df = coin_df.dropna(subset=['y'])
            
            coin_symbol = coin_name.upper()[:3] if len(coin_name) > 3 else coin_name.upper()
            
            long_df = pd.DataFrame({
                'unique_id': coin_symbol,
                'ds': coin_df['ds'],
                'y': coin_df['y']
            })
            
            all_dfs.append(long_df)
            logger.info(f"Prepared {len(long_df)} samples for {coin_name} ({coin_symbol})")
        
        if not all_dfs:
            raise ValueError("No valid data to prepare")
        
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
        if self._model is None:
            self._init_model()
        
        required_cols = ['unique_id', 'ds', 'y']
        for col in required_cols:
            if col not in df_long.columns:
                raise ValueError(f"Missing required column: {col}")
        
        logger.info(f"Starting N-BEATS training on {len(df_long)} samples...")
        start_time = datetime.now()
        
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
        price_forecasts = {}
        
        for coin_id in predictions['unique_id'].unique():
            coin_preds = predictions[predictions['unique_id'] == coin_id][return_col].values
            
            if coin_id not in last_prices:
                logger.warning(f"No last price for {coin_id}, skipping")
                continue
            
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
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        
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
    
    @staticmethod
    def evaluate_log_return(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        y_true_returns = np.diff(np.log(y_true + 1e-8))
        y_pred_returns = np.diff(np.log(y_pred + 1e-8))
        
        mae = float(np.mean(np.abs(y_true_returns - y_pred_returns)))
        rmse = float(np.sqrt(np.mean((y_true_returns - y_pred_returns) ** 2)))
        
        y_true_dir = np.sign(y_true_returns)
        y_pred_dir = np.sign(y_pred_returns)
        dir_acc = float(np.mean(y_true_dir == y_pred_dir))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': dir_acc
        }
    
    def save(self, path: Union[str, Path]) -> None:
        if not self._is_fitted:
            raise ValueError("Cannot save untrained model")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        nf = self._model.neural_forecast
        nf.save(path=str(path), model_index=None, overwrite=True, save_dataset=True)
        
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
        try:
            from neuralforecast import NeuralForecast
        except ImportError:
            raise ImportError("neuralforecast is required. Install with: pip install neuralforecast")
        
        path = Path(path)
        
        with open(path / 'params.json', 'r') as f:
            params = json.load(f)
        
        predictor = cls(**params)
        
        predictor._model = NBEATSModel(**params)
        predictor._model._nf = NeuralForecast.load(path=str(path))
        predictor._model._is_initialized = True
        predictor._is_fitted = True
        
        logger.info(f"N-BEATS model loaded from {path}")
        return predictor
