
import logging
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np

class FeatureEngineer:

    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        default_config = {
            'rsi_period': 7,
            'macd_fast_period': 6,
            'macd_slow_period': 13,
            'macd_signal_period': 5,
            'bollinger_window': 10,
            'bollinger_num_std': 2.0,
            'sma_periods': [10, 20],
            'roc_periods': [3, 5],
            'volume_analysis': {
                'enabled': True,
                'window': 10
            }
        }
        self.config = {**default_config, **(config or {})}
        self._all_features = []

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        period = self.config['rsi_period']
        delta = prices.diff()
        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)
        avg_gains = gains.rolling(window=period, min_periods=1).mean()
        avg_losses = losses.rolling(window=period, min_periods=1).mean()
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        fast = self.config['macd_fast_period']
        slow = self.config['macd_slow_period']
        signal = self.config['macd_signal_period']
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - signal_line
        return macd_line, signal_line, macd_hist

    def calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        window = self.config['bollinger_window']
        num_std = self.config['bollinger_num_std']
        middle_band = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = middle_band + (rolling_std * num_std)
        lower_band = middle_band - (rolling_std * num_std)
        return upper_band, middle_band, lower_band

    def calculate_smas(self, prices: pd.Series) -> Dict[str, pd.Series]:
        smas = {}
        for period in self.config['sma_periods']:
            smas[f'sma_{period}'] = prices.rolling(window=period).mean()
        return smas

    def calculate_roc(self, prices: pd.Series) -> Dict[str, pd.Series]:
        rocs = {}
        for period in self.config['roc_periods']:
            rocs[f'roc_{period}'] = prices.pct_change(periods=period) * 100
        return rocs

    def calculate_volume_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        if not self.config['volume_analysis']['enabled']:
            return {}
        window = self.config['volume_analysis']['window']
        volume_features = {
            'volume_ma': df['volume'].rolling(window=window).mean(),
            'volume_std': df['volume'].rolling(window=window).std(),
            'volume_roc': df['volume'].pct_change() * 100
        }
        return volume_features

    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Empty DataFrame provided.")

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df_features = df.copy()

        df_features['rsi'] = self.calculate_rsi(df_features['close'])

        macd_line, macd_signal, macd_hist = self.calculate_macd(df_features['close'])
        df_features['macd'] = macd_line
        df_features['macd_signal'] = macd_signal
        df_features['macd_hist'] = macd_hist

        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df_features['close'])
        df_features['bb_upper'] = bb_upper
        df_features['bb_middle'] = bb_middle
        df_features['bb_lower'] = bb_lower

        smas = self.calculate_smas(df_features['close'])
        df_features.update(smas)

        rocs = self.calculate_roc(df_features['close'])
        df_features.update(rocs)

        volume_feats = self.calculate_volume_features(df_features)
        df_features.update(volume_feats)

        if df_features.isnull().values.any():
            initial_rows = len(df_features)
            self.logger.warning("Missing values detected, applying forward fill only (no bfill).")
            df_features.ffill(inplace=True)
            df_features.dropna(inplace=True)
            dropped = initial_rows - len(df_features)
            if dropped > 0:
                self.logger.info(f"Dropped {dropped} rows with NaN values after ffill.")

        self.all_features = df_features.columns.tolist()
        self.logger.info(f"Added {len(df_features.columns) - len(required_cols)} technical features.")
        return df_features

    def create_target_variable(self, df: pd.DataFrame, forecast_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if 'close' not in df.columns:
            raise ValueError("The 'close' column is required.")

        target = df['close'].shift(-forecast_horizon)
        df = df.iloc[:-forecast_horizon]
        target = target.iloc[:-forecast_horizon]

        self.logger.info(f"Created target variable with horizon={forecast_horizon}. "
                         f"Features shape: {df.shape}, Target shape: {target.shape}")
        return df, target

    def get_feature_groups(self) -> Dict[str, List[str]]:
        feature_groups = {
            'price': ['open', 'high', 'low', 'close'],
            'volume': ['volume'],
            'momentum': [],
            'trend': [],
            'volatility': [],
            'oscillator': []
        }

        for col in self.all_features:
            if col.startswith('sma_'):
                feature_groups['trend'].append(col)
            elif col.startswith('bb_'):
                feature_groups['volatility'].append(col)
            elif col.startswith('roc_') or 'macd' in col or col == 'rsi':
                feature_groups['momentum'].append(col)
            elif col.startswith('volume_') and col not in feature_groups['volume']:
                feature_groups['volume'].append(col)

        self.logger.debug(f"Feature groups: {feature_groups}")
        return feature_groups

    @property
    def all_features(self) -> List[str]:
        if not hasattr(self, '_all_features'):
            self.logger.warning("Accessing 'all_features' before it is set.")
            return []
        return self._all_features

    @all_features.setter
    def all_features(self, features: List[str]):
        if not isinstance(features, list):
            raise ValueError("'features' must be a list of strings.")
        if not all(isinstance(f, str) for f in features):
            raise ValueError("All elements in 'features' must be strings.")
        self._all_features = features
        self.logger.info(f"All features updated, total columns: {len(features)}")