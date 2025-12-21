import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import mplfinance as mpf

class NotebookVisualizer:

    def __init__(self, style: str = 'seaborn', fig_size: tuple = (10, 6)):
        plt.style.use(style)
        self.fig_size = fig_size
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def plot_price_history(self, df: pd.DataFrame, title: str = "Price History", use_candlestick: bool = False):
        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required}")
        
        if use_candlestick:
            if not pd.api.types.is_datetime64_ns_dtype(df.index):
                raise ValueError("For candlestick charts, DataFrame index must be datetime.")
            
            mpf.plot(df, type='candle', volume=True, title=title, figratio=(14,7))
        else:
            plt.figure(figsize=self.fig_size)
            plt.plot(df.index, df['close'], label='Close Price')
            plt.title(title)
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.show()

    def plot_volume(self, df: pd.DataFrame, title: str = "Trading Volume"):
        if 'volume' not in df.columns:
            raise ValueError("DataFrame must have a 'volume' column.")
        plt.figure(figsize=self.fig_size)
        plt.bar(df.index, df['volume'], width=0.8)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.show()

    def plot_correlation_matrix(self, df: pd.DataFrame, title: str = "Correlation Matrix"):
        plt.figure(figsize=self.fig_size)
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdBu_r', center=0)
        plt.title(title)
        plt.show()

    def plot_feature_distribution(self, df: pd.DataFrame, columns: Optional[List[str]] = None, title_prefix: str = "Distribution of"):
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            plt.figure(figsize=self.fig_size)
            sns.histplot(df[col], kde=True)
            plt.title(f"{title_prefix} {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()

    def plot_technical_indicators(self, df: pd.DataFrame, indicators: List[str], title_prefix: str = "Technical Indicator"):
        for ind in indicators:
            if ind not in df.columns:
                self.logger.warning(f"{ind} not found in DataFrame columns.")
                continue
            plt.figure(figsize=self.fig_size)
            plt.plot(df.index, df[ind], label=ind)
            plt.title(f"{title_prefix}: {ind}")
            plt.xlabel('Date')
            plt.ylabel(ind)
            plt.legend()
            plt.show()

    def plot_training_history(self, history: Dict[str, List[float]], title: str = "Training History"):
        plt.figure(figsize=(10,8))
        
        if 'loss' in history:
            plt.plot(history['loss'], label='Train Loss', color='blue')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Val Loss', color='red', linestyle='--')

        metric_keys = [k for k in history.keys() if k not in ('loss', 'val_loss')]
        for mk in metric_keys:
            plt.plot(history[mk], label=mk.capitalize())

        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss / Metrics')
        plt.legend()
        plt.show()

    def plot_actual_vs_predicted(self, actual: pd.Series, predicted: pd.Series, title: str = "Actual vs Predicted"):
        if len(actual) != len(predicted):
            raise ValueError("Actual and predicted must have same length.")
        
        plt.figure(figsize=self.fig_size)
        plt.plot(actual.index, actual.values, label='Actual', color='blue')
        plt.plot(predicted.index, predicted.values, label='Predicted', color='red', linestyle='--')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def plot_future_predictions(
            self, 
            df: pd.DataFrame, 
            future_predictions: list, 
            future_days: int = 5, 
            title: str = "Future Predictions",
            test_actual: pd.Series = None,
            test_predicted: pd.Series = None
        ):
        if 'close' not in df.columns:
            raise ValueError("df must have a 'close' column.")
        if future_days != len(future_predictions):
            raise ValueError("future_days must match length of future_predictions.")

        hist_df = df.tail(50)

        last_date = df.index[-1]

        future_dates = pd.date_range(start=last_date, periods=future_days+1, freq='D')[1:]

        plt.figure(figsize=self.fig_size)

        plt.plot(hist_df.index, hist_df['close'], label='Historical (Last 50 Days)', color='blue')

        if test_actual is not None and test_predicted is not None:
            plt.plot(test_actual.index, test_actual.values, label='Test Actual', color='red')
            plt.plot(test_predicted.index, test_predicted.values, label='Test Predicted', color='orange', linestyle='--')
        elif test_actual is not None:
            plt.plot(test_actual.index, test_actual.values, label='Test Actual', color='red')
        elif test_predicted is not None:
            plt.plot(test_predicted.index, test_predicted.values, label='Test Predicted', color='orange', linestyle='--')

        plt.plot(future_dates, future_predictions, label='Future Predicted', color='green', linestyle='--', marker='o')

        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def save_figure(self, filename: str):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, bbox_inches='tight')
        self.logger.info(f"Figure saved to {filename}")