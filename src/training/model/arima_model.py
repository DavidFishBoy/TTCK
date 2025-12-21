

import logging
from typing import Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class ARIMAModel:
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (5, 1, 0),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)
    ):
        self.logger = self._setup_logger()
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model = None
        
        self.logger.info(f"ARIMAModel initialized: order={order}, seasonal={seasonal_order}")
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def fit(self, prices: pd.Series) -> None:
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            raise ImportError("statsmodels required. Install: pip install statsmodels")
        
        if len(prices) < 30:
            raise ValueError(f"Insufficient data: {len(prices)} points (minimum 30 required)")
        
        self.logger.info(f"Fitting ARIMA{self.order} on {len(prices)} data points...")
        
        try:
            model = ARIMA(prices, order=self.order)
            self.fitted_model = model.fit()
            
            self.logger.info(f"ARIMA fitted successfully")
            self.logger.info(f"AIC: {self.fitted_model.aic:.2f}, BIC: {self.fitted_model.bic:.2f}")
            
        except Exception as e:
            self.logger.error(f"ARIMA fitting failed: {e}")
            raise
    
    def predict(
        self,
        horizon: int = 7,
        return_conf_int: bool = False,
        alpha: float = 0.05
    ) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.logger.info(f"Forecasting {horizon} steps ahead...")
        
        try:
            forecast_result = self.fitted_model.forecast(steps=horizon)
            predictions = forecast_result.values if hasattr(forecast_result, 'values') else forecast_result
            
            if return_conf_int:
                forecast_obj = self.fitted_model.get_forecast(steps=horizon)
                conf_int = forecast_obj.conf_int(alpha=alpha)
                lower_ci = conf_int.iloc[:, 0].values
                upper_ci = conf_int.iloc[:, 1].values
                
                self.logger.info(f"Forecast with CI generated: mean={predictions.mean():.2f}")
                return predictions, (lower_ci, upper_ci)
            else:
                self.logger.info(f"Forecast generated: mean={predictions.mean():.2f}")
                return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    def evaluate(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray
    ) -> Dict:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        r2 = r2_score(actuals, predictions)
        
        actual_direction = np.sign(np.diff(actuals, prepend=actuals[0]))
        pred_direction = np.sign(np.diff(predictions, prepend=predictions[0]))
        dir_acc = np.mean(actual_direction == pred_direction)
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2_score': float(r2),
            'directional_accuracy': float(dir_acc)
        }
        
        self.logger.info(f"Evaluation: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
        return metrics
    
    @staticmethod
    def auto_select_order(
        prices: pd.Series,
        max_p: int = 5,
        max_q: int = 5,
        d: int = 1
    ) -> Tuple[int, int, int]:
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            return (5, 1, 0)
        
        best_aic = np.inf
        best_order = (1, d, 0)
        
        logger = logging.getLogger(__name__)
        logger.info("Auto-selecting ARIMA order...")
        
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                
                try:
                    model = ARIMA(prices, order=(p, d, q))
                    fitted = model.fit()
                    
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                
                except:
                    continue
        
        logger.info(f"Best order: {best_order} (AIC={best_aic:.2f})")
        return best_order
