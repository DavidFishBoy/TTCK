# src/training/model/nbeats_model.py
"""
N-BEATS (Neural Basis Expansion Analysis for Time Series) model definition.
Contains the core model class that wraps NeuralForecast NBEATS.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class NBEATSModel:
    """
    N-BEATS model wrapper using NeuralForecast library.
    
    This class handles the model initialization and configuration.
    For full prediction pipeline, use NBEATSPredictor from nbeats_predictor.py
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
        Initialize N-BEATS model configuration.
        
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
        
        self._model = None
        self._nf = None  # NeuralForecast wrapper
        self._is_initialized = False
        
        logger.info(
            f"NBEATSModel configured: horizon={horizon}, input_size={input_size}, "
            f"learning_rate={learning_rate}, max_steps={max_steps}"
        )
    
    def _check_neuralforecast(self):
        """Check if neuralforecast is installed."""
        try:
            from neuralforecast import NeuralForecast
            from neuralforecast.models import NBEATS
            return True
        except ImportError:
            raise ImportError(
                "neuralforecast is required for N-BEATS model. "
                "Install with: pip install neuralforecast"
            )
    
    def build(self):
        """Build and initialize the N-BEATS model."""
        self._check_neuralforecast()
        
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATS
        
        self._model = NBEATS(
            h=self.horizon,
            input_size=self.input_size,
            learning_rate=self.learning_rate,
            max_steps=self.max_steps,
            random_seed=self.random_seed,
            stack_types=['trend', 'seasonality', 'identity'][:self.num_stacks],
        )
        
        self._nf = NeuralForecast(
            models=[self._model],
            freq='D'  # Daily frequency
        )
        
        self._is_initialized = True
        logger.info("N-BEATS model built successfully")
        return self
    
    @property
    def model(self):
        """Get the underlying NBEATS model."""
        return self._model
    
    @property
    def neural_forecast(self):
        """Get the NeuralForecast wrapper."""
        return self._nf
    
    @property
    def is_initialized(self) -> bool:
        """Check if model is initialized."""
        return self._is_initialized
    
    def get_config(self) -> dict:
        """Get model configuration as dictionary."""
        return {
            'horizon': self.horizon,
            'input_size': self.input_size,
            'learning_rate': self.learning_rate,
            'max_steps': self.max_steps,
            'num_stacks': self.num_stacks,
            'random_seed': self.random_seed
        }
