# src/training/models/lstm_model.py

"""
Advanced LSTM Model for Cryptocurrency Price Prediction

State-of-the-art deep learning model with:
- Bidirectional LSTM layers for temporal pattern recognition
- Multi-head attention mechanism for feature importance
- CNN layers for local feature extraction  
- Residual connections for better gradient flow
- Layer normalization for training stability
- Monte Carlo dropout for uncertainty estimation
- Direction-integrated loss for trend prediction

Architecture follows best practices from the reference pipeline:
https://github.com/MatNuCuoii/CryptoCurrency
"""

import logging
from typing import List, Tuple, Optional, Callable, Union, Dict
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, 
    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add
)
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from pathlib import Path
import json

from src.utils.custom_losses import di_mse_loss, directional_accuracy


class LSTMModel:
    """
    Advanced LSTM-based cryptocurrency price prediction model.
    
    Implements state-of-the-art architecture with:
    - Bidirectional LSTM for temporal patterns
    - Multi-head attention for feature importance
    - CNN layers for local feature extraction
    - Residual connections for deeper networks
    - Layer normalization for stability
    - Monte Carlo dropout for uncertainty
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        lstm_units: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
        dense_units: Optional[List[int]] = None,
        learning_rate: float = 0.001,
        clip_norm: Optional[float] = 1.0,
        l2_reg: float = 1e-5,
        use_attention: bool = True,
        attention_heads: int = 4,
        use_cnn: bool = True,
        cnn_filters: Optional[List[int]] = None,
        use_residual: bool = True,
        mc_dropout: bool = False,
    ):
        """
        Initialize advanced LSTM model.
        
        Args:
            input_shape: (sequence_length, n_features)
            lstm_units: LSTM units per layer (default: [128, 64])
            dropout_rate: Dropout rate (default: 0.3)
            dense_units: Dense units per layer (default: [64, 32])
            learning_rate: Initial learning rate (default: 0.001)
            clip_norm: Gradient clipping norm (default: 1.0)
            l2_reg: L2 regularization factor (default: 1e-5)
            use_attention: Enable multi-head attention (default: True)
            attention_heads: Number of attention heads (default: 4)
            use_cnn: Enable CNN feature extraction (default: True)
            cnn_filters: CNN filter sizes (default: [64, 32])
            use_residual: Enable residual connections (default: True)
            mc_dropout: Enable Monte Carlo dropout (default: False)
        """
        self.logger = self._setup_logger()
        
        # Core architecture
        self.input_shape = input_shape
        self.lstm_units = lstm_units or [128, 64]
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units or [64, 32]
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.l2_reg = l2_reg
        
        # Advanced features
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.use_cnn = use_cnn
        self.cnn_filters = cnn_filters or [64, 32]
        self.use_residual = use_residual
        self.mc_dropout = mc_dropout
        
        self.model = None
        self.compiled = False
        
        self.logger.info(
            f"LSTMModel initialized: input_shape={input_shape}, "
            f"lstm_units={self.lstm_units}, dense_units={self.dense_units}, "
            f"dropout={dropout_rate}, lr={learning_rate}, "
            f"attention={use_attention}, cnn={use_cnn}, residual={use_residual}"
        )
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup logger for the model"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def build(self) -> None:
        """
        Build advanced LSTM architecture.
        
        Architecture:
        1. CNN branch for local feature extraction (optional)
        2. Bidirectional LSTM layers with layer norm
        3. Multi-head attention mechanism (optional)
        4. Dense layers with residual connections (optional)
        5. Final prediction layer
        """
        self.logger.info("Building advanced LSTM architecture...")
        
        inputs = Input(shape=self.input_shape, name="input")
        x = inputs
        
        # Branch 1: CNN for local feature extraction
        if self.use_cnn:
            cnn_branch = x
            for i, filters in enumerate(self.cnn_filters):
                cnn_branch = Conv1D(
                    filters=filters,
                    kernel_size=3,
                    padding='same',
                    activation='relu',
                    kernel_regularizer=l2(self.l2_reg),
                    name=f"conv1d_{i+1}"
                )(cnn_branch)
                cnn_branch = LayerNormalization(name=f"ln_cnn_{i+1}")(cnn_branch)
                cnn_branch = Dropout(
                    self.dropout_rate, 
                    name=f"dropout_cnn_{i+1}"
                )(cnn_branch, training=self.mc_dropout)
            x = cnn_branch
        
        # Branch 2: Bidirectional LSTM with residual connections
        lstm_output = x
        for i, units in enumerate(self.lstm_units):
            # Always return sequences for attention
            return_sequences = True
            
            # Bidirectional LSTM
            lstm_layer = Bidirectional(
                LSTM(
                    units, 
                    return_sequences=return_sequences,
                    kernel_regularizer=l2(self.l2_reg)
                ),
                name=f"bi_lstm_{i+1}"
            )(lstm_output)
            
            # Layer normalization
            lstm_layer = LayerNormalization(name=f"ln_lstm_{i+1}")(lstm_layer)
            
            # Residual connection (if enabled and dimensions match)
            if self.use_residual and i > 0:
                if lstm_output.shape[-1] == lstm_layer.shape[-1]:
                    lstm_output = Add(name=f"residual_{i+1}")([lstm_output, lstm_layer])
                else:
                    lstm_output = lstm_layer
            else:
                lstm_output = lstm_layer
            
            # Dropout
            lstm_output = Dropout(
                self.dropout_rate,
                name=f"dropout_lstm_{i+1}"
            )(lstm_output, training=self.mc_dropout)
        
        # Multi-head attention
        if self.use_attention:
            # Self-attention mechanism
            attention_output = MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=lstm_output.shape[-1] // self.attention_heads,
                dropout=self.dropout_rate,
                name="multi_head_attention"
            )(lstm_output, lstm_output)
            
            # Residual + layer norm
            attention_output = Add(name="attention_residual")([lstm_output, attention_output])
            attention_output = LayerNormalization(name="ln_attention")(attention_output)
            
            # Global average pooling
            pooled = GlobalAveragePooling1D(name="global_avg_pool")(attention_output)
        else:
            # Use last timestep
            pooled = lstm_output[:, -1, :]
        
        # Dense layers with residual connections
        dense_output = pooled
        for i, units in enumerate(self.dense_units):
            dense_layer = Dense(
                units,
                activation="relu",
                kernel_regularizer=l2(self.l2_reg),
                name=f"dense_{i+1}"
            )(dense_output)
            dense_layer = LayerNormalization(name=f"ln_dense_{i+1}")(dense_layer)
            
            # Residual if dimensions match
            if self.use_residual and dense_output.shape[-1] == units:
                dense_output = Add(name=f"dense_residual_{i+1}")([dense_output, dense_layer])
            else:
                dense_output = dense_layer
            
            dense_output = Dropout(
                self.dropout_rate,
                name=f"dropout_dense_{i+1}"
            )(dense_output, training=self.mc_dropout)
        
        # Output layer (single price prediction)
        output = Dense(1, name="price_prediction")(dense_output)
        
        # Create model
        self.model = Model(inputs, output, name="Advanced_LSTM_Predictor")
        
        self.logger.info("✓ Advanced LSTM built successfully")
        self.logger.info(f"  Total parameters: {self.model.count_params():,}")
        self.logger.info(f"  CNN: {self.use_cnn}, Attention: {self.use_attention}, Residual: {self.use_residual}")
        self.logger.info(f"  MC Dropout: {self.mc_dropout}")
    
    def compile(
        self,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        loss: Optional[Union[str, Callable]] = None,
    ) -> None:
        """
        Compile model with optimizer and loss function.
        
        Args:
            optimizer: Keras optimizer (default: Adam with clipnorm)
            loss: Loss function (default: di_mse_loss for direction-integrated prediction)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        # Setup optimizer
        if optimizer is None:
            if self.clip_norm:
                optimizer = Adam(
                    learning_rate=self.learning_rate,
                    clipnorm=self.clip_norm
                )
            else:
                optimizer = Adam(learning_rate=self.learning_rate)
            
            self.logger.info(f"Using Adam optimizer: lr={self.learning_rate}, clipnorm={self.clip_norm}")
        
        # Setup loss
        if loss is None:
            loss = di_mse_loss
            self.logger.info("Using DI-MSE loss (direction-integrated for trend prediction)")
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["mae", RootMeanSquaredError(name="rmse"), directional_accuracy]
        )
        
        self.compiled = True
        self.logger.info("✓ Model compiled successfully")
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X_train: Training features (samples, sequence_length, features)
            y_train: Training targets (samples, 2) - [current_price, prev_price]
            epochs: Number of epochs
            batch_size: Batch size
            validation_data: Optional (X_val, y_val) tuple
            callbacks: Optional list of callbacks
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if self.model is None or not self.compiled:
            raise ValueError("Model not ready. Build and compile first.")
        
        self.logger.info(f"Training model: epochs={epochs}, batch_size={batch_size}")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.logger.info("✓ Training complete")
        return history
    
    def predict(
        self, 
        X: np.ndarray, 
        verbose: int = 0,
        mc_samples: int = 0
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions with optional Monte Carlo dropout for uncertainty estimation.
        
        Args:
            X: Input features (samples, sequence_length, features)
            verbose: Verbosity level
            mc_samples: Number of MC dropout samples (0 = deterministic, >0 = stochastic)
            
        Returns:
            If mc_samples == 0: predictions array
            If mc_samples > 0: (mean_predictions, std_predictions) for uncertainty
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        if mc_samples > 0 and self.mc_dropout:
            # Monte Carlo dropout for uncertainty estimation
            self.logger.info(f"Generating MC dropout predictions ({mc_samples} samples)...")
            predictions = []
            
            for _ in range(mc_samples):
                # Enable dropout during inference
                pred = self.model(X, training=True)
                predictions.append(pred.numpy())
            
            predictions = np.array(predictions)
            mean_pred = predictions.mean(axis=0)
            std_pred = predictions.std(axis=0)
            
            self.logger.info(f"MC predictions: mean={mean_pred.mean():.2f}, std={std_pred.mean():.4f}")
            return mean_pred, std_pred
        else:
            # Standard deterministic prediction
            self.logger.info(f"Generating predictions for {len(X)} samples...")
            predictions = self.model.predict(X, verbose=verbose)
            return predictions
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: int = 0
    ) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            X: Input features
            y: True labels (samples, 2) - [current_price, prev_price]
            verbose: Verbosity level
            
        Returns:
            Dict of metrics: {loss, mae, rmse, directional_accuracy}
        """
        if self.model is None or not self.compiled:
            raise ValueError("Model not ready. Build and compile first.")
        
        self.logger.info("Evaluating model...")
        results = self.model.evaluate(X, y, verbose=verbose, return_dict=False)
        
        metrics_names = ["loss", "mae", "rmse", "directional_accuracy"]
        metrics = {name: float(val) for name, val in zip(metrics_names, results)}
        
        self.logger.info(
            f"Evaluation results: "
            f"MAE={metrics['mae']:.4f}, "
            f"RMSE={metrics['rmse']:.4f}, "
            f"Dir_Acc={metrics['directional_accuracy']:.4f}"
        )
        
        return metrics
    
    def summary(self) -> None:
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        self.model.summary()
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to file.
        
        Args:
            path: Path to save model (e.g., 'models/lstm/model.keras')
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(path)
        self.logger.info(f"✓ Model saved to {path}")
        
        # Save config for easy loading
        config_path = path.parent / "model_config.json"
        config = {
            "input_shape": self.input_shape,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "dense_units": self.dense_units,
            "learning_rate": self.learning_rate,
            "clip_norm": self.clip_norm,
            "l2_reg": self.l2_reg,
            "use_attention": self.use_attention,
            "attention_heads": self.attention_heads,
            "use_cnn": self.use_cnn,
            "cnn_filters": self.cnn_filters,
            "use_residual": self.use_residual,
            "mc_dropout": self.mc_dropout
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"✓ Model config saved to {config_path}")
    
    @classmethod
    def load(cls, model_path: Union[str, Path]) -> "LSTMModel":
        """
        Load saved model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            LSTMModel instance with loaded weights
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load Keras model with custom objects
        keras_model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "directional_accuracy": directional_accuracy,
                "di_mse_loss": di_mse_loss
            }
        )
        
        # Load config if available
        config_path = model_path.parent / "model_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create instance with saved config
            instance = cls(**config)
        else:
            # Fallback: create with input shape only
            input_shape = keras_model.input_shape[1:]
            instance = cls(input_shape=input_shape)
        
        instance.model = keras_model
        instance.compiled = True
        
        logger = instance.logger
        logger.info(f"✓ Model loaded from {model_path}")
        
        return instance
