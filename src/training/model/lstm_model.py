

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
        self.logger = self._setup_logger()
        
        self.input_shape = input_shape
        self.lstm_units = lstm_units or [128, 64]
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units or [64, 32]
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.l2_reg = l2_reg
        
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
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def build(self) -> None:
        self.logger.info("Building advanced LSTM architecture...")
        
        inputs = Input(shape=self.input_shape, name="input")
        x = inputs
        
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
        
        lstm_output = x
        for i, units in enumerate(self.lstm_units):
            return_sequences = True
            
            lstm_layer = Bidirectional(
                LSTM(
                    units, 
                    return_sequences=return_sequences,
                    kernel_regularizer=l2(self.l2_reg)
                ),
                name=f"bi_lstm_{i+1}"
            )(lstm_output)
            
            lstm_layer = LayerNormalization(name=f"ln_lstm_{i+1}")(lstm_layer)
            
            if self.use_residual and i > 0:
                if lstm_output.shape[-1] == lstm_layer.shape[-1]:
                    lstm_output = Add(name=f"residual_{i+1}")([lstm_output, lstm_layer])
                else:
                    lstm_output = lstm_layer
            else:
                lstm_output = lstm_layer
            
            lstm_output = Dropout(
                self.dropout_rate,
                name=f"dropout_lstm_{i+1}"
            )(lstm_output, training=self.mc_dropout)
        
        if self.use_attention:
            attention_output = MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=lstm_output.shape[-1] // self.attention_heads,
                dropout=self.dropout_rate,
                name="multi_head_attention"
            )(lstm_output, lstm_output)
            
            attention_output = Add(name="attention_residual")([lstm_output, attention_output])
            attention_output = LayerNormalization(name="ln_attention")(attention_output)
            
            pooled = GlobalAveragePooling1D(name="global_avg_pool")(attention_output)
        else:
            pooled = lstm_output[:, -1, :]
        
        dense_output = pooled
        for i, units in enumerate(self.dense_units):
            dense_layer = Dense(
                units,
                activation="relu",
                kernel_regularizer=l2(self.l2_reg),
                name=f"dense_{i+1}"
            )(dense_output)
            dense_layer = LayerNormalization(name=f"ln_dense_{i+1}")(dense_layer)
            
            if self.use_residual and dense_output.shape[-1] == units:
                dense_output = Add(name=f"dense_residual_{i+1}")([dense_output, dense_layer])
            else:
                dense_output = dense_layer
            
            dense_output = Dropout(
                self.dropout_rate,
                name=f"dropout_dense_{i+1}"
            )(dense_output, training=self.mc_dropout)
        
        output = Dense(1, name="price_prediction")(dense_output)
        
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
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        if optimizer is None:
            if self.clip_norm:
                optimizer = Adam(
                    learning_rate=self.learning_rate,
                    clipnorm=self.clip_norm
                )
            else:
                optimizer = Adam(learning_rate=self.learning_rate)
            
            self.logger.info(f"Using Adam optimizer: lr={self.learning_rate}, clipnorm={self.clip_norm}")
        
        if loss is None:
            loss = di_mse_loss
            self.logger.info("Using DI-MSE loss (direction-integrated for trend prediction)")
        
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
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        if mc_samples > 0 and self.mc_dropout:
            self.logger.info(f"Generating MC dropout predictions ({mc_samples} samples)...")
            predictions = []
            
            for _ in range(mc_samples):
                pred = self.model(X, training=True)
                predictions.append(pred.numpy())
            
            predictions = np.array(predictions)
            mean_pred = predictions.mean(axis=0)
            std_pred = predictions.std(axis=0)
            
            self.logger.info(f"MC predictions: mean={mean_pred.mean():.2f}, std={std_pred.mean():.4f}")
            return mean_pred, std_pred
        else:
            self.logger.info(f"Generating predictions for {len(X)} samples...")
            predictions = self.model.predict(X, verbose=verbose)
            return predictions
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: int = 0
    ) -> Dict:
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
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        self.model.summary()
    
    def save(self, path: Union[str, Path]) -> None:
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(path)
        self.logger.info(f"✓ Model saved to {path}")
        
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
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        keras_model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "directional_accuracy": directional_accuracy,
                "di_mse_loss": di_mse_loss
            }
        )
        
        config_path = model_path.parent / "model_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            instance = cls(**config)
        else:
            input_shape = keras_model.input_shape[1:]
            instance = cls(input_shape=input_shape)
        
        instance.model = keras_model
        instance.compiled = True
        
        logger = instance.logger
        logger.info(f"✓ Model loaded from {model_path}")
        
        return instance
