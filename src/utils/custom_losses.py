"""
Custom loss functions and metrics for cryptocurrency return prediction.
Optimized for multi-horizon forecasting with directional awareness.
"""

import tensorflow as tf
from typing import Optional


def directional_accuracy_multistep(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Directional accuracy for multi-horizon returns.
    Measures how often predicted return direction matches actual direction.
    
    Args:
        y_true: shape (batch, horizon) - actual log returns
        y_pred: shape (batch, horizon) - predicted log returns
    
    Returns:
        Scalar tensor with accuracy (0.0 to 1.0)
    """
    direction_true = tf.sign(y_true)
    direction_pred = tf.sign(y_pred)
    correct = tf.cast(tf.equal(direction_pred, direction_true), tf.float32)
    return tf.reduce_mean(correct)


def direction_aware_huber_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    delta: float = 1.0,
    direction_weight: float = 1.5
) -> tf.Tensor:
    """
    Direction-aware Huber loss for return prediction.
    
    Combines:
    1. Huber loss (robust to outliers in return magnitudes)
    2. Direction penalty (wrong trend direction gets extra weight)
    
    This addresses the critical issue in the old di_mse_loss where wrong
    directions were penalized by a constant, not by error magnitude.
    
    Args:
        y_true: shape (batch, horizon) - actual log returns
        y_pred: shape (batch, horizon) - predicted log returns
        delta: Huber loss threshold (1.0 means errors < 1.0 use quadratic loss)
        direction_weight: multiplier for wrong-direction predictions (e.g., 1.5)
    
    Returns:
        Scalar loss value
    """
    # Standard Huber loss calculation
    error = y_true - y_pred
    abs_error = tf.abs(error)
    
    # Huber: quadratic for small errors, linear for large errors
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    huber = 0.5 * quadratic ** 2 + delta * linear
    
    # Direction-based weighting
    direction_true = tf.sign(y_true)
    direction_pred = tf.sign(y_pred)
    
    # Mask for wrong direction predictions
    wrong_direction = tf.cast(
        tf.not_equal(direction_pred, direction_true),
        tf.float32
    )
    
    # Apply extra penalty for wrong direction
    # If correct direction: weight = 1.0
    # If wrong direction: weight = direction_weight (e.g., 1.5)
    weights = 1.0 + (direction_weight - 1.0) * wrong_direction
    weighted_huber = huber * weights
    
    return tf.reduce_mean(weighted_huber)


def mse_return_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Simple MSE loss for returns (baseline).
    
    Args:
        y_true: shape (batch, horizon) - actual log returns
        y_pred: shape (batch, horizon) - predicted log returns
    
    Returns:
        Scalar MSE loss
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))


def mae_return_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    MAE metric for returns.
    
    Args:
        y_true: shape (batch, horizon) - actual log returns
        y_pred: shape (batch, horizon) - predicted log returns
    
    Returns:
        Scalar MAE value
    """
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def rmse_return_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    RMSE metric for returns.
    
    Args:
        y_true: shape (batch, horizon) - actual log returns
        y_pred: shape (batch, horizon) - predicted log returns
    
    Returns:
        Scalar RMSE value
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


# Backward compatibility aliases (deprecated - kept for migration)
directional_accuracy = directional_accuracy_multistep
di_mse_loss = direction_aware_huber_loss
