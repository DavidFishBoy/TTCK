
import tensorflow as tf
from typing import Optional

def directional_accuracy_multistep(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
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
    error = y_true - y_pred
    abs_error = tf.abs(error)
    
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    huber = 0.5 * quadratic ** 2 + delta * linear
    
    direction_true = tf.sign(y_true)
    direction_pred = tf.sign(y_pred)
    
    wrong_direction = tf.cast(
        tf.not_equal(direction_pred, direction_true),
        tf.float32
    )
    
    weights = 1.0 + (direction_weight - 1.0) * wrong_direction
    weighted_huber = huber * weights
    
    return tf.reduce_mean(weighted_huber)

def mse_return_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.square(y_true - y_pred))

def mae_return_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def rmse_return_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

directional_accuracy = directional_accuracy_multistep
di_mse_loss = direction_aware_huber_loss
