import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_collection.data_collector import DataCollector
from src.preprocessing.pipeline import Pipeline
from src.training.lstm_model import CryptoPredictor
from src.training.trainer import ModelTrainer
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.visualization.visualizer import CryptoVisualizer
from src.utils.custom_losses import (
    direction_aware_huber_loss,
    directional_accuracy_multistep,
    mae_return_metric,
    rmse_return_metric
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


async def collect_data(config: Config, logger: logging.Logger, coins: Optional[List[str]] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    logger.info("Starting data collection.")
    data_config = config.get_data_config()
    selected_coins = coins or data_config.get('coins', [])
    logger.info(f"Selected coins: {selected_coins}")

    raw_data_dir = Path(config.get_path('raw_data_dir'))
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime('%Y%m%d')
    existing_data = {}
    coins_to_fetch = []

    for coin in selected_coins:
        binance_file = raw_data_dir / f"{coin}_binance_{today_str}.csv"
        if binance_file.exists():
            df = pd.read_csv(binance_file, index_col=0)
            existing_data[coin] = {'binance': df}
            logger.info(f"Data for {coin} already exists. Skipping collection.")
        else:
            coins_to_fetch.append(coin)

    if not coins_to_fetch:
        logger.info("All selected coins already have today's data. Exiting data collection successfully.")
        return existing_data

    collector = DataCollector(
        coins=coins_to_fetch,
        days=data_config['days'],
        symbol_mapping=data_config.get('symbol_mapping', []),  
        coin_map=data_config.get('coin_map', {}),
        outlier_detection=True,
        outlier_threshold=3.0,
        cryptocompare_api_key=data_config.get('cryptocompare_api_key'),
        cryptocompare_symbol_map=data_config.get('cryptocompare_symbol_map', {})
    )

    data = await collector.collect_all_data(coins_to_fetch)

    for coin, sources in data.items():
        for source, df in sources.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                filename = f"{coin}_{source}_{today_str}.csv"
                df.to_csv(raw_data_dir / filename)
                logger.info(f"Saved {filename} for {coin}.")
            else:
                logger.warning(f"No data for {coin} from {source}.")

    all_data = {**existing_data, **data}
    logger.info("Data collection completed.")
    return all_data


def preprocess_and_train(config: Config, logger: logging.Logger, data: Dict[str, Dict[str, pd.DataFrame]]):
    logger.info("Starting preprocessing and training.")
    processed_data = {}
    pipelines = {}

    processed_dir = Path(config.get_path('processed_data_dir'))
    processed_dir.mkdir(parents=True, exist_ok=True)

    for coin, sources in data.items():
        # We now only have binance data
        binance_df = sources.get('binance')

        if binance_df is not None and not binance_df.empty:
            df = binance_df
        else:
            logger.warning(f"No valid data for {coin}, skipping.")
            continue

        pipeline = Pipeline(config=config)
        # Pass save_dir to ensure scalers are saved
        coin_data = pipeline.run(df, save_dir=str(processed_dir / coin))
        processed_data[coin] = coin_data
        pipelines[coin] = pipeline
        logger.info(f"Preprocessed data for {coin}")

    results = {}
    model_config = config.get_model_config()
    training_config = config.get_training_config()

    for coin, coin_data in processed_data.items():
        logger.info(f"Training model for {coin}...")

        input_shape = (model_config["sequence_length"], coin_data["X_train"].shape[2])
        model = CryptoPredictor(
            input_shape=input_shape,
            lstm_units=model_config['lstm_units'],
            dropout_rate=model_config['dropout_rate'],
            dense_units=model_config['dense_units'],
            learning_rate=model_config['learning_rate'],
            clip_norm=model_config.get('clip_norm', None)
        )

        model.build()
        model.compile(loss=direction_aware_huber_loss)

        trainer = ModelTrainer(
            model=model.model,
            model_dir=config.get_path('models_dir'),
            batch_size=training_config['batch_size'],
            epochs=training_config['epochs'],
            early_stopping_patience=training_config['early_stopping']['patience'],
            min_delta=training_config['early_stopping']['min_delta']
        )

        history = trainer.train(
            X_train=coin_data["X_train"],
            y_train=coin_data["y_train"],
            X_val=coin_data["X_val"],
            y_val=coin_data["y_val"],
            additional_callbacks=None  
        )

        eval_results = model.evaluate(coin_data["X_test"], coin_data["y_test"])
        preds_returns = model.predict(coin_data["X_test"])  

        test_last_prices = coin_data["test_last_prices"]
        preds_prices = pipelines[coin].inverse_transform_predictions(preds_returns, test_last_prices)
        y_test_prices = pipelines[coin].inverse_transform_actuals(coin_data["y_test"], test_last_prices)

        coin_model_dir = Path(config.get_path('models_dir')) / coin
        coin_model_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = coin_model_dir / "model.keras"
        model.save(model_save_path)

        results[coin] = {
            "history": history.history,
            "evaluation": eval_results,
            "predictions": preds_prices.tolist(),  
            "actual_prices": y_test_prices.tolist()  
        }

    return results, processed_data, pipelines


def train_nbeats(
    config: Config,
    logger: logging.Logger,
    data: Dict[str, Dict[str, pd.DataFrame]]
) -> Dict:

    nbeats_config = config.get_nbeats_config()
    
    if not nbeats_config.get('enabled', False):
        logger.info("N-BEATS is disabled in config, skipping...")
        return {}
    
    logger.info("=" * 50)
    logger.info("Starting N-BEATS training...")
    logger.info("=" * 50)
    
    from src.training.nbeats_predictor import NBEATSPredictor
    
    nbeats = NBEATSPredictor(
        horizon=nbeats_config.get('horizon', 5),
        input_size=nbeats_config.get('input_size', 90),
        learning_rate=nbeats_config.get('learning_rate', 0.001),
        max_steps=nbeats_config.get('max_steps', 2000),
        num_stacks=nbeats_config.get('num_stacks', 3)
    )
    
    data_for_nbeats = {}
    for coin, sources in data.items():
        binance_df = sources.get('binance')
        if binance_df is not None and not binance_df.empty:
            data_for_nbeats[coin] = binance_df
    
    if not data_for_nbeats:
        logger.error("No valid data for N-BEATS training")
        return {}
    
    try:
        df_long = nbeats.prepare_long_format(data_for_nbeats)
        
        training_info = nbeats.train(df_long)
        
        predictions = nbeats.predict()
        
        last_prices = {}
        for coin, df in data_for_nbeats.items():
            coin_symbol = coin.upper()[:3] if len(coin) > 3 else coin.upper()
            if 'close' in df.columns:
                last_prices[coin_symbol] = float(df['close'].iloc[-1])
        

        price_forecasts = nbeats.predict_returns_to_prices(predictions, last_prices)
        
        nbeats_model_dir = Path(config.get_path('models_dir')) / "nbeats"
        nbeats.save(nbeats_model_dir)
        
        results = {
            "training_info": training_info,
            "predictions": predictions.to_dict('records'),
            "price_forecasts": {k: v for k, v in price_forecasts.items()},
            "model_path": str(nbeats_model_dir)
        }
        
        logger.info("N-BEATS training completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"N-BEATS training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def save_results(
    results: Dict[str, Dict],
    config: Config,
    logger: logging.Logger,
    model_type: str = "lstm"
):

    logger.info(f"Saving {model_type} results...")
    results_dir = Path(config.get_path('results_dir')) / model_type
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if model_type == "nbeats":
        result_path = results_dir / f"nbeats_global_results_{timestamp}.json"
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        logger.info(f"Saved N-BEATS results at {result_path}")
    else:
        for coin, result in results.items():
            result_path = results_dir / f"{coin}_results_{timestamp}.json"
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=4)
            logger.info(f"Saved results for {coin} at {result_path}")

async def run_prediction(config: Config, logger: logging.Logger, coins: Optional[List[str]] = None):
    logger.info("Starting prediction mode...")
    data_config = config.get_data_config()
    selected_coins = coins or data_config.get('coins', [])
    logger.info(f"Selected coins for prediction: {selected_coins}")

    raw_predict_dir = Path(config.get_path('raw_predict_dir'))
    raw_predict_dir.mkdir(parents=True, exist_ok=True)

    days_back = 100
    collector = DataCollector(
        coins=selected_coins,
        days=days_back,
        symbol_mapping=data_config.get('symbol_mapping', []),
        coin_map=data_config.get('coin_map', {}),
        outlier_detection=True,
        outlier_threshold=3.0,
        cryptocompare_api_key=data_config.get('cryptocompare_api_key'),
        cryptocompare_symbol_map=data_config.get('cryptocompare_symbol_map', {})
    )

    today_str = datetime.now().strftime('%Y%m%d')
    all_predictions = {}

    prediction_output_dir = Path(config.get_path('results_dir')) / "predictions"
    prediction_output_dir.mkdir(parents=True, exist_ok=True)

    for coin in selected_coins:
        recent_file = raw_predict_dir / f"{coin}_binance_recent_{today_str}.csv"

        if not recent_file.exists():
            logger.info(f"Recent data for {coin} not found. Collecting now...")

        data = await collector.collect_all_data([coin])

        coin_data = data.get(coin, {})
        binance_df = coin_data.get("binance")

        if binance_df is not None and not binance_df.empty:
            binance_df.to_csv(recent_file)
            logger.info(f"Saved recent binance data for {coin} at {recent_file}")
        else:
            logger.error(f"Failed to collect recent data for {coin}, skipping prediction.")
            continue


        if not recent_file.exists():
            logger.error(f"Failed to collect recent data for {coin}, skipping prediction.")
            continue

        df = pd.read_csv(recent_file, index_col=0)
        if df.empty:
            logger.warning(f"Recent data for {coin} is empty, skipping.")
            continue

        pipeline = Pipeline(config=config)

        processed_dir = Path(config.get_path('processed_data_dir')) / coin
        scaler_dir = processed_dir / "scalers"
        if not scaler_dir.exists():
            logger.error(f"No scaler directory found for {coin}. Cannot predict.")
            continue

        pipeline.load_scaler(
            scaler_path=scaler_dir / "feature_scaler.joblib",
            target_scaler_path=scaler_dir / "target_scaler.joblib"
        )

        numeric_features_path = processed_dir / "numeric_features.json"
        if numeric_features_path.exists():
            with open(numeric_features_path, 'r') as f:
                numeric_features = json.load(f)
            pipeline.numeric_features = numeric_features
        else:
            logger.error(f"No numeric_features.json found for {coin}, cannot proceed.")
            continue

        prediction_data = pipeline.run(df, prediction_mode=True)
        X = prediction_data['X']
        last_price = prediction_data['last_price']

        model_path = Path(config.get_path('models_dir')) / coin / "model.keras"
        logger.info(f"Loading model for coin='{coin}' from '{model_path}'...")
        if not model_path.exists():
            logger.error(f"No trained model found for {coin} at {model_path}. Skipping prediction.")
            continue

        model = CryptoPredictor.load(model_path)
        future_days = 5

        preds_returns = model.predict(X)
        
        predictions = pipeline.inverse_transform_predictions(preds_returns[0], last_price)

        prediction_output = {
            "coin": coin,
            "prediction_generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "forecast_horizon_days": future_days,
            "predictions": [
                {"day": i+1, "expected_price": predictions[i]} for i in range(future_days)
            ],
            "explanation": (
                f"These predictions represent the model's best estimate of {coin} closing prices "
                f"for the next {future_days} days, based on historical patterns and recent data."
            )
        }

        coin_prediction_path = prediction_output_dir / f"{coin}_future_predictions.json"
        with open(coin_prediction_path, 'w') as f:
            json.dump(prediction_output, f, indent=4)
        logger.info(f"{future_days}-day future predictions for {coin} saved to {coin_prediction_path}")

        all_predictions[coin] = prediction_output

    logger.info("Prediction mode completed.")

async def main():
    parser = argparse.ArgumentParser(description="Cryptocurrency Price Prediction")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "train-lstm", "train-nbeats", "predict", "collect-data", "full-pipeline", "compare-models"],
                        help="Pipeline mode: train (both), train-lstm, train-nbeats, predict, collect-data, full-pipeline, or compare-models")
    parser.add_argument("--coins", type=str, nargs="*", default=None,
                        help="List of coins to process (overrides config file)")
    args = parser.parse_args()

    temp_logger = logging.getLogger("default")
    temp_logger.addHandler(logging.StreamHandler(sys.stdout))
    temp_logger.setLevel(logging.INFO)

    try:
        config = Config(args.config)
    except Exception as e:
        temp_logger.error(f"Error loading config: {e}")
        sys.exit(1)

    logger_config = config.get_logging_config()
    log_level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    console_level = log_level_mapping.get(logger_config.get("console_level", "INFO"), logging.INFO)
    file_level = log_level_mapping.get(logger_config.get("file_level", "DEBUG"), logging.DEBUG)
    logger = setup_logger(
        name=logger_config["name"],
        log_dir=logger_config["log_dir"],
        console_level=console_level,
        file_level=file_level,
        rotation=logger_config["rotation"]
    )

    logger.info(f"Running in mode: {args.mode}")

    if args.mode == "collect-data":
        data = await collect_data(config, logger, coins=args.coins)
        logger.info("Data collection completed.")
    elif args.mode == "train":
        data = await collect_data(config, logger, coins=args.coins)
        

        logger.info("=" * 50)
        logger.info("Phase 1: Training LSTM models...")
        logger.info("=" * 50)
        lstm_results, processed_data, pipelines = preprocess_and_train(config, logger, data)
        save_results(lstm_results, config, logger, model_type="lstm")
        logger.info("LSTM training completed.")
        
        logger.info("=" * 50)
        logger.info("Phase 2: Training N-BEATS model...")
        logger.info("=" * 50)
        nbeats_results = train_nbeats(config, logger, data)
        if nbeats_results:
            save_results(nbeats_results, config, logger, model_type="nbeats")
        
        logger.info("=" * 50)
        logger.info("All training completed!")
        logger.info("=" * 50)
    elif args.mode == "train-lstm":
        data = await collect_data(config, logger, coins=args.coins)
        logger.info("=" * 50)
        logger.info("Training LSTM models only...")
        logger.info("=" * 50)
        lstm_results, processed_data, pipelines = preprocess_and_train(config, logger, data)
        save_results(lstm_results, config, logger, model_type="lstm")
        logger.info("LSTM training completed.")
    elif args.mode == "train-nbeats":
        data = await collect_data(config, logger, coins=args.coins)
        logger.info("=" * 50)
        logger.info("Training N-BEATS model only...")
        logger.info("=" * 50)
        nbeats_results = train_nbeats(config, logger, data)
        if nbeats_results:
            save_results(nbeats_results, config, logger, model_type="nbeats")
        logger.info("N-BEATS training completed.")
    elif args.mode == "predict":
        await run_prediction(config, logger, coins=args.coins)
    elif args.mode == "full-pipeline":
        data = await collect_data(config, logger, coins=args.coins)
        
        logger.info("=" * 50)
        logger.info("Phase 1: Training LSTM models...")
        logger.info("=" * 50)
        lstm_results, processed_data, pipelines = preprocess_and_train(config, logger, data)
        save_results(lstm_results, config, logger, model_type="lstm")
        
        logger.info("=" * 50)
        logger.info("Phase 2: Training N-BEATS model...")
        logger.info("=" * 50)
        nbeats_results = train_nbeats(config, logger, data)
        if nbeats_results:
            save_results(nbeats_results, config, logger, model_type="nbeats")
        

        logger.info("=" * 50)
        logger.info("Phase 3: Running predictions...")
        logger.info("=" * 50)
        await run_prediction(config, logger, coins=args.coins)
        logger.info("Full pipeline completed.")
    elif args.mode == "compare-models":
        await compare_models_mode(config, logger, coins=args.coins)
    else:
        logger.warning("Unknown mode selected.")


async def compare_models_mode(config: Config, logger: logging.Logger, coins: Optional[List[str]] = None):
    """Compare LSTM, baseline, and ARIMA models."""
    logger.info("Starting model comparison mode...")
    
    data_config = config.get_data_config()
    selected_coins = coins or data_config.get('coins', [])
    
    from src.training.baseline_models import get_all_baseline_models
    from src.training.arima_predictor import ARIMAPredictor
    
    results_dir = Path(config.get_path('results_dir'))
    comparison_dir = results_dir / "model_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    for coin in selected_coins:
        logger.info(f"Comparing models for {coin}...")
        
        
        processed_dir = Path(config.get_path('processed_data_dir')) / coin
        
        if not processed_dir.exists():
            logger.warning(f"No processed data found for {coin}, skipping")
            continue
        
        try:
            X_test = np.load(processed_dir / "X_test.npy")
            y_test = np.load(processed_dir / "y_test.npy")
            
            pipeline = Pipeline(config=config)
            scaler_dir = processed_dir / "scalers"
            pipeline.load_scaler(
                scaler_dir / "feature_scaler.joblib",
                scaler_dir / "target_scaler.joblib"
            )
            
            y_test_prices = pipeline.inverse_transform_actuals(y_test)
            
            comparison_results = {
                'coin': coin,
                'models': {},
                'comparison_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            lstm_model_path = Path(config.get_path('models_dir')) / coin / "model.keras"
            if lstm_model_path.exists():
                logger.info(f"Loading LSTM model for {coin}...")
                lstm_model = CryptoPredictor.load(lstm_model_path)
                lstm_preds_scaled = lstm_model.predict(X_test)
                lstm_preds = pipeline.inverse_transform_predictions(lstm_preds_scaled.flatten())
                
                comparison_results['models']['lstm'] = {
                    'mae': float(np.mean(np.abs(y_test_prices - lstm_preds))),
                    'rmse': float(np.sqrt(np.mean((y_test_prices - lstm_preds)**2))),
                    'directional_accuracy': float(np.mean(
                        np.sign(np.diff(y_test_prices, prepend=y_test_prices[0])) ==
                        np.sign(np.diff(lstm_preds, prepend=lstm_preds[0]))
                    ))
                }
                logger.info(f"LSTM metrics: {comparison_results['models']['lstm']}")
            else:
                logger.warning(f"No LSTM model found for {coin}")
            
           
            logger.info("Running baseline models...")
            close_idx = len(pipeline.numeric_features)  
            
            baseline_models = get_all_baseline_models()
            for baseline in baseline_models:
                preds_scaled = baseline.predict(X_test, close_idx=close_idx)
                preds = pipeline.inverse_transform_predictions(preds_scaled)
                
                metrics = baseline.evaluate(y_test_prices, preds)
                comparison_results['models'][baseline.name.lower().replace('(', '_').replace(')', '').replace('=', '')] = metrics
                logger.info(f"{baseline.name} metrics: {metrics}")
            
            
            logger.info("Running ARIMA model...")
            try:
                arima = ARIMAPredictor()
                arima_preds_scaled = arima.predict_from_sequences(X_test, close_idx=close_idx)
                arima_preds = pipeline.inverse_transform_predictions(arima_preds_scaled)
                
                arima_metrics = arima.evaluate(y_test_prices, arima_preds)
                comparison_results['models']['arima'] = arima_metrics
                logger.info(f"ARIMA metrics: {arima_metrics}")
            except Exception as e:
                logger.error(f"ARIMA failed for {coin}: {e}")
            
            
            best_model = max(
                comparison_results['models'].items(),
                key=lambda x: x[1]['directional_accuracy']
            )
            comparison_results['best_model'] = best_model[0]
            comparison_results['best_directional_accuracy'] = best_model[1]['directional_accuracy']
            
            
            result_file = comparison_dir / f"{coin}_comparison.json"
            with open(result_file, 'w') as f:
                json.dump(comparison_results, f, indent=4)
            
            logger.info(f"Comparison results saved to {result_file}")
            logger.info(f"Best model for {coin}: {best_model[0]} (Dir Acc: {best_model[1]['directional_accuracy']:.4f})")
            
        except Exception as e:
            logger.error(f"Error comparing models for {coin}: {e}")
            continue
    
    logger.info("Model comparison completed for all coins.")


if __name__ == "__main__":
    asyncio.run(main())