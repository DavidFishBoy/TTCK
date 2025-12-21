# train_nbeats.py
"""
Standalone N-BEATS training script.
Run this separately from main.py to avoid TensorFlow/PyTorch DLL conflicts.

Usage:
    python train_nbeats.py --config configs/config.yaml
"""

import argparse
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================
# WORKAROUND: Pre-load c10.dll to fix WinError 1114 on Windows
# ============================================================
import ctypes
if sys.platform == 'win32':
    try:
        # Find torch installation path
        import site
        for site_path in site.getsitepackages():
            c10_path = Path(site_path) / 'torch' / 'lib' / 'c10.dll'
            if c10_path.exists():
                ctypes.CDLL(str(c10_path))
                break
    except Exception:
        pass  # If preload fails, continue anyway
# ============================================================

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('nbeats_trainer')


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_existing_data(config: dict) -> Dict[str, pd.DataFrame]:
    """Load existing data from raw data directory."""
    raw_data_dir = Path(config['paths']['raw_data_dir'])
    coins = config['data']['coins']
    today_str = datetime.now().strftime('%Y%m%d')
    
    data = {}
    for coin in coins:
        # Look for today's data or most recent data
        binance_file = raw_data_dir / f"{coin}_binance_{today_str}.csv"
        
        if binance_file.exists():
            df = pd.read_csv(binance_file, index_col=0)
            data[coin] = df
            logger.info(f"Loaded data for {coin}: {len(df)} rows")
        else:
            # Try to find any recent binance file
            pattern = f"{coin}_binance_*.csv"
            files = list(raw_data_dir.glob(pattern))
            if files:
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                df = pd.read_csv(latest_file, index_col=0)
                data[coin] = df
                logger.info(f"Loaded recent data for {coin}: {len(df)} rows from {latest_file.name}")
            else:
                logger.warning(f"No data found for {coin}")
    
    return data


def prepare_long_format(data_dict: Dict[str, pd.DataFrame], target_col: str = "close") -> pd.DataFrame:
    """Convert per-coin DataFrames to NeuralForecast long format."""
    all_dfs = []
    
    for coin_name, df in data_dict.items():
        if df is None or df.empty:
            logger.warning(f"Skipping {coin_name}: empty DataFrame")
            continue
        
        coin_df = df.copy()
        
        # Handle timestamp
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
        
        # Calculate log returns
        if target_col not in coin_df.columns:
            logger.error(f"Column '{target_col}' not found in {coin_name}")
            continue
        
        coin_df['log_price'] = np.log(coin_df[target_col].astype(float))
        coin_df['y'] = coin_df['log_price'].diff()
        coin_df = coin_df.dropna(subset=['y'])
        
        # Create unique_id
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
    
    logger.info(f"Long format data: {len(result)} samples, {result['unique_id'].nunique()} coins")
    return result


def train_nbeats(config: dict, df_long: pd.DataFrame) -> dict:
    """Train N-BEATS model."""
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS
    
    nbeats_config = config.get('nbeats', {})
    
    logger.info("Initializing N-BEATS model...")
    
    model = NBEATS(
        h=nbeats_config.get('horizon', 5),
        input_size=nbeats_config.get('input_size', 90),
        learning_rate=nbeats_config.get('learning_rate', 0.001),
        max_steps=nbeats_config.get('max_steps', 2000),
        random_seed=42,
        stack_types=['trend', 'seasonality', 'identity'][:nbeats_config.get('num_stacks', 3)],
    )
    
    nf = NeuralForecast(models=[model], freq='D')
    
    logger.info(f"Starting training on {len(df_long)} samples...")
    start_time = datetime.now()
    
    nf.fit(df=df_long)
    
    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Training completed in {training_time:.2f}s")
    
    # Generate predictions
    predictions = nf.predict()
    logger.info(f"Generated {len(predictions)} predictions")
    
    # Save model
    model_dir = Path(config['paths']['models_dir']) / "nbeats"
    model_dir.mkdir(parents=True, exist_ok=True)
    nf.save(path=str(model_dir), model_index=None, overwrite=True, save_dataset=True)
    
    # Save params
    params = {
        'horizon': nbeats_config.get('horizon', 5),
        'input_size': nbeats_config.get('input_size', 90),
        'learning_rate': nbeats_config.get('learning_rate', 0.001),
        'max_steps': nbeats_config.get('max_steps', 2000),
        'num_stacks': nbeats_config.get('num_stacks', 3),
    }
    with open(model_dir / 'params.json', 'w') as f:
        json.dump(params, f, indent=2)
    
    logger.info(f"Model saved to {model_dir}")
    
    return {
        'training_info': {
            'training_time_seconds': training_time,
            'n_samples': len(df_long),
            'n_coins': df_long['unique_id'].nunique(),
            **params
        },
        'predictions': predictions.to_dict('records'),
        'model_path': str(model_dir)
    }


def main():
    parser = argparse.ArgumentParser(description="N-BEATS Training Script")
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                        help="Path to configuration file")
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("N-BEATS Standalone Training Script")
    logger.info("=" * 50)
    
    # Load config
    config = load_config(args.config)
    
    nbeats_config = config.get('nbeats', {})
    if not nbeats_config.get('enabled', True):
        logger.info("N-BEATS is disabled in config")
        return
    
    # Load data
    logger.info("Loading data...")
    data = load_existing_data(config)
    
    if not data:
        logger.error("No data found. Please run data collection first.")
        sys.exit(1)
    
    # Prepare long format
    logger.info("Preparing data...")
    df_long = prepare_long_format(data)
    
    # Train
    logger.info("Training N-BEATS model...")
    results = train_nbeats(config, df_long)
    
    # Save results
    results_dir = Path(config['paths']['results_dir']) / "nbeats"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = results_dir / f"nbeats_global_results_{timestamp}.json"
    
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    logger.info(f"Results saved to {result_path}")
    logger.info("=" * 50)
    logger.info("N-BEATS training completed successfully!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
