# src/data_collection/data_collector.py
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Tuple

import aiohttp
import pandas as pd
import numpy as np


class DataCollector:
    def __init__(self, coins: List[str], days: int, symbol_mapping: List[Dict[str, str]], coin_map: Dict[str, str], 
                 cryptocompare_api_key: str = None, cryptocompare_symbol_map: Dict[str, str] = None,
                 outlier_detection: bool = True, outlier_threshold: float = 3.0):
        """
        Initialize the DataCollector.

        Args:
            coins (List[str]): List of cryptocurrency names.
            days (int): Number of days of historical data to collect.
            symbol_mapping (List[Dict[str, str]]): Binance symbol mappings.
            coin_map (Dict[str, str]): Mapping of coin names to their base assets.
            cryptocompare_api_key (str): CryptoCompare API key for market cap data.
            cryptocompare_symbol_map (Dict[str, str]): Mapping of coin names to CryptoCompare symbols.
        """
        self.coins = coins
        self.days = days
        self.symbol_mapping = symbol_mapping
        self.coin_map = coin_map
        self.cryptocompare_api_key = cryptocompare_api_key
        self.cryptocompare_symbol_map = cryptocompare_symbol_map or {}
        self.binance_api = "https://api.binance.com/api/v3"
        self.cryptocompare_api = "https://min-api.cryptocompare.com/data"
        
        # Outlier detection settings (IQR method only)
        self.outlier_detection = outlier_detection
        self.outlier_threshold = outlier_threshold

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self._setup_logger()

        # Validate the timeframe for historical data
        self.validate_timeframe(days)

    def _setup_logger(self):
        """Set up logger with a consistent configuration."""
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def get_base_asset(self, coin: str) -> str:
        """
        Retrieve the base asset for a given coin name.

        Args:
            coin (str): Coin name (e.g., 'ethereum', 'litecoin').

        Returns:
            str: Base asset symbol (e.g., 'ETH', 'LTC').
        """
        base_asset = self.coin_map.get(coin.lower())
        if base_asset is None:
            base_asset = coin[:3].upper()
            self.logger.warning(f"Base asset for {coin} not found in coin_map. Using default: {base_asset}")
        return base_asset

    def get_binance_symbol(self, base_asset: str, quote_asset: str) -> str:
        """
        Retrieve the Binance trading symbol for a given base and quote asset.

        Args:
            base_asset (str): Base asset symbol (e.g., 'BTC').
            quote_asset (str): Quote asset symbol (e.g., 'USDT').

        Returns:
            str: Binance trading symbol (e.g., 'BTCUSDT'), or an empty string if not found.
        """
        for entry in self.symbol_mapping:
            if entry['baseAsset'] == base_asset and entry['quoteAsset'] == quote_asset:
                return entry['symbol']

        warning_message = f"No Binance symbol found for {base_asset}/{quote_asset}"
        if not hasattr(self, '_logged_warnings'):
            self._logged_warnings = set()
        if warning_message not in self._logged_warnings:
            self.logger.warning(warning_message)
            self._logged_warnings.add(warning_message)

        return ""

    @staticmethod
    def validate_timeframe(days: int) -> None:
        """
        Validate the timeframe for data collection.

        Args:
            days (int): Number of days of historical data.

        Raises:
            ValueError: If the timeframe is invalid.
        """
        if days <= 0:
            raise ValueError("Days must be a positive integer.")
        if days > 2000:
            raise ValueError("The maximum allowable timeframe is 2000 days.")

    async def fetch_binance_data(self, base_asset: str, quote_asset: str) -> pd.DataFrame:
        """
        Fetch historical data from the Binance API.

        Args:
            base_asset (str): Base asset (e.g., 'BTC').
            quote_asset (str): Quote asset (e.g., 'USDT').

        Returns:
            pd.DataFrame: DataFrame with OHLCV data, or an empty DataFrame on failure.
        """
        symbol = self.get_binance_symbol(base_asset, quote_asset)
        if not symbol:
            self.logger.warning(f"Binance symbol not found for {base_asset}/{quote_asset}")
            return pd.DataFrame()

        endpoint = f"{self.binance_api}/klines"
        params = {'symbol': symbol, 'interval': '1d', 'limit': self.days}

        self.logger.info(f"Fetching Binance data for symbol={symbol} with days={self.days}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, params=params) as response:
                    if response.status != 200:
                        raise Exception(f"Binance API request failed with status {response.status}")

                    data = await response.json()
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close',
                        'volume', 'close_time', 'quote_volume', 'trades',
                        'taker_buy_base', 'taker_buy_quote', 'ignored'
                    ])

                    # Format and process DataFrame
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

                    self.logger.info(f"Successfully fetched Binance data for {symbol}")
                    return df

        except Exception as e:
            self.logger.error(f"Error fetching Binance data for {base_asset}/{quote_asset}: {str(e)}")
            return pd.DataFrame()

    async def fetch_cryptocompare_market_cap(self, coin_name: str) -> pd.DataFrame:
        """
        Fetch historical market cap data from CryptoCompare API.

        Args:
            coin_name (str): Coin name (e.g., 'bitcoin', 'ethereum').

        Returns:
            pd.DataFrame: DataFrame with market cap data, or an empty DataFrame on failure.
        """
        # Get the CryptoCompare symbol from mapping
        symbol = self.cryptocompare_symbol_map.get(coin_name, self.coin_map.get(coin_name, coin_name.upper()))
        
        # Use histoday endpoint to get daily OHLCV data which includes market cap info
        endpoint = f"{self.cryptocompare_api}/v2/histoday"
        
        params = {
            'fsym': symbol,
            'tsym': 'USD',
            'limit': min(self.days, 2000),  # CryptoCompare max limit is 2000
            'api_key': self.cryptocompare_api_key
        }

        self.logger.info(f"Fetching CryptoCompare market cap data for {coin_name} (symbol: {symbol}) with days={self.days}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"CryptoCompare API request failed with status {response.status}: {error_text}")

                    data = await response.json()
                    
                    if 'Data' not in data or 'Data' not in data['Data']:
                        raise Exception("No data in response")
                    
                    # Extract data from response
                    historical_data = data['Data']['Data']
                    market_cap_list = []
                    
                    for entry in historical_data:
                        timestamp = entry['time']
                        # Calculate market cap as close price * volumefrom (trading volume in base currency)
                        # This is an approximation since free API doesn't provide direct market cap
                        close_price = entry.get('close', 0)
                        volume = entry.get('volumefrom', 0)
                        # Better approximation: use volumeto (volume in USD) as market indicator
                        market_cap = entry.get('volumeto', close_price * volume)
                        
                        market_cap_list.append({
                            'timestamp': timestamp,
                            'market_cap': market_cap
                        })
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(market_cap_list)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.set_index('timestamp', inplace=True)
                    df['market_cap'] = df['market_cap'].astype(float)

                    self.logger.info(f"Successfully fetched {len(df)} market cap records for {coin_name}")
                    return df

        except Exception as e:
            self.logger.error(f"Error fetching CryptoCompare market cap for {coin_name}: {str(e)}")
            return pd.DataFrame()

    def process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw data by handling missing values and basic data cleaning.

        Args:
            df (pd.DataFrame): Raw DataFrame to process.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        if df.empty:
            self.logger.warning("Received an empty DataFrame for processing")
            return df

        if df.isnull().any().any():
            self.logger.warning("Found missing values in the data. Interpolating and backfilling.")
            df = df.interpolate(method='time')
            df = df.bfill()

        # Apply outlier detection and handling if enabled
        if self.outlier_detection:
            df = self.handle_outliers(df)

        return df
    
    def detect_outliers_iqr(self, series: pd.Series, threshold: float = 1.5) -> pd.Series:
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        df_cleaned = df.copy()
        columns_to_check = ['open', 'high', 'low', 'close', 'volume']
        
        # Filter only existing columns
        columns_to_check = [col for col in columns_to_check if col in df_cleaned.columns]
        
        if not columns_to_check:
            self.logger.warning("No OHLCV columns found for outlier detection")
            return df_cleaned
        
        total_outliers = 0
        outlier_details = {}
        
        for col in columns_to_check:
            # Detect outliers using IQR method
            outliers_mask = self.detect_outliers_iqr(df_cleaned[col], self.outlier_threshold)
            
            num_outliers = outliers_mask.sum()
            
            if num_outliers > 0:
                total_outliers += num_outliers
                outlier_details[col] = num_outliers
                
                # Handle outliers using interpolation
                # Store original values for logging
                original_values = df_cleaned.loc[outliers_mask, col].copy()
                
                # Replace outliers with NaN then interpolate
                df_cleaned.loc[outliers_mask, col] = np.nan
                df_cleaned[col] = df_cleaned[col].interpolate(method='linear', limit_direction='both')
                
                # If still NaN (edge cases), use forward/backward fill
                df_cleaned[col] = df_cleaned[col].ffill().bfill()
                
                # Log details for small number of outliers
                if num_outliers <= 5:
                    for idx in original_values.index:
                        original = original_values[idx]
                        replaced = df_cleaned.loc[idx, col]
                        self.logger.debug(
                            f"Outlier in {col} at {idx}: {original:.2f} → {replaced:.2f}"
                        )
        
        if total_outliers > 0:
            self.logger.info(
                f"Detected and handled {total_outliers} outliers using IQR method "
                f"(threshold={self.outlier_threshold})"
            )
            for col, count in outlier_details.items():
                self.logger.info(f"  - {col}: {count} outliers")
        else:
            self.logger.debug("No outliers detected in the data")
        
        return df_cleaned

    async def collect_all_data(self, coins: List[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect data from Binance and CryptoCompare for the specified coins.

        Args:
            coins (List[str]): List of coin names. If None, use the object's configured coins.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: Data dictionary with binance data (including market cap) for each coin.
        """
        coins_to_process = coins if coins else self.coins
        self.logger.info(f"Starting collection of data for {len(coins_to_process)} coins: {', '.join(coins_to_process)}")
        all_data = {}
        
        successful_coins = []
        failed_coins = []

        for coin in coins_to_process:
            try:
                base_asset = self.get_base_asset(coin)
                quote_asset = 'USDT'

                # Fetch Binance data
                binance_data = await self.fetch_binance_data(base_asset, quote_asset)
                binance_data = self.process_raw_data(binance_data)

                # Fetch CryptoCompare market cap data if API key is available
                market_cap_data = pd.DataFrame()
                if self.cryptocompare_api_key:
                    market_cap_data = await self.fetch_cryptocompare_market_cap(coin)
                    market_cap_data = self.process_raw_data(market_cap_data)

                # Merge market cap data with binance data
                if not binance_data.empty and not market_cap_data.empty:
                    # Merge on index (timestamp) - use left join to keep all Binance data
                    merged_data = binance_data.merge(
                        market_cap_data, 
                        left_index=True, 
                        right_index=True, 
                        how='left'
                    )
                    # Fill missing market cap values using forward fill then backward fill
                    merged_data['market_cap'] = merged_data['market_cap'].ffill().bfill()
                    
                    all_data[coin] = {
                        'binance': merged_data
                    }
                    successful_coins.append(coin)
                    self.logger.info(f"Successfully collected {len(merged_data)} records for {coin} (with market cap)")
                elif not binance_data.empty:
                    # If only binance data available, use it anyway
                    all_data[coin] = {
                        'binance': binance_data
                    }
                    successful_coins.append(coin)
                    self.logger.warning(f"⚠️ Collected {len(binance_data)} records for {coin} (without market cap)")
                else:
                    failed_coins.append(coin)
                    self.logger.warning(f"❌ No data collected for {coin}")

            except Exception as e:
                failed_coins.append(coin)
                self.logger.error(f"❌ Error collecting data for {coin}: {str(e)}")

        # Summary
        self.logger.info("="*60)
        self.logger.info(f"Collection Summary:")
        self.logger.info(f"Successful: {len(successful_coins)}/{len(coins_to_process)} coins")
        self.logger.info(f"Failed: {len(failed_coins)}/{len(coins_to_process)} coins")
        if successful_coins:
            self.logger.info(f"  Success list: {', '.join(successful_coins)}")
        if failed_coins:
            self.logger.warning(f"  Failed list: {', '.join(failed_coins)}")
        self.logger.info("="*60)
        
        return all_data

    def save_data(self, data: Dict[str, Dict[str, pd.DataFrame]], save_dir: Union[str, Path]) -> None:
        """
        Save collected data to CSV files.

        Args:
            data (dict): Dictionary containing the collected data.
            save_dir (Union[str, Path]): Directory to save the files.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving collected data to directory: {save_dir}")

        saved_count = 0
        skipped_count = 0
        
        for coin, sources in data.items():
            for source, df in sources.items():
                if df.empty or not isinstance(df, pd.DataFrame):
                    self.logger.warning(f"Data for {coin} from {source} is empty or invalid. Skipping save.")
                    skipped_count += 1
                    continue
                filename = f"{coin}_{source}_{datetime.now().strftime('%Y%m%d')}.csv"
                filepath = save_dir / filename
                try:
                    df.to_csv(filepath)
                    saved_count += 1
                    self.logger.info(f"Saved {coin} ({len(df)} records) to {filename}")
                except Exception as e:
                    skipped_count += 1
                    self.logger.error(f"Failed to save data for {coin} from {source}: {str(e)}")
        
        # Summary
        self.logger.info("="*60)
        self.logger.info(f"Save Summary:")
        self.logger.info(f"Successfully saved: {saved_count} files")
        self.logger.info(f"Skipped: {skipped_count} files")
        self.logger.info(f"Location: {save_dir.absolute()}")
        self.logger.info("="*60)


if __name__ == "__main__":
    import asyncio

    # Example configuration
    test_coins = ["bitcoin"]
    test_days = 30
    test_symbol_mapping = [
        {
            "symbol": "BTCUSDT",
            "baseAsset": "BTC",
            "quoteAsset": "USDT"
        }
    ]
    test_coin_map = {"bitcoin": "BTC"}

    # Initialize DataCollector
    collector = DataCollector(
        coins=test_coins,
        days=test_days,
        symbol_mapping=test_symbol_mapping,
        coin_map=test_coin_map
    )

    # Specify save directory
    save_dir = "data/raw/train"

    # Run data collection
    data = asyncio.run(collector.collect_all_data())
    collector.save_data(data, save_dir)
