import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import aiohttp
import pandas as pd


class SentimentCollector:
    """
    Thu thập dữ liệu Fear & Greed Index từ Alternative.me API.
    
    API là miễn phí và không cần API key.
    """
    
    # Fear & Greed classifications
    FNG_LABELS = {
        (0, 25): "Extreme Fear",
        (26, 49): "Fear", 
        (50, 59): "Neutral",
        (60, 74): "Greed",
        (75, 100): "Extreme Greed"
    }
    
    def __init__(
        self,
        api_url: str = "https://api.alternative.me/fng/?limit=0",
        date_start: str = "2023-03-24",
        date_end: Optional[str] = None,
        data_dir: str = "data/sentiment"
    ):
        """
        Initialize the SentimentCollector.
        
        Args:
            api_url: URL của Fear & Greed Index API.
            date_start: Ngày bắt đầu lấy dữ liệu (YYYY-MM-DD).
            date_end: Ngày kết thúc lấy dữ liệu, mặc định là ngày hiện tại.
            data_dir: Thư mục lưu dữ liệu sentiment.
        """
        self.api_url = api_url
        self.date_start = date_start
        self.date_end = date_end or datetime.now().strftime("%Y-%m-%d")
        self.data_dir = Path(data_dir)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logger with a consistent configuration."""
        logger = logging.getLogger("SentimentCollector")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @staticmethod
    def _get_fng_label(value: int) -> str:
        """
        Get the classification label for a Fear & Greed value.
        
        Args:
            value: Fear & Greed index value (0-100).
            
        Returns:
            Classification label as string.
        """
        for (low, high), label in SentimentCollector.FNG_LABELS.items():
            if low <= value <= high:
                return label
        return "Unknown"
    
    async def fetch_fear_greed_index(self) -> pd.DataFrame:
        """
        Fetch historical Fear & Greed Index data from API.
        
        Returns:
            DataFrame with columns: date, fng_value, fng_label, timestamp.
        """
        self.logger.info("Fetching Fear & Greed Index data from API...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, timeout=30) as response:
                    if response.status != 200:
                        self.logger.error(f"API request failed with status {response.status}")
                        return pd.DataFrame()
                    
                    data = await response.json()
                    
            if "data" not in data:
                self.logger.error("Invalid API response: 'data' key not found")
                return pd.DataFrame()
            
            # Parse API response
            records = []
            for item in data["data"]:
                timestamp = int(item["timestamp"])
                value = int(item["value"])
                date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
                label = item.get("value_classification", self._get_fng_label(value))
                
                records.append({
                    "date": date,
                    "fng_value": value,
                    "fng_label": label,
                    "timestamp": timestamp
                })
            
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            
            self.logger.info(f"Successfully fetched {len(df)} records from API")
            return df
            
        except asyncio.TimeoutError:
            self.logger.error("API request timed out")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def filter_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to the specified date range.
        
        Args:
            df: DataFrame with 'date' column.
            
        Returns:
            Filtered DataFrame.
        """
        if df.empty:
            return df
            
        start = pd.to_datetime(self.date_start)
        end = pd.to_datetime(self.date_end)
        
        mask = (df["date"] >= start) & (df["date"] <= end)
        filtered = df.loc[mask].copy()
        
        self.logger.info(
            f"Filtered data from {self.date_start} to {self.date_end}: "
            f"{len(filtered)} records"
        )
        return filtered
    
    def add_lag_features(
        self, 
        df: pd.DataFrame, 
        lag_periods: List[int] = [0, 1, 3, 7, 14]
    ) -> pd.DataFrame:
        """
        Add lagged sentiment features for correlation analysis.
        
        Args:
            df: DataFrame with 'fng_value' column.
            lag_periods: List of lag periods to create.
            
        Returns:
            DataFrame with additional lag columns.
        """
        df = df.copy()
        for lag in lag_periods:
            df[f"fng_lag_{lag}"] = df["fng_value"].shift(lag)
        
        self.logger.info(f"Added lag features for periods: {lag_periods}")
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str = "fear_greed_daily.csv") -> Path:
        """
        Save sentiment data to CSV file.
        
        Args:
            df: DataFrame to save.
            filename: Output filename.
            
        Returns:
            Path to saved file.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.data_dir / filename
        
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved sentiment data to {filepath}")
        
        return filepath
    
    def load_data(self, filename: str = "fear_greed_daily.csv") -> pd.DataFrame:
        """
        Load sentiment data from CSV file.
        
        Args:
            filename: Input filename.
            
        Returns:
            DataFrame with sentiment data, or empty DataFrame if file not found.
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            self.logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])
        
        self.logger.info(f"Loaded {len(df)} records from {filepath}")
        return df
    
    async def collect_and_save(self) -> pd.DataFrame:
        """
        Main method: fetch, filter, and save sentiment data.
        
        Returns:
            Processed DataFrame.
        """
        # Fetch from API
        df = await self.fetch_fear_greed_index()
        
        if df.empty:
            self.logger.error("Failed to collect sentiment data")
            return df
        
        # Filter to date range
        df = self.filter_date_range(df)
        
        # Add lag features
        df = self.add_lag_features(df)
        
        # Save to file
        self.save_data(df)
        
        return df
    
    def get_extreme_events(
        self, 
        df: pd.DataFrame,
        fear_threshold: int = 25,
        greed_threshold: int = 75
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract extreme fear and greed events for event study.
        
        Args:
            df: DataFrame with 'fng_value' column.
            fear_threshold: Maximum value for extreme fear.
            greed_threshold: Minimum value for extreme greed.
            
        Returns:
            Dictionary with 'extreme_fear' and 'extreme_greed' DataFrames.
        """
        extreme_fear = df[df["fng_value"] <= fear_threshold].copy()
        extreme_greed = df[df["fng_value"] >= greed_threshold].copy()
        
        self.logger.info(
            f"Found {len(extreme_fear)} extreme fear events and "
            f"{len(extreme_greed)} extreme greed events"
        )
        
        return {
            "extreme_fear": extreme_fear,
            "extreme_greed": extreme_greed
        }


# Utility functions for easy access
def get_sentiment_data(refresh: bool = False) -> pd.DataFrame:
    """
    Get sentiment data, loading from cache or fetching from API.
    
    Args:
        refresh: If True, force refresh from API.
        
    Returns:
        DataFrame with sentiment data.
    """
    collector = SentimentCollector()
    
    if not refresh:
        df = collector.load_data()
        if not df.empty:
            return df
    
    # Fetch fresh data
    return asyncio.run(collector.collect_and_save())


def merge_sentiment_with_price(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    price_date_col: str = "date"
) -> pd.DataFrame:
    """
    Merge sentiment data with price data on date.
    
    Args:
        price_df: DataFrame with price data.
        sentiment_df: DataFrame with sentiment data.
        price_date_col: Name of date column in price DataFrame.
        
    Returns:
        Merged DataFrame.
    """
    # Ensure date columns are datetime
    price_df = price_df.copy()
    sentiment_df = sentiment_df.copy()
    
    price_df[price_date_col] = pd.to_datetime(price_df[price_date_col])
    
    # Extract date only (no time) for proper merging
    price_df["_merge_date"] = price_df[price_date_col].dt.date
    sentiment_df["_merge_date"] = sentiment_df["date"].dt.date
    
    # Merge on date
    merged = price_df.merge(
        sentiment_df[["_merge_date", "fng_value", "fng_label"]],
        on="_merge_date",
        how="left"
    )
    
    # Clean up
    merged = merged.drop(columns=["_merge_date"])
    
    return merged


if __name__ == "__main__":
    # Test the collector
    async def test():
        collector = SentimentCollector()
        df = await collector.collect_and_save()
        print(f"\nCollected {len(df)} records")
        print(f"\nFirst 5 rows:\n{df.head()}")
        print(f"\nLast 5 rows:\n{df.tail()}")
        print(f"\nFear & Greed distribution:\n{df['fng_label'].value_counts()}")
        
        # Test extreme events
        events = collector.get_extreme_events(df)
        print(f"\nExtreme Fear events: {len(events['extreme_fear'])}")
        print(f"Extreme Greed events: {len(events['extreme_greed'])}")
    
    asyncio.run(test())
