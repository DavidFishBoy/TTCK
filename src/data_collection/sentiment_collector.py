import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import aiohttp
import pandas as pd

class SentimentCollector:
    
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
        self.api_url = api_url
        self.date_start = date_start
        self.date_end = date_end or datetime.now().strftime("%Y-%m-%d")
        self.data_dir = Path(data_dir)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
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
        for (low, high), label in SentimentCollector.FNG_LABELS.items():
            if low <= value <= high:
                return label
        return "Unknown"
    
    async def fetch_fear_greed_index(self) -> pd.DataFrame:
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
        df = df.copy()
        for lag in lag_periods:
            df[f"fng_lag_{lag}"] = df["fng_value"].shift(lag)
        
        self.logger.info(f"Added lag features for periods: {lag_periods}")
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str = "fear_greed_daily.csv") -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.data_dir / filename
        
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved sentiment data to {filepath}")
        
        return filepath
    
    def load_data(self, filename: str = "fear_greed_daily.csv") -> pd.DataFrame:
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            self.logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])
        
        self.logger.info(f"Loaded {len(df)} records from {filepath}")
        return df
    
    async def collect_and_save(self) -> pd.DataFrame:
        df = await self.fetch_fear_greed_index()
        
        if df.empty:
            self.logger.error("Failed to collect sentiment data")
            return df
        
        df = self.filter_date_range(df)
        
        df = self.add_lag_features(df)
        
        self.save_data(df)
        
        return df
    
    def get_extreme_events(
        self, 
        df: pd.DataFrame,
        fear_threshold: int = 25,
        greed_threshold: int = 75
    ) -> Dict[str, pd.DataFrame]:
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

def get_sentiment_data(refresh: bool = False) -> pd.DataFrame:
    collector = SentimentCollector()
    
    if not refresh:
        df = collector.load_data()
        if not df.empty:
            return df
    
    return asyncio.run(collector.collect_and_save())

def merge_sentiment_with_price(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    price_date_col: str = "date"
) -> pd.DataFrame:
    price_df = price_df.copy()
    sentiment_df = sentiment_df.copy()
    
    price_df[price_date_col] = pd.to_datetime(price_df[price_date_col])
    
    price_df["_merge_date"] = price_df[price_date_col].dt.date
    sentiment_df["_merge_date"] = sentiment_df["date"].dt.date
    
    merged = price_df.merge(
        sentiment_df[["_merge_date", "fng_value", "fng_label"]],
        on="_merge_date",
        how="left"
    )
    
    merged = merged.drop(columns=["_merge_date"])
    
    return merged

if __name__ == "__main__":
    async def test():
        collector = SentimentCollector()
        df = await collector.collect_and_save()
        print(f"\nCollected {len(df)} records")
        print(f"\nFirst 5 rows:\n{df.head()}")
        print(f"\nLast 5 rows:\n{df.tail()}")
        print(f"\nFear & Greed distribution:\n{df['fng_label'].value_counts()}")
        
        events = collector.get_extreme_events(df)
        print(f"\nExtreme Fear events: {len(events['extreme_fear'])}")
        print(f"Extreme Greed events: {len(events['extreme_greed'])}")
    
    asyncio.run(test())
