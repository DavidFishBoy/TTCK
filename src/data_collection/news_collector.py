import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

import aiohttp
import pandas as pd
import re

# VADER for sentiment scoring
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


class NewsCollector:
    """
    Thu thập tin tức crypto từ NewsAPI và chấm sentiment.
    
    Yêu cầu: API key từ newsapi.org (free tier: 100 requests/ngày)
    """
    
    def __init__(
        self,
        api_key: str = "41a892afbab4440781da3aa950ee741d",
        keywords: List[str] = None,
        days_back: int = 7,
        data_dir: str = "data/sentiment"
    ):
        """
        Initialize the NewsCollector.
        
        Args:
            api_key: NewsAPI key.
            keywords: List of keywords to search for.
            days_back: Number of days to fetch news for.
            data_dir: Directory to save data.
        """
        self.api_key = api_key
        self.keywords = keywords or ["crypto", "cryptocurrency"]
        self.days_back = days_back
        self.data_dir = Path(data_dir)
        self.base_url = "https://newsapi.org/v2/everything"
        self.logger = self._setup_logger()
        
        # Initialize VADER
        if VADER_AVAILABLE:
            self.sia = SentimentIntensityAnalyzer()
        else:
            self.sia = None
            self.logger.warning("VADER not available. Install nltk: pip install nltk")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger with a consistent configuration."""
        logger = logging.getLogger("NewsCollector")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis."""
        if not text:
            return ""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def score_sentiment(self, text: str) -> Dict[str, float]:
        """
        Score sentiment using VADER.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Dictionary with compound, pos, neg, neu scores.
        """
        if not self.sia or not text:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        
        cleaned = self._clean_text(text)
        if not cleaned:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        
        scores = self.sia.polarity_scores(cleaned)
        return scores
    
    def get_sentiment_label(self, compound: float) -> str:
        """
        Get sentiment label from compound score.
        
        Args:
            compound: VADER compound score.
            
        Returns:
            Label: positive, negative, or neutral.
        """
        if compound >= 0.05:
            return "positive"
        elif compound <= -0.05:
            return "negative"
        else:
            return "neutral"
    
    async def fetch_news(self, keyword: str = None) -> List[Dict]:
        """
        Fetch news articles from NewsAPI.
        
        Args:
            keyword: Optional specific keyword to search.
            
        Returns:
            List of article dictionaries.
        """
        query = keyword or " OR ".join(self.keywords)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)
        
        params = {
            "q": query,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,  # Max allowed
            "apiKey": self.api_key
        }
        
        self.logger.info(f"Fetching news for '{query}' from {start_date.date()} to {end_date.date()}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params, timeout=30) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        self.logger.error(f"API error: {error_data.get('message', 'Unknown error')}")
                        return []
                    
                    data = await response.json()
                    
            articles = data.get("articles", [])
            self.logger.info(f"Fetched {len(articles)} articles")
            return articles
            
        except asyncio.TimeoutError:
            self.logger.error("API request timed out")
            return []
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            return []
    
    def process_articles(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Process articles and score sentiment.
        
        Args:
            articles: List of article dictionaries.
            
        Returns:
            DataFrame with processed articles.
        """
        if not articles:
            return pd.DataFrame()
        
        records = []
        for article in articles:
            title = article.get("title", "") or ""
            description = article.get("description", "") or ""
            published_at = article.get("publishedAt", "")
            source = article.get("source", {}).get("name", "Unknown")
            url = article.get("url", "")
            
            # Skip removed articles
            if "[Removed]" in title or not title:
                continue
            
            # Combine title and description for sentiment
            full_text = f"{title} {description}"
            
            # Score sentiment
            scores = self.score_sentiment(full_text)
            compound = scores["compound"]
            label = self.get_sentiment_label(compound)
            
            # Parse date
            try:
                date = pd.to_datetime(published_at).date()
            except:
                continue
            
            records.append({
                "date": date,
                "title": title,
                "description": description,
                "source": source,
                "url": url,
                "sentiment_score": compound,
                "sentiment_label": label,
                "pos": scores["pos"],
                "neg": scores["neg"],
                "neu": scores["neu"]
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date", ascending=False).reset_index(drop=True)
        
        self.logger.info(f"Processed {len(df)} articles after filtering")
        return df
    
    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment by day.
        
        Args:
            df: DataFrame with article-level sentiment.
            
        Returns:
            DataFrame with daily aggregated sentiment.
        """
        if df.empty:
            return pd.DataFrame()
        
        daily = df.groupby("date").agg({
            "sentiment_score": ["mean", "median", "std"],
            "title": "count",
            "sentiment_label": lambda x: (x == "positive").sum() / len(x)
        }).reset_index()
        
        # Flatten column names
        daily.columns = [
            "date", 
            "news_sentiment_mean", 
            "news_sentiment_median",
            "news_sentiment_std",
            "news_count",
            "news_positive_ratio"
        ]
        
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date").reset_index(drop=True)
        
        self.logger.info(f"Aggregated to {len(daily)} daily records")
        return daily
    
    def save_data(
        self, 
        articles_df: pd.DataFrame, 
        daily_df: pd.DataFrame
    ) -> Dict[str, Path]:
        """
        Save data to CSV files.
        
        Args:
            articles_df: Article-level DataFrame.
            daily_df: Daily aggregated DataFrame.
            
        Returns:
            Dictionary with paths to saved files.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Save articles
        articles_path = self.data_dir / "news_articles.csv"
        articles_df.to_csv(articles_path, index=False)
        paths["articles"] = articles_path
        self.logger.info(f"Saved articles to {articles_path}")
        
        # Save daily
        daily_path = self.data_dir / "news_sentiment_daily.csv"
        daily_df.to_csv(daily_path, index=False)
        paths["daily"] = daily_path
        self.logger.info(f"Saved daily sentiment to {daily_path}")
        
        return paths
    
    def load_articles(self) -> pd.DataFrame:
        """Load saved articles data."""
        filepath = self.data_dir / "news_articles.csv"
        if not filepath.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])
        return df
    
    def load_daily(self) -> pd.DataFrame:
        """Load saved daily sentiment data."""
        filepath = self.data_dir / "news_sentiment_daily.csv"
        if not filepath.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])
        return df
    
    async def collect_and_save(self) -> Dict[str, pd.DataFrame]:
        """
        Main method: fetch news, process, aggregate, and save.
        
        Returns:
            Dictionary with 'articles' and 'daily' DataFrames.
        """
        # Fetch news
        articles = await self.fetch_news()
        
        if not articles:
            self.logger.error("No articles fetched")
            return {"articles": pd.DataFrame(), "daily": pd.DataFrame()}
        
        # Process articles
        articles_df = self.process_articles(articles)
        
        if articles_df.empty:
            self.logger.error("No valid articles after processing")
            return {"articles": pd.DataFrame(), "daily": pd.DataFrame()}
        
        # Aggregate daily
        daily_df = self.aggregate_daily(articles_df)
        
        # Save
        self.save_data(articles_df, daily_df)
        
        return {"articles": articles_df, "daily": daily_df}


# Utility function
def get_news_sentiment_data(refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Get news sentiment data, loading from cache or fetching fresh.
    
    Args:
        refresh: If True, force refresh from API.
        
    Returns:
        Dictionary with 'articles' and 'daily' DataFrames.
    """
    collector = NewsCollector()
    
    if not refresh:
        articles = collector.load_articles()
        daily = collector.load_daily()
        if not articles.empty and not daily.empty:
            return {"articles": articles, "daily": daily}
    
    # Fetch fresh data
    return asyncio.run(collector.collect_and_save())


if __name__ == "__main__":
    # Test the collector
    async def test():
        collector = NewsCollector()
        result = await collector.collect_and_save()
        
        print(f"\n=== Articles ({len(result['articles'])}) ===")
        if not result['articles'].empty:
            print(result['articles'][['date', 'title', 'sentiment_score', 'sentiment_label']].head(10))
        
        print(f"\n=== Daily Sentiment ({len(result['daily'])}) ===")
        if not result['daily'].empty:
            print(result['daily'])
    
    asyncio.run(test())
