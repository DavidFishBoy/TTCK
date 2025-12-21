# src/assistant/chart_analyzer.py

"""
ChartAnalyzer - Phân tích biểu đồ bằng GPT-4 với hệ thống cache.
Tích hợp vào dashboard Streamlit để cung cấp phân tích AI cho mỗi biểu đồ.
"""

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

# Load .env file for API key
from dotenv import load_dotenv
load_dotenv()

from .prompts import get_prompt, get_system_prompt


class ChartAnalyzer:
    """
    Phân tích biểu đồ cryptocurrency bằng GPT-4o-mini.
    
    Features:
    - Prompt templates riêng cho từng loại biểu đồ
    - Cache kết quả để tiết kiệm API calls
    - Tích hợp dễ dàng với Streamlit
    
    Example:
        analyzer = ChartAnalyzer()
        result = analyzer.analyze_chart(
            coin="bitcoin",
            chart_type="rolling_volatility",
            chart_data={"vol_14d_latest": 3.5, ...},
            chart_title="Biến Động Lăn"
        )
        st.markdown(result)
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        cache_enabled: bool = True,
        cache_duration_hours: int = 24,
        cache_dir: str = "data/cache/chart_analysis",
        model: str = "gpt-4o-mini"
    ):
        """
        Khởi tạo ChartAnalyzer.
        
        Args:
            api_key: OpenAI API key. Nếu None, lấy từ OPENAI_API_KEY env var.
            cache_enabled: Bật/tắt cache.
            cache_duration_hours: Thời gian cache hết hạn (giờ).
            cache_dir: Thư mục lưu cache.
            model: Tên model OpenAI (gpt-4o-mini, gpt-4o, gpt-4-turbo, etc.)
        """
        # API key
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.api_key = api_key
        
        # Cache settings
        self.cache_enabled = cache_enabled
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.cache_dir = Path(cache_dir)
        
        # Model
        self.model = model
        
        # OpenAI client
        self.client = None
        self._init_openai()
        
        # Ensure cache directory exists
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("⚠️ openai package not installed. Run: pip install openai")
            except Exception as e:
                print(f"⚠️ Failed to initialize OpenAI: {e}")
    
    def _generate_cache_key(
        self, 
        coin: str, 
        chart_type: str, 
        chart_data: Dict
    ) -> str:
        """
        Tạo cache key dựa trên coin, chart_type và data hash.
        
        Returns:
            Cache key string: {coin}_{chart_type}_{data_hash}_{date}
        """
        # Hash chart_data để tạo unique key
        data_str = json.dumps(chart_data, sort_keys=True, default=str)
        data_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]
        
        # Include current date để cache hết hạn khi ngày mới
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        return f"{coin}_{chart_type}_{data_hash}_{date_str}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to cache file."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _get_cached(
        self, 
        coin: str, 
        chart_type: str, 
        chart_data: Dict
    ) -> Optional[str]:
        """
        Lấy kết quả từ cache nếu còn hạn.
        
        Returns:
            Cached analysis string hoặc None nếu không có/hết hạn.
        """
        if not self.cache_enabled:
            return None
        
        cache_key = self._generate_cache_key(coin, chart_type, chart_data)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check expiration
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cached_time > self.cache_duration:
                # Cache expired
                cache_path.unlink()  # Delete expired cache
                return None
            
            return cache_data['analysis']
            
        except Exception:
            return None
    
    def _save_cache(
        self, 
        coin: str, 
        chart_type: str, 
        chart_data: Dict,
        analysis: str
    ) -> None:
        """Lưu kết quả phân tích vào cache."""
        if not self.cache_enabled:
            return
        
        cache_key = self._generate_cache_key(coin, chart_type, chart_data)
        cache_path = self._get_cache_path(cache_key)
        
        cache_data = {
            'coin': coin,
            'chart_type': chart_type,
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")
    
    def _build_prompt(
        self, 
        chart_type: str, 
        coin: str,
        chart_data: Dict,
        chart_title: str
    ) -> str:
        """
        Xây dựng prompt hoàn chỉnh từ template và data.
        
        Args:
            chart_type: Loại biểu đồ
            coin: Tên coin
            chart_data: Dữ liệu từ biểu đồ
            chart_title: Tiêu đề biểu đồ
            
        Returns:
            Prompt string đã điền data
        """
        template = get_prompt(chart_type)
        
        if not template:
            return f"""## PHÂN TÍCH BIỂU ĐỒ

**Coin:** {coin}
**Tiêu đề:** {chart_title}

### DỮ LIỆU:
{json.dumps(chart_data, ensure_ascii=False, indent=2)}

### YÊU CẦU:
Hãy phân tích biểu đồ này và đưa ra nhận xét chi tiết về ý nghĩa của dữ liệu.
"""
        
        # Prepare data for formatting
        format_data = {
            'coin': coin,
            'chart_title': chart_title,
            **chart_data
        }
        
        try:
            return template.format(**format_data)
        except KeyError as e:
            # Handle missing keys gracefully
            return template + f"\n\n**Dữ liệu bổ sung:** {json.dumps(chart_data, ensure_ascii=False)}"
    
    def _call_openai(self, prompt: str) -> str:
        """
        Gọi OpenAI API để phân tích.
        
        Args:
            prompt: User prompt
            
        Returns:
            Phân tích từ GPT
        """
        if not self.client:
            return self._get_fallback_analysis(prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_str = str(e)
            if "insufficient_quota" in error_str.lower():
                return f"❌ **Hết quota API:** Vui lòng nạp thêm credit tại [platform.openai.com/account/billing](https://platform.openai.com/account/billing)\n\n*Chi tiết: {error_str}*"
            return f"❌ **Lỗi khi gọi API:** {error_str}\n\nVui lòng kiểm tra API key và kết nối mạng."
    
    def _get_fallback_analysis(self, prompt: str) -> str:
        """
        Fallback khi không có API key - trả về hướng dẫn.
        """
        return """⚠️ **Chưa cấu hình API Key**

Để sử dụng tính năng phân tích AI, vui lòng:

1. **Lấy API key từ OpenAI:**
   - Truy cập [platform.openai.com](https://platform.openai.com)
   - Tạo API key mới

2. **Thêm vào file `.env`:**
   ```
   OPENAI_API_KEY=sk-proj-xxxxx
   ```

3. **Khởi động lại dashboard**

---

💡 *Model gpt-4o-mini rất rẻ: ~$0.15/1M tokens input*
"""
    
    def analyze_chart(
        self,
        coin: str,
        chart_type: str,
        chart_data: Dict[str, Any],
        chart_title: str,
        force_refresh: bool = False
    ) -> str:
        """
        Phân tích một biểu đồ cụ thể.
        
        Args:
            coin: Tên coin (ví dụ: "bitcoin", "ethereum")
            chart_type: Loại biểu đồ (từ prompts.CHART_PROMPTS keys)
            chart_data: Dictionary chứa dữ liệu từ biểu đồ
            chart_title: Tiêu đề hiển thị của biểu đồ
            force_refresh: Bỏ qua cache và gọi API mới
            
        Returns:
            Phân tích chi tiết dưới dạng markdown string
        """
        coin = coin.lower()
        
        # Step 1: Check cache
        if not force_refresh:
            cached = self._get_cached(coin, chart_type, chart_data)
            if cached:
                return cached + "\n\n---\n*📦 Từ cache - Click để làm mới*"
        
        # Step 2: Build prompt
        prompt = self._build_prompt(chart_type, coin, chart_data, chart_title)
        
        # Step 3: Call OpenAI
        analysis = self._call_openai(prompt)
        
        # Step 4: Save to cache
        if "❌" not in analysis and "⚠️ **Chưa cấu hình" not in analysis:
            self._save_cache(coin, chart_type, chart_data, analysis)
        
        return analysis
    
    def clear_cache(self, coin: Optional[str] = None) -> int:
        """
        Xóa cache.
        
        Args:
            coin: Nếu chỉ định, chỉ xóa cache của coin đó. 
                  Nếu None, xóa toàn bộ cache.
                  
        Returns:
            Số file cache đã xóa
        """
        if not self.cache_dir.exists():
            return 0
        
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if coin is None or cache_file.name.startswith(coin):
                cache_file.unlink()
                count += 1
        
        return count
    
    def get_cache_stats(self) -> Dict:
        """
        Lấy thống kê cache.
        
        Returns:
            Dictionary với thông tin cache
        """
        if not self.cache_dir.exists():
            return {"total_files": 0, "total_size_kb": 0}
        
        files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "total_files": len(files),
            "total_size_kb": round(total_size / 1024, 2),
            "cache_dir": str(self.cache_dir)
        }


# Singleton instance for easy import
_analyzer_instance: Optional[ChartAnalyzer] = None


def get_chart_analyzer() -> ChartAnalyzer:
    """
    Lấy singleton instance của ChartAnalyzer.
    Tiện lợi để sử dụng trong Streamlit mà không cần khởi tạo nhiều lần.
    
    Returns:
        ChartAnalyzer instance
        
    Example:
        from src.assistant.chart_analyzer import get_chart_analyzer
        
        analyzer = get_chart_analyzer()
        result = analyzer.analyze_chart(...)
    """
    global _analyzer_instance
    
    if _analyzer_instance is None:
        _analyzer_instance = ChartAnalyzer()
    
    return _analyzer_instance
