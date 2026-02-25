
import json
import hashlib
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

from dotenv import load_dotenv
load_dotenv()

from .prompts import get_prompt, get_system_prompt

class ChartAnalyzer:
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        cache_enabled: bool = True,
        cache_duration_hours: int = 24,
        cache_dir: str = "data/cache/chart_analysis",
        model: Optional[str] = None
    ):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        self.api_key = api_key
        
        if model is None:
            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.model = model
        
        self.cache_enabled = cache_enabled
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.cache_dir = Path(cache_dir)
        
        self.client = None
        self._init_gemini()
        
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_gemini(self):
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except ImportError:
                print("⚠️ google-genai package not installed. Run: pip install google-genai")
            except Exception as e:
                print(f"⚠️ Failed to initialize Gemini: {e}")
    
    def _generate_cache_key(
        self, 
        coin: str, 
        chart_type: str, 
        chart_data: Dict
    ) -> str:
        data_str = json.dumps(chart_data, sort_keys=True, default=str)
        data_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        return f"{coin}_{chart_type}_{data_hash}_{date_str}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"
    
    def _get_cached(
        self, 
        coin: str, 
        chart_type: str, 
        chart_data: Dict
    ) -> Optional[str]:
        if not self.cache_enabled:
            return None
        
        cache_key = self._generate_cache_key(coin, chart_type, chart_data)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cached_time > self.cache_duration:
                cache_path.unlink()
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
        template = get_prompt(chart_type)
        
        if not template:
            return f"""## PHÂN TÍCH BIỂU ĐỒ

**Coin:** {coin}
**Tiêu đề:** {chart_title}

{json.dumps(chart_data, ensure_ascii=False, indent=2)}

Hãy phân tích biểu đồ này và đưa ra nhận xét chi tiết về ý nghĩa của dữ liệu.
"""
        
        format_data = {
            'coin': coin,
            'chart_title': chart_title,
            **chart_data
        }
        
        try:
            return template.format(**format_data)
        except KeyError as e:
            return template + f"\n\n**Dữ liệu bổ sung:** {json.dumps(chart_data, ensure_ascii=False)}"
    
    def _call_gemini(self, prompt: str) -> str:
        if not self.client:
            return self._get_fallback_analysis(prompt)
        
        try:
            # Combine system prompt with user prompt for Gemini
            system_prompt = get_system_prompt()
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config={
                    "temperature": 0.7,
                    "max_output_tokens": 2000,
                }
            )
            
            return response.text
            
        except Exception as e:
            error_str = str(e)
            if "quota" in error_str.lower() or "rate" in error_str.lower():
                return f"❌ **Hết quota API:** Vui lòng kiểm tra quota tại [Google AI Studio](https://aistudio.google.com)\n\n*Chi tiết: {error_str}*"
            return f"❌ **Lỗi khi gọi API:** {error_str}\n\nVui lòng kiểm tra API key và kết nối mạng."
    
    def _get_fallback_analysis(self, prompt: str) -> str:
        return """⚠️ **Chưa cấu hình API Key**

Để sử dụng tính năng phân tích AI, vui lòng:

1. **Lấy API key từ Google AI Studio:**
   - Truy cập [aistudio.google.com](https://aistudio.google.com)
   - Tạo API key mới

2. **Thêm vào file `.env`:**
   ```
   GEMINI_API_KEY=AIzaSy...xxxxx
   ```

3. **Khởi động lại dashboard**

---

💡 *Gemini API miễn phí với quota hàng ngày rất lớn!*
"""
    
    def analyze_chart(
        self,
        coin: str,
        chart_type: str,
        chart_data: Dict[str, Any],
        chart_title: str,
        force_refresh: bool = False
    ) -> str:
        coin = coin.lower()
        
        if not force_refresh:
            cached = self._get_cached(coin, chart_type, chart_data)
            if cached:
                return cached + "\n\n---\n*📦 Từ cache - Click để làm mới*"
        
        prompt = self._build_prompt(chart_type, coin, chart_data, chart_title)
        
        analysis = self._call_gemini(prompt)
        
        if "❌" not in analysis and "⚠️ **Chưa cấu hình" not in analysis:
            self._save_cache(coin, chart_type, chart_data, analysis)
        
        return analysis
    
    def clear_cache(self, coin: Optional[str] = None) -> int:
        if not self.cache_dir.exists():
            return 0
        
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if coin is None or cache_file.name.startswith(coin):
                cache_file.unlink()
                count += 1
        
        return count
    
    def get_cache_stats(self) -> Dict:
        if not self.cache_dir.exists():
            return {"total_files": 0, "total_size_kb": 0}
        
        files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "total_files": len(files),
            "total_size_kb": round(total_size / 1024, 2),
            "cache_dir": str(self.cache_dir)
        }

_analyzer_instance: Optional[ChartAnalyzer] = None

def get_chart_analyzer() -> ChartAnalyzer:
    global _analyzer_instance
    
    if _analyzer_instance is None:
        _analyzer_instance = ChartAnalyzer()
    
    return _analyzer_instance
