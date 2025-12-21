import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime

class RAGCryptoAssistant:
    
    def __init__(self, api_key: Optional[str] = None, data_dir: str = "data/raw", 
                 model: str = "gpt-4", predictions_dir: str = "results/predictions"):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Chưa có API key! Vui lòng:\n"
                    "1. Truyền trực tiếp: RAGCryptoAssistant(api_key='sk-proj-xxx')\n"
                    "2. Hoặc set biến môi trường: $env:OPENAI_API_KEY='sk-proj-xxx'"
                )
        
        self.api_key = api_key
        self.data_dir = Path(data_dir)
        self.predictions_dir = Path(predictions_dir)
        self.model = model
        self.knowledge_base = {}
        self.predictions_data = {}
        
        try:
            import openai
            self.openai = openai
            self.openai.api_key = self.api_key
            print(f"✓ Khởi tạo thành công với model: {model}")
        except ImportError:
            raise ImportError("Cần cài đặt: pip install openai")
    
    def load_historical_data(self):
        print("Đang load dữ liệu lịch sử...")
        
        train_dir = self.data_dir / "train"
        if not train_dir.exists():
            print(f"Không tìm thấy thư mục: {train_dir}")
            return
        
        for csv_file in train_dir.glob("*_binance_*.csv"):
            coin_name = csv_file.stem.split('_')[0]
            
            try:
                df = pd.read_csv(csv_file, index_col=0)
                df.index = pd.to_datetime(df.index)
                
                self.knowledge_base[coin_name] = {
                    'data': df,
                    'summary': self._create_summary(df, coin_name),
                    'statistics': self._calculate_statistics(df)
                }
                
                print(f"✓ Loaded {coin_name}: {len(df)} records")
                
            except Exception as e:
                print(f"✗ Lỗi khi load {coin_name}: {e}")
        
        print(f"\nĐã load {len(self.knowledge_base)} coins")
        
        self._load_predictions()
    
    def _load_predictions(self):
        print("\nĐang load dữ liệu dự đoán...")
        
        if not self.predictions_dir.exists():
            print(f"Không tìm thấy thư mục: {self.predictions_dir}")
            return
        
        for pred_file in self.predictions_dir.glob("*_future_predictions.json"):
            coin_name = pred_file.stem.replace('_future_predictions', '')
            
            try:
                with open(pred_file, 'r') as f:
                    pred_data = json.load(f)
                
                predictions = pred_data.get('predictions', [])
                if predictions:
                    self.predictions_data[coin_name] = {
                        'predictions': predictions,
                        'forecast_horizon': pred_data.get('forecast_horizon_days', 5),
                        'generated_at': pred_data.get('prediction_generated_at', 'Unknown')
                    }
                    print(f"✓ Loaded predictions for {coin_name}: {len(predictions)} days")
                
            except Exception as e:
                print(f"✗ Lỗi khi load predictions {coin_name}: {e}")
        
        print(f"Đã load predictions cho {len(self.predictions_data)} coins")
    
    def _create_summary(self, df: pd.DataFrame, coin: str) -> str:
        
        summaries = []
        for i in range(0, len(df), 7):
            week_data = df.iloc[i:i+7]
            if len(week_data) < 3:
                continue
            
            start_date = week_data.index[0].strftime('%Y-%m-%d')
            end_date = week_data.index[-1].strftime('%Y-%m-%d')
            open_price = week_data['open'].iloc[0]
            close_price = week_data['close'].iloc[-1]
            high = week_data['high'].max()
            low = week_data['low'].min()
            volume = week_data['volume'].mean()
            change = ((close_price - open_price) / open_price) * 100
            
            summary = (
                f"Tuần {start_date} đến {end_date}: "
                f"Giá ${open_price:.2f} → ${close_price:.2f} ({change:+.2f}%), "
                f"Cao nhất ${high:.2f}, Thấp nhất ${low:.2f}, "
                f"Volume TB {volume:.0f}"
            )
            summaries.append(summary)
        
        return "\n".join(summaries[-20:])
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        
        return {
            'current_price': float(df['close'].iloc[-1]),
            'max_price_30d': float(df['close'][-30:].max()),
            'min_price_30d': float(df['close'][-30:].min()),
            'avg_price_30d': float(df['close'][-30:].mean()),
            'avg_volume_30d': float(df['volume'][-30:].mean()),
            'volatility_30d': float(df['close'][-30:].std()),
            'total_days': len(df),
            'last_update': df.index[-1].strftime('%Y-%m-%d'),
            'first_date': df.index[0].strftime('%Y-%m-%d')
        }
    
    def get_price_by_date(self, coin: str, date_str: str) -> Dict:
        if coin not in self.knowledge_base:
            return {"error": f"Không có dữ liệu cho {coin}"}
        
        df = self.knowledge_base[coin]['data']
        
        try:
            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d']:
                try:
                    target_date = pd.to_datetime(date_str, format=fmt)
                    break
                except:
                    continue
            else:
                return {"error": f"Không thể parse ngày '{date_str}'. Dùng format YYYY-MM-DD hoặc DD/MM/YYYY"}
            
            target_date_str = target_date.strftime('%Y-%m-%d')
            
            if target_date in df.index:
                row = df.loc[target_date]
                return {
                    "date": target_date_str,
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume'])
                }
            else:
                idx = df.index.searchsorted(target_date)
                if idx >= len(df):
                    idx = len(df) - 1
                elif idx > 0:
                    if abs(df.index[idx] - target_date) > abs(df.index[idx-1] - target_date):
                        idx = idx - 1
                
                nearest_date = df.index[idx]
                row = df.iloc[idx]
                
                return {
                    "requested_date": target_date_str,
                    "nearest_date": nearest_date.strftime('%Y-%m-%d'),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume']),
                    "note": f"Không có dữ liệu chính xác cho {target_date_str}. Hiển thị dữ liệu gần nhất."
                }
                
        except Exception as e:
            return {"error": f"Lỗi: {str(e)}"}
    
    def _retrieve_context(self, coin: str, query: str = "") -> str:
        if coin not in self.knowledge_base:
            return f"Không có dữ liệu cho {coin}"
        
        kb = self.knowledge_base[coin]
        stats = kb['statistics']
        df = kb['data']
        
        first_date = df.index[0].strftime('%Y-%m-%d')
        last_date = df.index[-1].strftime('%Y-%m-%d')
        total_days = len(df)
        
        context = f"""
**Dữ liệu lịch sử {coin.upper()}:**

Phạm vi dữ liệu: {first_date} đến {last_date} ({total_days} ngày)

Thống kê 30 ngày gần nhất:
- Giá hiện tại: ${stats['current_price']:.2f}
- Giá cao nhất: ${stats['max_price_30d']:.2f}
- Giá thấp nhất: ${stats['min_price_30d']:.2f}
- Giá trung bình: ${stats['avg_price_30d']:.2f}
- Volume trung bình: {stats['avg_volume_30d']:.0f}
- Độ biến động (Std): ${stats['volatility_30d']:.2f}

Lịch sử giao dịch (20 tuần gần nhất):
{kb['summary']}
"""
        
        if coin in self.predictions_data:
            pred = self.predictions_data[coin]
            predictions = pred['predictions']
            
            pred_prices = [p['expected_price'] for p in predictions]
            pred_dates = [p.get('date', f"Day {i+1}") for i, p in enumerate(predictions)]
            
            current_price = stats['current_price']
            future_price = pred_prices[-1]
            change = future_price - current_price
            change_pct = (change / current_price) * 100
            trend = "TĂNG" if change > 0 else "GIẢM"
            
            context += f"""

**DỰ ĐOÁN TỪ MÔ HÌNH AI (Deep Learning):**
Generated at: {pred['generated_at']}
Forecast horizon: {pred['forecast_horizon']} ngày

Dự đoán giá {pred['forecast_horizon']} ngày tới:
"""
            for i, (date, price) in enumerate(zip(pred_dates, pred_prices)):
                context += f"\n- {date}: ${price:.2f}"
            
            context += f"""

Phân tích dự đoán:
- Giá hiện tại: ${current_price:.2f}
- Giá dự đoán cuối kỳ: ${future_price:.2f}
- Biến động dự kiến: {change:+.2f} USD ({change_pct:+.2f}%)
- Xu hướng: {trend}
"""
        
        return context
    
    def get_investment_advice(self, coin: str, current_price: float, 
                             predictions: List[float]) -> str:
        context = self._retrieve_context(coin)
        
        trend = "tăng" if predictions[-1] > current_price else "giảm"
        change_percent = ((predictions[-1] - current_price) / current_price) * 100
        
        prompt = f"""
Bạn là chuyên gia tư vấn đầu tư cryptocurrency với 10 năm kinh nghiệm.

{context}

**Dự đoán AI:**
- Giá hiện tại: ${current_price:.2f}
- Dự đoán 5 ngày tới: {[f'${p:.2f}' for p in predictions]}
- Xu hướng dự đoán: {trend} {abs(change_percent):.2f}%

Hãy phân tích và đưa ra lời khuyên đầu tư chi tiết theo cấu trúc sau:

(Phân tích xu hướng ngắn hạn dựa trên dự đoán và lịch sử)

(Đánh giá rủi ro: Thấp/Trung bình/Cao và lý do)

(MUA/BÁN/GIỮ và giải thích chi tiết)

- Điểm vào lệnh: (mức giá nên mua nếu khuyến nghị mua)
- Mức cắt lỗ (Stop Loss): (mức giá nên bán để giảm thiểu thua lỗ)
- Mức chốt lời (Take Profit): (mức giá nên bán để chốt lời)

(Các lưu ý quan trọng cho nhà đầu tư)

Trả lời bằng tiếng Việt, ngắn gọn, súc tích.
"""
        
        try:
            response = self.openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "Bạn là chuyên gia tư vấn đầu tư cryptocurrency chuyên nghiệp."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Lỗi khi gọi API: {str(e)}\n\nVui lòng kiểm tra API key và kết nối mạng."
    
    def chat(self, coin: str, user_message: str, 
             conversation_history: Optional[List[Dict]] = None) -> str:
        import re
        date_patterns = [
            r'\d{2}/\d{2}/\d{4}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{4}/\d{2}/\d{2}',
        ]
        
        found_date = None
        for pattern in date_patterns:
            match = re.search(pattern, user_message)
            if match:
                found_date = match.group()
                break
        
        price_info = ""
        if found_date and coin in self.knowledge_base:
            price_data = self.get_price_by_date(coin, found_date)
            if 'error' not in price_data:
                if 'note' in price_data:
                    price_info = f"""
**Dữ liệu được tìm thấy:**
Ngày yêu cầu: {price_data['requested_date']}
Ngày gần nhất có dữ liệu: {price_data['nearest_date']}
- Open: ${price_data['open']:.2f}
- High: ${price_data['high']:.2f}
- Low: ${price_data['low']:.2f}
- Close: ${price_data['close']:.2f}
- Volume: {price_data['volume']:.0f}

{price_data['note']}
"""
                else:
                    price_info = f"""
**Giá {coin.upper()} ngày {price_data['date']}:**
- Open: ${price_data['open']:.2f}
- High: ${price_data['high']:.2f}
- Low: ${price_data['low']:.2f}
- Close: ${price_data['close']:.2f}
- Volume: {price_data['volume']:.0f}
"""
        
        context = self._retrieve_context(coin)
        
        if coin in self.knowledge_base:
            df = self.knowledge_base[coin]['data']
            latest_date = df.index[-1].strftime('%Y-%m-%d')
            
            recent_data = df.tail(10)
            recent_prices = "\n".join([
                f"- {date.strftime('%Y-%m-%d')}: Open=${row['open']:.2f}, Close=${row['close']:.2f}, High=${row['high']:.2f}, Low=${row['low']:.2f}"
                for date, row in recent_data.iterrows()
            ])
            
            context += f"""

**Dữ liệu 10 ngày gần nhất (chi tiết):**
{recent_prices}

**Ngày cập nhật mới nhất:** {latest_date}

{price_info}
"""
        
        system_message = f"""
Bạn là trợ lý AI chuyên về đầu tư cryptocurrency.

Bạn đang tư vấn về {coin.upper()}.

{context}

{price_info}

Nhiệm vụ:
- Trả lời câu hỏi dựa trên dữ liệu lịch sử VÀ dự đoán từ mô hình Deep Learning
- Sử dụng tiếng Việt
- Ngắn gọn, dễ hiểu, chuyên nghiệp
- Đưa ra con số cụ thể khi có thể
- Nếu người dùng hỏi về giá ngày cụ thể và có dữ liệu, hãy trả lời chính xác
- Nếu người dùng hỏi về tương lai (ngày mai, tuần sau), hãy dựa vào DỰ ĐOÁN từ mô hình
- Khi có dự đoán, phân tích xu hướng và đưa ra khuyến nghị đầu tư (MUA/BÁN/GIỮ)
- Luôn nêu rõ mức độ rủi ro và nhắc nhở đây chỉ là dự đoán, không phải lời khuyên tài chính chắc chắn
"""
        
        messages = [{"role": "system", "content": system_message}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0.8,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Lỗi: {str(e)}"
    
    def get_coin_analysis(self, coin: str) -> Dict:
        if coin not in self.knowledge_base:
            return {"error": f"Không có dữ liệu cho {coin}"}
        
        stats = self.knowledge_base[coin]['statistics']
        df = self.knowledge_base[coin]['data']
        
        last_7days = df['close'][-7:]
        last_30days = df['close'][-30:]
        
        trend_7d = "tăng" if last_7days.iloc[-1] > last_7days.iloc[0] else "giảm"
        trend_30d = "tăng" if last_30days.iloc[-1] > last_30days.iloc[0] else "giảm"
        
        change_7d = ((last_7days.iloc[-1] - last_7days.iloc[0]) / last_7days.iloc[0]) * 100
        change_30d = ((last_30days.iloc[-1] - last_30days.iloc[0]) / last_30days.iloc[0]) * 100
        
        return {
            "coin": coin,
            "statistics": stats,
            "trends": {
                "7_days": {"direction": trend_7d, "change_percent": change_7d},
                "30_days": {"direction": trend_30d, "change_percent": change_30d}
            },
            "support_level": float(last_30days.min()),
            "resistance_level": float(last_30days.max())
        }
    
    def compare_coins(self, coins: List[str]) -> str:
        comparisons = []
        
        for coin in coins:
            if coin in self.knowledge_base:
                analysis = self.get_coin_analysis(coin)
                comparisons.append({
                    'coin': coin,
                    'price': analysis['statistics']['current_price'],
                    'change_30d': analysis['trends']['30_days']['change_percent'],
                    'volatility': analysis['statistics']['volatility_30d']
                })
        
        if not comparisons:
            return "Không có dữ liệu để so sánh"
        
        comparisons.sort(key=lambda x: x['change_30d'], reverse=True)
        
        prompt = f"""
So sánh các cryptocurrency sau dựa trên performance 30 ngày:

{json.dumps(comparisons, indent=2, ensure_ascii=False)}

Hãy:
1. Xếp hạng các coin theo tiềm năng đầu tư
2. Phân tích ưu/nhược điểm của từng coin
3. Đưa ra khuyến nghị đa dạng hóa danh mục

Trả lời bằng tiếng Việt, ngắn gọn.
"""
        
        try:
            response = self.openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia phân tích thị trường crypto."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Lỗi: {str(e)}"
