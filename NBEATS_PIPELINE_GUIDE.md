📘 N-BEATS Pipeline Guide

(Crypto Forecasting – 9 coins – 5-day horizon)

1. N-BEATS là gì? (tóm tắt nhanh)

N-BEATS (Neural Basis Expansion Analysis for Time Series) là mô hình deep learning chuyên cho dự báo chuỗi thời gian:

Không dùng RNN/LSTM

Không dùng Attention/Transformer

Dùng MLP + residual blocks

Học trực tiếp multi-horizon forecast

📌 N-BEATS được thiết kế để:

Dự báo ổn định

Ít overfit

Dễ làm baseline mạnh cho time-series

2. Khi nào nên dùng N-BEATS?

N-BEATS rất phù hợp khi:

Dữ liệu daily, không quá dài (~1000–5000 điểm)

Muốn forecast nhiều bước (5–30 ngày)

Muốn baseline loss thấp, ít drama

Có nhiều chuỗi (multi-coin)

👉 Với project của bạn (9 coin × ~1550 ngày): N-BEATS là lựa chọn rất đúng

3. Tổng quan pipeline N-BEATS (high-level)
Raw CSV (9 coins)
   ↓
Feature engineering (log_return)
   ↓
Long-format dataset (unique_id, ds, y)
   ↓
Global N-BEATS model (train 1 model cho 9 coin)
   ↓
Forecast log-return (5 ngày)
   ↓
Convert return → price
   ↓
Dashboard / Backtest / Comparison

4. Quyết định quan trọng trước khi triển khai
4.1 Target (BẮT BUỘC)

❌ Không dùng price

✅ Dùng log-return

𝑟
𝑡
=
log
⁡
(
𝑃
𝑡
/
𝑃
𝑡
−
1
)
r
t
	​

=log(P
t
	​

/P
t−1
	​

)

Lý do:

Stationary hơn

Scale đồng nhất giữa coin

Model học “thay đổi” thay vì “mức giá”

4.2 Global model hay per-coin?
Cách	Đánh giá
9 model riêng	❌ data ít, dễ overfit
1 model chung	✅ khuyến nghị

👉 N-BEATS rất mạnh ở global forecasting

4.3 Horizon & Lookback

horizon (H) = 5

input_size (lookback) nên thử: 60 / 90 / 120

5. Chuẩn bị dữ liệu cho N-BEATS
5.1 Format dữ liệu chuẩn (long format)

N-BEATS (NeuralForecast) yêu cầu DataFrame dạng:

column	ý nghĩa
unique_id	mã chuỗi (coin)
ds	timestamp
y	target (log_return)
5.2 Ví dụ từ CSV coin
import pandas as pd
import numpy as np

df = pd.read_csv("bitcoin.csv")
df["ds"] = pd.to_datetime(df["timestamp"])
df["log_close"] = np.log(df["close"])
df["y"] = df["log_close"].diff()
df = df.dropna()

df_long = pd.DataFrame({
    "unique_id": "BTC",
    "ds": df["ds"],
    "y": df["y"]
})


👉 Lặp cho 9 coin → concat lại thành 1 DataFrame duy nhất

6. Huấn luyện N-BEATS (NeuralForecast)
6.1 Khởi tạo model
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS

model = NBEATS(
    h=5,                 # forecast 5 ngày
    input_size=90,       # lookback
    learning_rate=1e-3,
    max_steps=2000       # tương đương epochs
)

nf = NeuralForecast(
    models=[model],
    freq="D"
)

6.2 Train model
nf.fit(df=data_long)


📌 Lưu ý:

Không cần scaler thủ công (N-BEATS xử lý ổn)

Không cần epoch/batch loop thủ công

Train rất ổn định so với LSTM

7. Dự báo 5 ngày tới
pred = nf.predict()


Output dạng:

unique_id	ds	NBEATS
BTC	t+1	r₁
BTC	t+2	r₂
…	…	…
8. Convert log-return → price (để hiển thị)

Giả sử:

Giá hiện tại: P₀

Dự đoán log-return: r₁, r₂, … r₅

import numpy as np

prices = []
cur = np.log(P0)
for r in returns:
    cur += r
    prices.append(np.exp(cur))


👉 Không mất thông tin giá, chỉ đổi cách học

9. Đánh giá model đúng cách
9.1 Metric nên dùng

MAE / RMSE trên return

Directional Accuracy:

sign(y_true) == sign(y_pred)

9.2 Validation

❌ Không random split

✅ Walk-forward / rolling window

10. So sánh N-BEATS vs LSTM trong project
Tiêu chí	N-BEATS	LSTM
Stability	⭐⭐⭐⭐⭐	⭐⭐⭐
Overfit risk	Thấp	Cao nếu data ít
Multi-horizon	Native	Phải xử lý
Tuning	Dễ	Khó
Global model	Rất tốt	Phải custom
Interpretability	Trung bình	Thấp

👉 N-BEATS = baseline rất mạnh
👉 LSTM = model nâng cao khi pipeline đã sạch

11. Best practices cho project của bạn

Dùng N-BEATS làm baseline chính

So sánh với:

ARIMA (benchmark)

LSTM (return + 5-day)

Có thể ensemble:

final_forecast = mean(NBEATS, LSTM)

12. Checklist triển khai N-BEATS

 Target = log_return

 Data long-format (unique_id, ds, y)

 Global model cho 9 coin

 Horizon = 5

 Lookback = 60/90/120

 Walk-forward validation

 Convert return → price cho dashboard

13. Kết luận (chốt lại)

❝ N-BEATS không phức tạp, nhưng rất mạnh
vì nó tập trung đúng vào bản chất của forecasting. ❞

Trong project crypto của bạn:

N-BEATS = baseline sạch & mạnh

LSTM = model học pattern nâng cao