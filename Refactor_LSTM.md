📌 Crypto LSTM Project – Review & Refactor Notes

(Accuracy ↑ | Loss ↓ | Generalization ↑ cho 9 coin, forecast 5 ngày)

1. Mục tiêu bài toán

Dự đoán 5 ngày tới cho 9 loại tiền điện tử

Mô hình dùng LSTM

Yêu cầu:

Loss ổn định, không ảo

Directional Accuracy tốt (đúng hướng tăng/giảm)

Generalize tốt giữa các coin (BTC, ETH, ALTCOINS)

2. Các vấn đề cốt lõi hiện tại trong project
2.1. ❌ Mismatch giữa output model và shape của y (CRITICAL)

Hiện trạng

model.py:

output = Dense(1)(x)


trainer.py:

y shape = (N, 2)
y[:,0] = current price
y[:,1] = previous price


Vấn đề

Output (N,1) nhưng target (N,2)

Phải “hack” loss (di_mse_loss) để xử lý → dễ:

broadcast sai

gradient lệch

train ra kết quả “ảo”

Kết luận

❌ Thiết kế target (current, prev) là không đúng bản chất supervised learning.

2.2. ❌ Target đang là price → khó generalize cho nhiều coin

Hiện trạng

target = df['close'].shift(-forecast_horizon)


Vấn đề

Giá tuyệt đối:

Scale khác nhau (BTC vs ADA)

Non-stationary (trend mạnh)

Model dễ học “trend” thay vì “pattern”

Loss thấp trên train nhưng out-of-sample kém

Best practice (finance/crypto ML)

🔥 Dự đoán return, không dự đoán price

2.3. ❌ Forecast 5 ngày nhưng model chỉ output 1 bước

Hiện trạng

Dense(1)


Vấn đề

Không đúng mục tiêu “predict 5 days”

Nếu rollout 5 lần → lỗi tích lũy rất lớn

Khuyến nghị

✅ Multi-horizon direct forecasting

Output: (5,)

y shape: (N,5) = [t+1, t+2, t+3, t+4, t+5]

2.4. ⚠️ Bidirectional LSTM dễ tạo “accuracy ảo”

Hiện trạng

Bidirectional(LSTM(...))


Rủi ro

BiLSTM học quan hệ 2 chiều trong window

Nếu có leakage (scaling/split) → accuracy nhìn rất đẹp

Dễ overfit với data ít (~1550 ngày/coin)

Khuyến nghị

Baseline trước bằng LSTM thường

So sánh BiLSTM như một ablation study

2.5. ❌ Custom loss di_mse_loss đang phạt sai hướng bằng hằng số

Hiện trạng

wrong_loss = wrong_mask * DIRECTION_WEIGHT_FACTOR


Vấn đề nghiêm trọng

Sai hướng nhưng:

sai ít hay sai nhiều → phạt như nhau

Gradient không kéo model về giá trị đúng

Model dễ học “đổi dấu” hơn là học magnitude

Hệ quả

Loss giảm nhưng prediction không thực sự tốt

2.6. ❌ Thiết kế loss/metric phụ thuộc vào prev_price trong y

Hiện trạng

y phải chứa (current_price, prev_price)

Loss/metric phải “hack” y

Vấn đề

Pipeline phức tạp

Không mở rộng được lên horizon = 5

Sai bản chất học có giám sát

2.7. ⚠️ Feature Engineering dùng bfill → leakage nhẹ
df_features.bfill()
df_features.ffill()


Vấn đề

bfill dùng future value lấp quá khứ

Với rolling indicators → dễ tạo leakage

Khuyến nghị

❌ Không dùng bfill

✅ dropna() hoặc chỉ ffill()

3. Kiến trúc được khuyến nghị (Production-grade)
3.1. Target & Output
Thành phần	Khuyến nghị
Target	log_return
Horizon	5 ngày
y shape	(N,5)
Output	Dense(5)
3.2. Loss & Metric

Loss: Direction-aware Huber

Metric:

MAE / RMSE (return)

Directional Accuracy (return)

👉 Direction không nên nhồi cứng vào loss bằng hằng số

3.3. Model Architecture (baseline mạnh & ổn định)
Input (T, F)
 → LayerNorm
 → LSTM(128)
 → Dropout
 → LSTM(64)
 → Dense(64, relu)
 → Dense(5)


(Sau đó mới thử BiLSTM / CNN-LSTM)

3.4. Multi-coin Training (rất quan trọng)

Vấn đề hiện tại

Mỗi coin ~1550 ngày → train riêng dễ overfit

Giải pháp

✅ 1 model chung cho 9 coin

Gộp data của 9 coin

Thêm feature coin_id (one-hot hoặc embedding)

Model học:

Pattern chung thị trường

Vẫn phân biệt từng coin

4. Các cải tiến bắt buộc để tránh “kết quả ảo”
4.1. Anti-leakage checklist

 Split theo thời gian (chronological)

 Fit scaler chỉ trên train

 Không bfill

 Không shuffle time-series

4.2. Validation đúng chuẩn

Walk-forward validation (3–5 folds)

Không chỉ 1 lần train/val/test

5. Tổng hợp Action Items (Checklist)
❗ Critical – phải sửa

 Bỏ y dạng (current, prev)

 Chuyển target sang log_return

 Output multi-horizon (5,)

 Sửa di_mse_loss hoặc thay bằng directional Huber

 Bỏ bfill trong feature engineering

🚀 Nâng cao chất lượng

 LSTM thường làm baseline (trước BiLSTM)

 1 model chung cho 9 coin

 Coin ID embedding

 Walk-forward validation

 Plot return → price forecast cho dashboard

6. Kết luận cuối cùng

❝ LSTM không giỏi học “giá là bao nhiêu”,
mà giỏi học “giá thay đổi như thế nào”. ❞

Để đạt accuracy tốt – loss thấp – không ảo cho crypto:

✅ Return + Multi-horizon + Multi-coin + Proper loss
❌ Price + Hack loss + Single-step + Single-coin