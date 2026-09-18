# Số liệu cho report pFedES (`_cos`: C = 100 %, AdamW, LR cosine 1e-3 → 1e-5, T = 50)

Sinh tự động bởi `scripts/report_data.py` từ artifact đã kéo về và verify local; không có số nào lấy từ W&B. Mọi metric là **trung bình không trọng số trên N client**, mỗi client đo trên **toàn bộ 10 761 343 dòng test toàn cục** bằng F_k(x). Caveat bắt buộc: `CONTEXT.md` §8 (test toàn cục ≠ test riêng client của bài báo; C = 100 % ≠ Table 1–2 của bài báo; một seed).

## 1. Tình trạng run

| kịch bản | client | round | trạng thái | phiên Kaggle | round-time trung vị | tổng giờ round |
|---|---:|---:|---|---|---:|---:|
| 20c | 20 | **50/50** | HOÀN TẤT | 2 phiên (28 round, 22 round) | 22.0 phút (train 17.4 + eval 4.6) | 18.2 h |
| 50c | 50 | **50/50** | HOÀN TẤT | 3 phiên (21 round, 21 round, 8 round) | 28.6 phút (train 16.9 + eval 11.5) | 23.9 h |
| 100c | 100 | **50/50** | HOÀN TẤT | 4 phiên (13 round, 12 round, 14 round, 11 round) | 44.2 phút (train 22.0 + eval 21.5) | 37.6 h |

## 2. Round cuối — đủ 10 metric (mean trên client; std/min/max của f1_macro)

accuracy = precision_micro = recall_micro = recall_weighted = f1_micro (đồng nhất thức đa lớp đơn nhãn).

| metric | 20c (r50) | 50c (r50) | 100c (r50) |
|---|---:|---:|---:|
| accuracy | 0.5276 | 0.4335 | 0.3692 |
| precision_macro | 0.5921 | 0.4882 | 0.3987 |
| precision_micro | 0.5276 | 0.4335 | 0.3692 |
| precision_weighted | 0.6805 | 0.6172 | 0.5576 |
| recall_macro | 0.5656 | 0.4481 | 0.3709 |
| recall_micro | 0.5276 | 0.4335 | 0.3692 |
| recall_weighted | 0.5276 | 0.4335 | 0.3692 |
| f1_macro | 0.4856 | 0.3652 | 0.2900 |
| f1_micro | 0.5276 | 0.4335 | 0.3692 |
| f1_weighted | 0.5054 | 0.4171 | 0.3556 |
| f1_macro std / min / max | 0.0947 / 0.2817 / 0.6450 | 0.0633 / 0.2298 / 0.5022 | 0.0489 / 0.1424 / 0.3855 |
| client yếu nhất / mạnh nhất (f1_macro) | #12 0.282 / #6 0.645 | #15 0.230 / #11 0.502 | #14 0.142 / #57 0.385 |

## 3. Hình dạng đường cong (round 1 → đỉnh hậu kiểm → cuối)

Đỉnh là quan sát *sau khi chạy*, không phải checkpoint được chọn; số công bố là round cuối.

| | 20c | 50c | 100c |
|---|---:|---:|---:|
| f1_macro round 1 | 0.4851 | 0.3893 | 0.3238 |
| f1_macro đỉnh (round) | 0.5010 (r5) | 0.3893 (r1) | 0.3238 (r1) |
| f1_macro cuối | 0.4856 | 0.3652 | 0.2900 |
| giảm so đỉnh | −3.1 % | −6.2 % | −10.5 % |
| accuracy round 1 → cuối | 0.5615 → 0.5276 | 0.4826 → 0.4335 | 0.4248 → 0.3692 |

## 4. Kiểm soát chất lượng pipeline

| | 20c | 50c | 100c |
|---|---:|---:|---:|
| backend train/eval | compiled | compiled | compiled |
| tổng cache_mismatch | 0 | 0 | 0 |
| evaluated = N mọi round | có | có | có |
| fingerprint | `3b90a02e92ab425d` | `42af8af8cc4d38af` | `f7c3d21a510cf7dc` |

## 5. File

| file | nội dung |
|---|---|
| `rounds_Kc.md`, `history_Kc.csv` | **mọi round**, đủ 10 metric (mean) + std/min/max, lr, train/eval giây — bảng bắt buộc của report |
| `per_client_final_Kc.csv` | 10 metric của từng client ở round cuối |
| `per_class_final_Kc.csv` | mỗi lớp ở round cuối: support, P/R/F1 trung bình trên client và trên confusion gộp |
| `confusion_final_Kc.csv` | confusion gộp (tổng N client) round cuối, 16×16 |
| `summary.json` | mọi số ở trên dạng máy đọc + cấu hình từ `reports/manifest.json` + provenance phiên |
| `convergence.png` | f1_macro / accuracy (mean ± std) theo round, cả ba K; lịch LR |
| `client_spread.png` | phân bố f1_macro theo client ở round cuối |
| `per_class_f1_Kc.png`, `confusion_Kc.png` | F1 từng lớp (theo tỉ lệ test) và confusion gộp chuẩn hoá theo hàng |
| `round_time.png` | phút train / eval mỗi round trên 2×T4 |
