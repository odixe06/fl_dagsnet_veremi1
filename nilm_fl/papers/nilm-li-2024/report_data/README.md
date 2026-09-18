# Số liệu cho REPORT.md — Lightweight-FL NILM (học tương hỗ liên bang) trên VeReMi / DAGSNet

Sinh tự động bởi `scripts/report_data.py` từ artifact đã kéo về, ghép phiên và verify local; không có số nào lấy từ W&B. Metric **cá nhân hoá** là trung bình không trọng số trên N model w_s^k, mỗi model đo trên **toàn bộ 10 761 343 dòng test toàn cục**; metric **proxy** là của một model w̄_r (server) trên cùng tập test. Caveat bắt buộc: `CONTEXT.md` §7.

## 1. Tình trạng run

| kịch bản | client | round | trạng thái | phiên Kaggle | round-time trung vị | tổng giờ round |
|---|---:|---:|---|---|---:|---:|
| 20c | 20 | **50/50** | HOÀN TẤT | 3 phiên (33 round, 16 round, 1 round) | 14.4 phút (train 9.5 + eval 4.9) | 12.2 h |
| 50c | 50 | **50/50** | HOÀN TẤT | 2 phiên (29 round, 21 round) | 21.3 phút (train 9.8 + eval 11.5) | 17.9 h |
| 100c | 100 | **50/50** | HOÀN TẤT | 5 phiên (14 round, 12 round, 6 round, 4 round, 14 round) | 40.4 phút (train 17.2 + eval 23.8) | 33.8 h |

## 2. Round cuối — đủ 10 metric (mean trên client; std/min/max của f1_macro; proxy w̄_r)

accuracy = precision_micro = recall_micro = recall_weighted = f1_micro (đồng nhất thức đa lớp đơn nhãn).

| metric | 20c w_s (r50) | 20c w̄_r | 50c w_s (r50) | 50c w̄_r | 100c w_s (r50) | 100c w̄_r |
|---|---:|---:|---:|---:|---:|---:|
| accuracy | 0.6805 | 0.7902 | 0.6602 | 0.7742 | 0.6289 | 0.7510 |
| precision_macro | 0.7842 | 0.8364 | 0.7334 | 0.7961 | 0.6921 | 0.7592 |
| precision_micro | 0.6805 | 0.7902 | 0.6602 | 0.7742 | 0.6289 | 0.7510 |
| precision_weighted | 0.7966 | 0.8340 | 0.7626 | 0.8084 | 0.7308 | 0.7793 |
| recall_macro | 0.7311 | 0.8286 | 0.6850 | 0.8011 | 0.6454 | 0.7711 |
| recall_micro | 0.6805 | 0.7902 | 0.6602 | 0.7742 | 0.6289 | 0.7510 |
| recall_weighted | 0.6805 | 0.7902 | 0.6602 | 0.7742 | 0.6289 | 0.7510 |
| f1_macro | 0.6963 | 0.8228 | 0.6543 | 0.7923 | 0.6111 | 0.7568 |
| f1_micro | 0.6805 | 0.7902 | 0.6602 | 0.7742 | 0.6289 | 0.7510 |
| f1_weighted | 0.6674 | 0.7990 | 0.6548 | 0.7809 | 0.6257 | 0.7532 |
| f1_macro std / min / max | 0.0709 / 0.5257 / 0.7924 | — | 0.0468 / 0.5073 / 0.7423 | — | 0.0496 / 0.4732 / 0.7159 | — |
| client yếu nhất / mạnh nhất (f1_macro) | #1 0.526 / #6 0.792 | — | #15 0.507 / #11 0.742 | — | #4 0.473 / #16 0.716 | — |

## 3. Hình dạng đường cong (round 1 → đỉnh hậu kiểm → cuối)

Đỉnh là quan sát *sau khi chạy*, không phải checkpoint được chọn; số công bố là round cuối.

| | 20c | 50c | 100c |
|---|---:|---:|---:|
| f1_macro w_s round 1 | 0.5143 | 0.4234 | 0.3654 |
| f1_macro w_s đỉnh (round) | 0.6963 (r50) | 0.6543 (r50) | 0.6111 (r50) |
| f1_macro w_s cuối | 0.6963 | 0.6543 | 0.6111 |
| chênh so đỉnh | −0.0 % | −0.0 % | −0.0 % |
| accuracy w_s round 1 → cuối | 0.5831 → 0.6805 | 0.5175 → 0.6602 | 0.4687 → 0.6289 |
| f1_macro w̄_r round 1 → 2 → cuối | 0.2407 → 0.6968 → 0.8228 | 0.2385 → 0.6762 → 0.7923 | 0.2649 → 0.6278 → 0.7568 |
| accuracy w̄_r round 1 → cuối | 0.4507 → 0.7902 | 0.4890 → 0.7742 | 0.4950 → 0.7510 |

## 4. Thành phần loss (trung bình client, các bước áp dụng) — round 1 / 2 / cuối

| | 20c | 50c | 100c |
|---|---:|---:|---:|
| ce_s | 0.219 / 0.144 / 0.072 | 0.244 / 0.148 / 0.087 | 0.227 / 0.134 / 0.091 |
| ce_r | 0.219 / 0.159 / 0.098 | 0.244 / 0.170 / 0.140 | 0.227 / 0.156 / 0.158 |
| kl_s | 0.033 / 0.032 / 0.029 | 0.042 / 0.044 / 0.050 | 0.049 / 0.048 / 0.062 |
| kl_r | 0.033 / 0.030 / 0.019 | 0.042 / 0.040 / 0.035 | 0.049 / 0.043 / 0.044 |
| gnorm_s | 0.950 / 0.741 / 0.407 | 1.074 / 0.851 / 0.464 | 1.310 / 1.042 / 0.509 |
| gnorm_r | 0.946 / 0.892 / 1.343 | 1.074 / 1.057 / 1.850 | 1.309 / 1.304 / 1.913 |

## 5. Kiểm soát chất lượng pipeline

| | 20c | 50c | 100c |
|---|---:|---:|---:|
| backend train | compiled | compiled | compiled |
| evaluated = N mọi round | có | có | có |
| bước/round, skip AMP tổng (max một round) | 84083, 1705 (49) | 84098, 1449 (44) | 168200, 3268 (123) |
| VRAM train max (GiB) | 7.12 | 7.22 | 7.19 |
| fingerprint | `9e11369e7011a414` | `11390552401b6746` | `dee9721a2e3ce653` |
| torch / CUDA | 2.10.0+cu128 / 12.8 | 2.10.0+cu128 / 12.8 | 2.10.0+cu128 / 12.8 |

## 6. File

| file | nội dung |
|---|---|
| `rounds_Kc.md`, `history_Kc.csv` | **mọi round**, đủ 10 metric (mean) + std/min/max, proxy, skip, lr, train/eval giây — bảng bắt buộc của report |
| `per_client_final_Kc.csv` | 10 metric của từng client ở round cuối |
| `per_class_final_Kc.csv` | mỗi lớp ở round cuối: support, P/R/F1 trung bình trên client, trên confusion gộp, và của w̄_r |
| `confusion_final_Kc.csv`, `confusion_global_Kc.csv` | confusion gộp (tổng N client) và của w̄_r, round cuối, 16×16 |
| `summary.json` | mọi số ở trên dạng máy đọc + cấu hình từ `reports/manifest.json` + provenance phiên |
| `convergence.png` | f1_macro / accuracy (mean ± std, nét liền) và proxy w̄_r (nét đứt) theo round, cả ba K; lịch LR |
| `mutual_loss.png` | ce_s, ce_r, kl_s, kl_r trung bình client theo round |
| `client_spread.png` | phân bố f1_macro theo client ở round cuối |
| `per_class_f1_Kc.png`, `confusion_Kc.png` | F1 từng lớp (theo tỉ lệ test) và confusion gộp chuẩn hoá theo hàng |
| `round_time.png` | phút train / eval mỗi round trên 2×T4 |
