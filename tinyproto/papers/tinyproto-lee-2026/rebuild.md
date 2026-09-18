# TinyProto-FP trên VeReMi NextGen — đặc tả thực thi

**Bài báo.** Gyuejeong Lee, Daeyoung Choi. *Communication-Efficient Heterogeneous Federated
Learning with Sparse Prototypes in Resource-Constrained Environments* (TinyProto), AAAI-26
submission 02846. Nguồn dùng ở đây: `02846-AAAI26.LeeG-ML.md` và `.pdf` (**9 trang, kết thúc ở
phần References — KHÔNG có phụ lục**).

**Biến thể được dựng: TinyProto-FP** (CPS + APS ghép vào FedProto). Đây là biến thể duy nhất
bài báo đặc tả đủ trong phần chính (Algorithm 1). TinyProto-FT bị bỏ theo quyết định của người
dùng ngày 2026-09-07.

⚠ **Đây KHÔNG phải tái lập bài báo.** Bài báo chạy 5 bộ dữ liệu ảnh + AG News, 20 client,
Dir(0,1), 4 kiến trúc CNN nhẹ khác nhau, 300 round, batch 32, d = 500. Bản này chạy VeReMi
NextGen 16 lớp, 20/50/100 client, Dir(0,5), **một** kiến trúc DAGSNet ở mọi client, 50 round,
batch 512/256, d = 256. Cái được kế thừa là **phương pháp**, không phải con số. Không đặt số của
bản này cạnh Table 1 của bài báo trong cùng một bảng.

---

## 1. Phương pháp — cái gì lấy nguyên từ bài báo

| | công thức | lấy nguyên |
|---|---|---|
| Prototype cục bộ | `c_L[i,j] = (1/n_ij) Σ_{(x,y)∈D_ij} f_i(θ_i; x)` | Eq. (3) |
| Mask nhị phân | `m_j ∈ {0,1}^d`, `s = Σ m_j`, cố định suốt run | §4.1 |
| Sparse có cấu trúc | `S(c; m) = c ⊙ m` | Eq. (7), Def. 1 |
| Nén | `C(c; m) = (c_i : m_i = 1) ∈ R^s` | Eq. (8), Def. 2 |
| APS phía client | gửi `n_ij · ĉ_L[i,j]`, **không** gửi `n_ij` riêng | Eq. (10) |
| APS phía server | `ĉ_G[j] = (1/|N_j|) Σ_{i∈N_j} n_ij ĉ_L[i,j]` | Eq. (10) |
| Regularizer | `R_i = Σ_j ρ(ĉ_L[i,j], μ ĉ_G[j])` | Eq. (11) |
| Mục tiêu cục bộ | `CE + λ R_i` | Eq. (5) |
| Dự đoán | `ŷ = argmin_j ‖f_i(θ_i;x) − c_L[i,j]‖₂` | Eq. (12) |
| Chi phí truyền tin | `Σ_i (K_i + K) × s` | Table 2 |
| Round 1 | huấn luyện **không** regularization | §3.1 Step 3 |
| λ | **1** | §5.1 ("we set λ = 1 following FedProto") |
| Tham gia | **100% client mỗi round** | §5.1 |
| Local epoch | **1** | §5.1 |
| Scheduler | **không có** | §5.1 ("No hyperparameter schedulers were applied") |
| Chỉ số báo cáo | trung bình trên toàn bộ client, mỗi round | §5.1 Evaluation Protocol |

---

## 2. Mọi chỗ bài báo để trống, và bản này chọn gì

Mỗi dòng dưới đây là **lựa chọn của bản dựng**, phải công bố như vậy, **không được gán cho tác giả**.

| hạng mục | bài báo nói gì | bản này làm gì | vì sao |
|---|---|---|---|
| **Kiến trúc client** | 4 CNN nhẹ khác nhau (ResNet-8, EfficientNet, ShuffleNet v2, MobileNet v2) chia cho các client | **DAGSNet ở MỌI client** | Người dùng yêu cầu. Hệ quả: bản này **không** kiểm chứng luận điểm "hỗ trợ kiến trúc dị thể" của bài báo. Client vẫn dị thể về **trọng số** (mỗi client giữ mô hình riêng, không bao giờ bị trung bình hoá) — đó là điều làm cho việc đo 10 metric trên từng client có nghĩa. |
| **Trích xuất đặc trưng** | có bước trích xuất đặc trưng riêng | **bỏ** — dùng thẳng 66 đặc trưng thô đã z-score | Người dùng yêu cầu. |
| **Feature layer d** | 500 | **256** — đầu ra `head[3]` (ReLU sau `Linear(544,256)`) của DAGSNet | Bị kiến trúc quy định, không phải tham số tự do. |
| **CPS dimension s** | 50 (nén 90% ở d = 500) | **50** (nén **80,5%** ở d = 256) | Giữ đúng **giá trị** bài báo công bố. Ở d = 256 hai phát biểu "s = 50" và "nén 90%" của bài báo mâu thuẫn nhau; bản này giữ giá trị. Muốn giữ tỉ lệ nén thì `s = 26`. |
| **Cách sinh mask** | "random allocation … optimizing for maximum inter-class Hamming distance separation" — không có thuật toán | **Cấp phát tham lam cân bằng** (mỗi lớp lấy s chiều đang ít dùng nhất) rồi **local search** đổi chỗ để hạ overlap lớn nhất | Đo được: overlap max **10** = ngưỡng dừng heuristic `⌈s²/d⌉` (làm tròn kỳ vọng overlap ngẫu nhiên, **không phải cận dưới lý thuyết**, chưa chứng minh tối ưu), mean 7,28 so với kỳ vọng ngẫu nhiên 9,77, Hamming min **80**. Ghi vào `config.json["mask_stats"]`. |
| **Optimizer** | không nêu tên; lr **0,01**, batch 32 | **AdamW**, lr **1e-3**, weight_decay **1e-4**, betas (0,9, 0,999), eps 1e-8 | Quyết định của người dùng 2026-09-07. Batch ở đây lớn hơn 8–16 lần batch của bài báo; đây đúng là cấu hình đã cho DAGSNet f1_macro 0,853 trên chính bộ dữ liệu này (`knowledge/ARCHITECTURE.md` §4.1). Code gốc FedProto/FedTGP dùng SGD — **đây là deviation**. |
| **Trạng thái optimizer giữa các round** | không nêu | **giữ liên tục theo từng client** | Mô hình của client là liên tục qua các round (FL cá thể hoá không bao giờ ghi đè bằng trung bình toàn cục), nên moment estimate vẫn còn giá trị. Reset mỗi round sẽ khởi động lại bias correction 50 lần. |
| **ρ(·,·)** | "Euclidean distance" | **Bình phương sai số trung bình trên s chiều được mask**: `‖(h − μ ĉ_G) ⊙ m‖² / s` | Khi s = d, công thức này **đúng bằng** `nn.MSELoss()` mà code FedProto dùng, nên nó suy biến về FedProto khi tắt CPS; và chia cho s giữ cho λ = 1 có cùng ý nghĩa ở mọi s. |
| **R_i theo lớp hay theo mẫu** | Eq. (11) tổng theo lớp | **theo mẫu**: mỗi hàng bị kéo về prototype toàn cục của lớp của chính nó | Là hành vi của code FedProto đã phát hành. Theo lớp chỉ cho một gradient trên mỗi trung bình lớp của batch. |
| **Thời điểm tính prototype** | Eq. (3) viết cho **một** θ_i | **một lượt inference sau khi hết epoch**, ở chế độ `eval()` | Code FedProto trung bình đặc trưng thu **trong lúc** epoch. Ổn ở ~59 bước/epoch (CIFAR-10); ở đây một local epoch lên tới **11.506 bước**, trọng số đã đi rất xa nên trung bình đó mô tả một mô hình không còn tồn tại. `eval()` để BatchNorm dùng running stats và tắt dropout — đúng bộ trích xuất mà Eq. (12) dùng lúc test. Giá phải trả: +1 lượt inference qua tập train mỗi round. |
| **μ** | 5 giá trị cho 5 bộ ảnh; "systematic parameter selection… detailed in the appendix" — **PDF không có phụ lục**, và **không xuất hiện chữ "grid search"** | **grid search**, chọn theo f1_macro trung bình client trên **validation tách từ train** | Người dùng chỉ định dùng grid search (2026-09-07). Ghi rõ: đây là cách hiện thực hoá của bản này, **không phải phương pháp tác giả công bố**. |
| **Chọn μ trên tập nào** | bài báo không nói | **2% mỗi client, phân tầng theo lớp, seed cố định** — sweep train trên 98% còn lại | Quyết định của người dùng 2026-09-07. Quy tắc project cấm điều chỉnh siêu tham số trên tập test. Ba run chính thức train trên **100%** train. |
| **Gradient clipping** | không nêu | **global norm 1,0**, sau `unscale_` | Cần cho ổn định fp16 AMP; cùng giá trị với bản centralized. |
| **Precision** | không nêu | **fp16 AMP + GradScaler**, loss tính fp32 ngoài autocast | T4 không có bf16/TF32. |
| **Seed** | 3 seed, báo cáo trung bình | **một seed = 42, một lần chạy** | Ngân sách GPU. **Phải công bố: chưa có replication.** |
| **Số round** | 300 | **50** | Người dùng yêu cầu; cũng là ngân sách tính toán. |
| **Khởi tạo client** | không nêu | **mọi client bắt đầu từ CÙNG một khởi tạo** (seed 42) | Khởi tạo khác nhau sẽ trộn lẫn hai nguồn dị thể. |

---

## 3. Cấu hình cố định cho ba kịch bản

| | 20 client | 50 client | 100 client |
|---|---:|---:|---:|
| batch/client | 512 | 512 | 256 |
| bước optimizer/round | 84.083 | 84.098 | 168.200 |
| round × local epoch | 50 × 1 | 50 × 1 | 50 × 1 |
| Σᵢ(K_i + K) | 640 | 1.594 | 3.184 |
| chi phí truyền tin/round (s = 50) | 32.000 tham số | 79.700 | 159.200 |
| so với FedProto (d = 256) | 5,12× thấp hơn | 5,12× | 5,12× |

Chung cho cả ba: `rounds=50`, `local_epochs=1`, `lam=1.0`, `cps_s=50`, `mask_seed=42`, `seed=42`,
`lr=1e-3`, `weight_decay=1e-4`, `clip=1.0`, `amp=True`, `eval_batch = 8192`,
tham gia 100%, đánh giá trên **đủ 10.761.343 dòng test mỗi round cho từng client**.

---

## 4. Đánh giá

**Hai quy tắc dự đoán, cùng một lượt forward, cùng báo cáo:**

* `proto` — Eq. (12), argmin khoảng cách L2 tới prototype cục bộ **dày** của chính client.
  Đây là quy tắc của bài báo cho PBFL và là **kết quả chính**.
* `clf` — argmax logit của head DAGSNet. Miễn phí (cùng một forward), và là quy tắc duy nhất có
  thể dự đoán một lớp mà client chưa từng thấy.

Lớp mà client không có (`n_ij = 0`) không có prototype nên **không bao giờ** được quy tắc `proto`
dự đoán; recall của lớp đó ở client đó bằng 0. Đây là bản chất của PBFL cá thể hoá, không phải lỗi.
Trên bộ này ảnh hưởng nhỏ: 20/20 client đủ 16 lớp, 44/50, 86/100.

**Ba mức tổng hợp, đều lưu:**

1. `per_client` — 10 metric cho **từng** client (đây là thứ người dùng yêu cầu).
2. `mean_over_clients` — trung bình cộng trên toàn bộ client. **Đây là con số của bài báo**
   ("the average test accuracy across all clients per round"). Kèm `std`/`min`/`max`.
3. `pooled` — cộng ma trận nhầm lẫn của mọi client rồi mới tính metric. **Không** phải số của
   bài báo; trả lời một câu hỏi khác.

10 metric theo đúng `METRIC_KEYS` của project: accuracy, precision_{macro,micro,weighted},
recall_{macro,micro,weighted}, f1_{macro,micro,weighted}. Với phân loại đơn nhãn đa lớp,
`precision_micro = recall_micro = f1_micro = recall_weighted = accuracy` — **đẳng thức**, không
phải lỗi. Với mất cân bằng 41:1 thì **`f1_macro` mới là con số đáng đọc**.

**Đánh giá chạy trên bản mô hình đã fold BatchNorm.** Ở `eval()`, BatchNorm là một phép affine
hệ số hằng nên gộp nó vào convolution đứng trước là **chính xác về mặt toán học**, và đo được
**nhanh hơn 1,6–1,8×**. Ma trận nhầm lẫn của quy tắc `proto` **trùng khít từng số** giữa hai
đường; quy tắc `clf` lệch 0,058% hàng trên mô hình khởi tạo ngẫu nhiên (nơi logit gần như hoà).

---

## 5. Hợp đồng artifact

```
runs/<run_name>/
  config.json            cấu hình hiệu lực + fingerprint + mask_stats
  masks.npy              (16, 256) uint8 — mask CPS
  mu.json                μ đã giải + nguồn gốc (mean n_ij, |N_j|)
  data_audit.json        thống kê giải mã + n_ij từng client
  weights/round_NNN.pt   (C, 395024) fp32 params + (C, 3264) buffers + (C, 31) num_batches_tracked
  protos/round_NNN.pt    prototype cục bộ dày (C,16,256), counts, ĉ_G nén + sparse, |N_j|
  confusion/round_NNN.npz  (C,16,16) int64 cho mỗi quy tắc
  metrics/round_NNN.json   10 metric/client × 2 quy tắc + 3 mức tổng hợp + digest
  metrics/history.csv      dựng lại từ các file JSON ở trên mỗi lần ghi
  client_log/round_NNN.csv rows, steps, applied/skipped, ce, reg, grad_norm, giây
  preds/y_true.npy         một lần
  preds/round_NNN_{proto,clf}.npy  chỉ ở round cuối
  resume/round_NNN.pt      round, RNG, ĉ_G, μ, fingerprint      ← TRƯỚC marker
  resume/round_NNN.wK.pt   AdamW + GradScaler của client thuộc worker K
  complete/round_NNN.done  ← CUỐI CÙNG
```

**Lưu trọng số, không lưu module.** `weights/round_NNN.pt` chỉ chứa tensor phẳng; dựng lại mô
hình bằng `FlatPacker.to_state_dict()` rồi `load_state_dict(strict=True)`. `config.json` mang
`packer_manifest` (tên khoá, hình dạng, thứ tự) nên việc dựng lại không cần đoán.

Ngân sách đĩa (fp32): 20 client 1,59 GB, 50 client 3,98 GB, 100 client **7,97 GB** cho 50 round,
cộng ~0,3 GB resume và **~2,15 GB** dự đoán ở round cuối của kịch bản 100 client
(100 client x 10.761.343 dòng x 2 byte: hai quy tắc dự đoán `proto` và `clf` đều được lưu).

Chỉ giữ **một** blob resume: blob của round trước bị xoá **sau khi** marker mới đã ghi.

---

## 6. Những gì đã kiểm chứng, và ở đâu

Xem [TESTLOG.md](TESTLOG.md) để biết chi tiết từng lần chạy. Tóm tắt phân biệt rõ hai mức:

**Đã kiểm ở local (CPU, **1 worker**, `compile=False`) — đúng/sai:**
mask CPS, 10 metric so với sklearn, gộp prototype, hợp đồng commit/resume mô phỏng crash,
import từ mount chỉ đọc, verifier bắt được 9/9 kiểu hỏng cố ý, fold BatchNorm, round-trip
trọng số phẳng, vòng lặp FL đầy đủ trên fixture. Bộ test local chạy **trên CPU một worker** để
vừa 7,6 GB RAM của máy phát triển; nó **không** kiểm chứng đường đi hai GPU.

**Đã kiểm trên Kaggle 2×T4 (2026-09-07/08) — đúng/sai:**
hai worker thật trên `cuda:0`/`cuda:1`, AMP, `torch.compile` bật được và tương đương số học
(max |Δlogit| ≤ 4,5e-07, 0 bất đồng trên hàng quyết định), artifact + verifier (335 check/kịch bản),
và resume qua ranh giới session — cả ba kịch bản 20/50/100 trên mẫu dữ liệu thật.

**Chỉ đo được trên Kaggle 2×T4 — nhanh/chậm và khả thi:**
thời gian một round thật trên dữ liệu đầy đủ, thời gian prepack 43 M dòng, `eval_group` tối ưu,
và do đó `max_hours`.

## 7. Quyết định và sửa chữa sau review ngày 2026-09-07

Người dùng xác nhận giữ phương pháp/cấu hình ở trên, nhưng yêu cầu **sweep μ độc lập cho từng
kịch bản 20/50/100 client**. Mỗi sweep dùng batch tương ứng 512/512/256, 2% validation từng client,
grid multiplier `[0.03,0.1,0.3,1,3]`, 4 round. Không dùng test để chọn μ. Mọi ứng viên phải đủ
4 round; nếu lỗi/hết giờ thì chỉ lưu trạng thái incomplete, không có winner. μ tuyệt đối thắng
của kịch bản đó được đưa vào production cùng kịch bản (100% train). Không chuyển μ giữa các
kịch bản. Notebook production dừng trước decode nếu chưa gắn đúng output sweep hoàn chỉnh.

Validation là lấy mẫu theo hàng, có thể chứa các hàng tương quan cùng receiver/session ở cả
hai phần train/validation; scaler có sẵn đã fit toàn bộ train kể cả holdout. Đây không phải
holdout độc lập theo thời gian/session, cần công bố khi trình bày kết quả chọn μ.

**Artifact v2:** fingerprint khóa cả cấu hình khoa học, scaler, class/feature order, validation
và worker assignment. RNG được lưu ở từng worker, mọi checkpoint mới đọc bằng
`torch.load(..., weights_only=True)`. Marker chứa SHA256 của artifact từng round; optimizer
chỉ giữ round cuối. Không ghép checkpoint trước review với mã v2; lỗi index validation và RNG
cũ không thể sửa hồi tố bằng sửa metadata. Resume từ v2 cần đúng notebook source/config.

Phạm vi identity dữ liệu: fingerprint parquet dựa trên footer, schema và đường dẫn tương đối
client; không phải SHA256 toàn bộ dữ liệu. Hai bản rewrite có cùng thống kê footer vẫn có thể
trùng fingerprint. Dùng dataset version cố định; hash validation arrays và artifact là hash
nội dung đầy đủ. Marker checksum giúp phát hiện hỏng file, không phải chữ ký chống người sửa
đồng thời cả artifact và manifest.

Báo cáo mặc định dùng **round hoàn thành cuối**, hiển thị số round thực tế/round kế hoạch.
Dừng sớm không được gọi là hoàn thành 50 round. Thời gian setup/compile và commit phải tính
trong ngân sách phiên; `logs/timing_NNN.json` chứa thời gian round gồm commit.

**Đây là mô phỏng FL trên một máy.** Công thức communication cost chỉ đếm payload prototype
của giao thức, không đo bytes của multiprocessing, artifact, mask hoặc metadata. Tiến trình điều
phối kiểm thử vẫn đọc class counts và trọng số để audit/checkpoint; không có secure aggregation,
DP, hay bảo đảm riêng tư triển khai. Câu “không trao đổi trọng số” ở phần phương pháp nói về
cập nhật học liên kết, không nói về IPC phục vụ lưu artifact trong simulator.

Các số đo performance local cũ không phải bằng chứng sm_75. Ngày 2026-09-07 WSL 8 GB bị sập
khi test bị chạy chồng; mọi test local mới phải tuần tự qua watchdog, mô phỏng notebook CPU nhỏ.
Kiểm tra CUDA/AMP/compile và hai GPU được chuyển sang Kaggle; xem TESTLOG để biết thực sự đã pass gì.

---

## 8. Số liệu suy từ dữ liệu, riêng cho TinyProto

Chuyển từ `knowledge/DATASET.md` ngày 2026-09-08: `knowledge/` chỉ giữ những gì đúng cho **mọi**
phương pháp dựng trên VeReMi, còn hai bảng dưới đây phụ thuộc TinyProto (s, d, Eq. 11) nên
thuộc về hồ sơ phương pháp này. Thống kê `n_ij` gốc vẫn ở `knowledge/DATASET.md` §3.

### 8.1 Chi phí truyền tin mỗi round

Theo công thức Table 2 của bài báo, `Σᵢ (K_i + K) × chiều`:

| client | FedProto (d = 256) | TinyProto (s = 50) | tỉ lệ nén |
|---:|---:|---:|---:|
| 20 | 0,1638 M | 0,0320 M | 5,1× |
| 50 | 0,4081 M | 0,0797 M | 5,1× |
| 100 | 0,8151 M | 0,1592 M | 5,1× |

`d = 256` là chiều feature layer của DAGSNet (`knowledge/ARCHITECTURE.md` §2.6: `head[3]` = ReLU
sau `Linear(544, 256)`). Bài báo dùng `d = 500` nên `s = 50` ở đó là nén 90%; ở đây `s = 50` chỉ
là nén 80,5%. Muốn giữ **đúng tỉ lệ nén 90%** của bài báo thì `s = 26`.

### 8.2 Ước lượng μ (APS) — chỉ là điểm khởi đầu, KHÔNG phải μ đang dùng

Hằng số scale ở Eq. (11). Prototype toàn cục `ĉ^G_j = (1/|N_j|) Σ n_{i,j} ĉ^L_{i,j}` có độ lớn
≈ `c̄_j × mean(n_{i,j})`, nên μ phải nghịch đảo đại lượng đó thì `μ ĉ^G_j` mới cùng thang với
prototype cục bộ:

| client | μ = 1 / mean(n_ij) | μ theo tỉ lệ suy từ bài báo (≈ 0,125 / mean n_ij) |
|---:|---:|---:|
| 20 | 7,43 × 10⁻⁶ | 9,29 × 10⁻⁷ |
| 50 | 1,85 × 10⁻⁵ | 2,32 × 10⁻⁶ |
| 100 | 3,68 × 10⁻⁵ | 4,60 × 10⁻⁶ |

⚠ Cột thứ hai là **suy luận**, không phải giá trị của tác giả. Bài báo chỉ công bố μ cho 5 bộ dữ
liệu ảnh (1,5×10⁻⁴ cho CIFAR-10 … 5,0×10⁻³ cho Flowers-102) và nói phương pháp chọn nằm ở phụ lục
**mà bản markdown đang có không chứa**. Hệ số 0,125 là tỉ số quan sát được giữa μ công bố và
`1/mean(n_ij)` ước lượng lại cho CIFAR-10/CIFAR-100. Phải ghi μ là **lựa chọn của bản dựng này**,
không được gán cho tác giả.

⚠⚠ **μ THỰC SỰ DÙNG cho production không phải hai cột trên.** Nó đến từ grid search theo giao
thức v1 (CONTEXT.md §5b): 20c `7.585696e-07`, 50c `5.646573e-06`, 100c `1.126461e-05`. Hai cột
trên chỉ là ước lượng ban đầu dùng để đặt lưới quét; đừng chép chúng vào notebook.
