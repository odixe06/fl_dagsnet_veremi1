# Máy local — cấu hình đo được và những gì test được / không test được ở đây

## Sự cố WSL và quy tắc bắt buộc — cập nhật 2026-09-07

**WSL chỉ được cấp 8 GB RAM** (Linux thấy 7.803 MiB ≈ 7,62 GiB), không phải RAM của
toàn bộ máy Windows. Người dùng báo WSL vừa tràn RAM và sập trong phiên review này.
Hai bộ kiểm thử resume/sweep đã được khởi chạy chồng; chưa có kernel OOM log để kết luận
tiến trình cụ thể. Sau khởi động lại: không còn tiến trình train/test của phiên cũ,
MemAvailable khoảng 5.148 MiB, swap 0/2.048 MiB. Đây là snapshot, phải đo lại trước test.

- Chỉ chạy **một** cây tiến trình kiểm thử tại một thời điểm; không chạy test GPU hoặc
  đọc dữ liệu nặng song song, kể cả test khác đang chờ verifier.
- Local dùng **một worker**, fixture nhỏ và `compile=False`. Hai worker/hai GPU và
  compile sm_75 kiểm trên Kaggle. Không chạy benchmark Kaggle nguyên kích thước ở local.
- Mọi kiểm thử local phải đi qua watchdog dưới đây: khóa chống chạy chồng, tính RSS
  toàn bộ process group (cả cháu/chắt), dừng riêng nhóm test nếu RSS vượt tối đa 3.000 MiB
  hoặc hệ thống còn dưới 1.536 MiB MemAvailable. Ngưỡng RSS giảm thêm theo RAM đang rảnh.
- Đặt OMP/MKL/OpenBLAS thread = 1; stream parquet theo batch nhỏ. Không nạp toàn bộ train,
  không bật Inductor ở local, không dùng `ulimit -v` cho CUDA.
- Watchdog là hàng rào chủ động, không đảm bảo tuyệt đối trước tăng RAM đột biến giữa
  hai lần đo hoặc ứng dụng khác chiếm RAM. Nếu watchdog dừng, giảm bài test/chuyển Kaggle;
  không tăng ngưỡng để cố chạy qua.

```bash
# mọi lệnh nặng đều bọc qua run_local_checked.py; phần sau dấu \ là bài test của dự án
conda run --no-capture-output -n nckh python scripts/run_local_checked.py \
  python <script của bạn> [tham số]
```

**Sức chứa đo được 2026-09-08.** Một suite tuần tự đầy đủ (mô phỏng notebook trên CPU,
unit/regression, verifier, resume, tamper) chạy trọn với peak RSS toàn cây **2.072 MiB**; một
smoke test dữ liệu thật quy mô nhỏ đạt peak **932 MiB**. Sau khi đóng bớt ứng dụng:
MemAvailable ~**5.297 MiB**, swap 0 MiB (snapshot).

Nghĩa là: một cây test tuần tự, CPU, 1 worker, không Inductor thì **vừa** máy này. Chạy song song
nhiều cây, hoặc bật Inductor, hoặc nạp cả tập train vào RAM thì **không**. Mọi kết luận ở đây chỉ
áp cho pipeline CPU; **không** chứng nhận bất cứ điều gì về GPU của Kaggle.
Log cụ thể của từng dự án nằm trong thư mục của dự án đó, không nằm ở `knowledge/`.

Các mục bên dưới chứa số đo/lịch sử trước sự cố; quy tắc mới ở đây được ưu tiên.

**Mục đích.** Máy này là máy **duy nhất** dùng để phát triển. File này ghi cấu hình thật (đo bằng
lệnh, không chép), và quan trọng hơn: **ranh giới** của nó — cái gì kiểm chứng được ở local và
cái gì bắt buộc phải chờ Kaggle. Đo ngày **2026-09-07**.

Dựng lại: `python knowledge/probe_local.py` (ghi `local_env.json`).

---

## 1. Cấu hình

| | |
|---|---|
| OS | Ubuntu 24.04.4 LTS trên **WSL2**, kernel `6.18.33.2-microsoft-standard-WSL2` |
| CPU | Intel Core i5-12500H — 8 core / **16 thread** |
| RAM | **7,6 GiB tổng**, thường đã dùng ~3,0 GiB → **còn ~4,7 GiB khả dụng** |
| Swap | 2,0 GiB (đã dùng 1,6 GiB — máy đang chịu áp lực bộ nhớ) |
| Đĩa | `/dev/sdd` 1007 GB, còn **896 GB** |
| GPU | **NVIDIA GeForce RTX 3050 Laptop**, **4,00 GiB VRAM**, **sm_86** (Ampere), 16 SM, driver 610.62 |
| VRAM rảnh | 3,22 / 4,00 GiB (phần còn lại là context của driver) |
| Python | 3.12.13, conda env **`nckh`** |
| PyTorch | **2.13.0+cu130**, CUDA 13.0, cuDNN 9.2.0, triton 3.7.1 |
| Thư viện | numpy 2.5.2, pyarrow 25.0.0, pandas 3.0.5, sklearn 1.9.0, wandb 0.29.0, kaggle 2.2.4 |

Kích hoạt môi trường — **mọi lệnh Python và Kaggle CLI đều phải chạy trong env này**:

```bash
source /home/odixe/miniforge3/etc/profile.d/conda.sh && conda activate nckh
```

⚠ `conda run -n nckh python - <<'PY' … PY` **nuốt stdout của script**. Một script sửa file báo
thành công sẽ không in gì, và một assertion hỏng trông y hệt như im lặng. Dùng
`source … && conda activate nckh && python …` hoặc ghi script ra file rồi chạy file đó.

---

## 2. Ba ranh giới cứng — đọc trước khi viết bất kỳ script test nào

### 2.1 RAM 7,6 GiB, thực tế còn ~4,7 GiB

| thứ | dung lượng | vừa RAM? |
|---|---:|---|
| train 43.045.415 × 66 **float32** | 11,36 GB | ✘ chết máy |
| train 43.045.415 × 66 **float16** | 5,68 GB | ✘ tràn swap |
| test 10.761.343 × 66 float32 | 2,84 GB | ⚠ sát trần |
| test 10.761.343 × 66 float16 | 1,42 GB | ✔ |
| một client 20-client lớn nhất (5.890.990 dòng) fp16 | 0,78 GB | ✔ |
| một client 100-client trung vị (383.001 dòng) fp16 | 0,05 GB | ✔ |

Quy tắc: **không bao giờ `to_table()` hay `read_parquet()` cả một split ở local.** Luôn stream
theo batch qua `pyarrow.dataset.to_batches(batch_size=…)` và tích luỹ thống kê. Script
`audit_dataset.py` trong thư mục này quét đủ 43 M + 10,7 M dòng mà giữ RSS ~1,6 GiB — dùng nó
làm khuôn mẫu.

Swap đã dùng 1,6/2,0 GiB ngay cả lúc rảnh. Vượt RAM ở đây không cho `MemoryError` sạch sẽ mà
làm cả WSL đơ, nên hàng rào phải là **chủ động** (batch nhỏ), không phải bắt exception.

### 2.2 VRAM 4 GiB — chỉ chạy được fixture, không chạy được kịch bản thật

Kaggle có **2 × T4 16 GiB = 32 GiB**. Local có **1 × 3,99 GiB**. Đo được: forward+backward
DAGSNet ở batch 512 chiếm peak **1,58 GiB**, nên train một client thì được; nhưng:

* **không** giữ được train fp16 thường trú (5,29 GiB) như notebook Kaggle làm;
* **không** giữ được test fp16 thường trú (1,32 GiB) cùng lúc với nhiều model;
* eval `vmap` gộp G client ở batch 32.768 **OOM ngay từ G = 4** — tensor activation
  `(G, B, 192, 11)` fp16 đã là 1,1 GB ở G = 8. Muốn đo tỉ lệ thì phải giữ **G × batch cố định**.

### 2.3 sm_86 ≠ sm_75 — bốn khác biệt làm sai kết luận nếu bỏ qua

| | local RTX 3050 (sm_86, Ampere) | Kaggle T4 (sm_75, Turing) |
|---|---|---|
| bf16 | **có** (`is_bf16_supported() == True`) | **không** — rơi về đường emulate rất chậm |
| TF32 | có | **không** |
| số GPU | **1** | **2** |
| fp16 tensor core | có | có |

Hệ quả bắt buộc: **luôn ép `dtype=torch.float16` trong `autocast`**, không dùng bf16, không bật
cờ TF32 — nếu không, code chạy ngon ở local rồi hỏng hoặc chậm bất thường trên T4.

---

## 3. Số đo local — dùng làm **tỉ lệ**, không phải dự báo tuyệt đối

DAGSNet 395.024 tham số, đầu vào `(B, 66)`, AMP fp16, AdamW fused, clip 1.0.

**Train (eager, chưa `torch.compile`):**

| batch | ms/step | samples/s |
|---:|---:|---:|
| 256 | 17,77 | 14.405 |
| 512 | 18,28 | 28.006 |
| 1024 | 22,05 | 46.439 |

ms/step gần như **phẳng** từ 256 → 512: GPU đang **chờ CPU phóng kernel** (launch-bound), không
phải thiếu năng lực tính. Đây là lý do `torch.compile(mode="reduce-overhead")` là khoản lời lớn
nhất trên model này (đo trên T4: 18,52 → 6,49 ms/step, **2,86×**), và là lý do batch 256 của
kịch bản 100 client tốn ~1,9× thời gian chứ không phải 1,0×.

**Eval (inference_mode + AMP, model đơn, chưa fold BN):**

| batch | ms/batch | samples/s |
|---:|---:|---:|
| 16.384 | 120,99 | 135.420 |
| 65.536 | 317,14 | 206.650 |

Đối chiếu: T4 đo được **320.296 samples/s** ở batch 16.384. Tức **T4 ≈ 2,4× máy local cho eval**.
Dùng tỉ lệ này để ước lượng, rồi **đo lại trên T4 trước khi trích dẫn bất kỳ con số nào**.

Peak VRAM khi train batch 512: **1,58 GiB**.

---

## 4. Test được gì ở local

**✔ Kiểm chứng được đầy đủ (đúng/sai, không phải nhanh/chậm):**

* số tham số 395.024, `load_state_dict(strict=True)`, round-trip flat-vector ↔ state_dict
  (đo được **max|Δlogit| = 0,0**);
* thiết kế mask CPS, tỉ lệ nén, khoảng cách Hamming;
* 10 metric đối chiếu `sklearn` (đo được **max|Δ| = 1,1 × 10⁻¹⁶**);
* gộp prototype, APS, hàm regularization, quy tắc Eq. (12);
* hợp đồng commit/resume: mô phỏng crash, round dở bị làm lại, `history.csv` không trùng dòng;
* fold BatchNorm đúng (đo được max|Δlogit| = 8,7 × 10⁻⁵ trên model khởi tạo ngẫu nhiên);
* đọc parquet, thứ tự cột, áp scaler — trên **một** client nhỏ;
* validator notebook, `kernel-metadata.json`, slug.

**✘ KHÔNG kiểm chứng được ở local — bắt buộc đo trên Kaggle:**

* mọi con số thời gian tuyệt đối (GPU khác thế hệ, khác số lượng);
* topology **2 worker / 2 GPU** — local chỉ có 1 GPU, chạy được 1 worker;
* `torch.compile(mode="reduce-overhead")` trên **sm_75** (Triton sm_75 là rủi ro đã biết;
  local là sm_86 nên pass ở đây **không** chứng minh gì);
* G tối ưu cho eval `vmap` (4 GiB OOM trước khi chạm vùng cần đo);
* prepack toàn bộ 43 M dòng (không đủ RAM lẫn VRAM);
* thời gian một round thật, và do đó ngân sách `max_hours`.

Khi báo cáo, luôn tách riêng: **"đã kiểm ở local"** và **"đã kiểm trên 2×T4"**. Cái trước không
chứng minh cái sau.

---

## 5. Kaggle — trạng thái tài khoản đo ngày 2026-09-07

`python .agents/skills/kaggle-training-notebook/scripts/kaggle_account.py list|quota`

| tài khoản | GPU còn | TPU còn | ghi chú |
|---|---:|---:|---|
| khanhngoc0304 | 30,00 h | 20,00 h | chưa dùng |
| odixe0502 | 30,00 h | 20,00 h | chủ sở hữu các dataset |
| **minhtran0601** (đang active) | **9,66 h** | 20,00 h | đang chạy 2 session AFPHA (20c, 50c) |

Quota refresh **2026-09-12T00:00:00Z**. Giới hạn: **tối đa 2 batch GPU session mỗi tài khoản**
chạy đồng thời; session dài tối đa **12 h**.

Đây là số liệu **có ngày tháng**, không phải số dư tương lai — đo lại trước mỗi lần launch.

**Roster và cách đổi tài khoản.** Số dư quota **không** ghi ở đây: nó đổi hằng tuần, nên bảng
trên chỉ là ví dụ có ngày tháng. Đọc lại bằng
`.agents/skills/kaggle-training-notebook/scripts/kaggle_account.py quota` trước mỗi lần launch.
Đổi tài khoản đi qua `kaggle_account.py use/ensure --confirm`, **không** copy tay
`credentials.json` — copy tay chỉ đổi CLI, để `.mcp.json`/host config giữ token cũ và hai bên
lệch nhau. Sau khi đổi, **reconnect MCP client** của host.

Probe MCP trực tiếp (`probe_kaggle_mcp.py`, hoặc gọi thẳng endpoint) trả `Unauthenticated` cho
**cả bốn** bearer đã lưu trong ngày 2026-09-08, kể cả hai tài khoản đang chạy tốt qua OAuth cùng
lúc đó. Vậy probe **không** dùng để nghiệm thu hay loại bỏ một MCP token mới; chỉ lời gọi native
của host mới kết luận được.

### Điều chỉnh kiểm thử sau khi watchdog chặn

Hai lần GPU smoke với một worker vẫn chạm ngưỡng RSS 3.000 MiB; watchdog dừng đúng nhóm test,
WSL không sập, swap không tăng. Không nới ngưỡng. Mô phỏng notebook local hiện dùng **CPU,
FP32, một worker, batch 64, eval batch 512**, fixture 4×2.048 train và 1.024 test.
Sweep 2 ứng viên × 2 round đã pass với peak RSS cây test khoảng **1.571 MiB**.
Đây là kiểm tra pipeline trên fixture, không chứng nhận AMP/compile/T4 hoặc tốc độ dữ liệu thật.
