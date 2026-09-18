# Máy local — cấu hình đo được và ranh giới của nó

**Mục đích.** Máy này là máy **duy nhất** dùng để phát triển. File này ghi cấu hình thật (đo bằng
lệnh, không chép) và — quan trọng hơn — **ranh giới**: cái gì kiểm chứng được ở local và cái gì
bắt buộc phải chờ Kaggle. Đo ngày **2026-09-07**.

**File này chỉ chứa thuộc tính bền của máy.** Số dư RAM/VRAM/đĩa còn trống, quota tài khoản,
tiến trình đang chạy và kết quả của một chiến dịch test cụ thể đều **đổi theo thời gian**: đo lại
tại chỗ, đừng đọc ở đây. Xem `README.md` cho ranh giới giữa `knowledge/` và `CONTEXT.md`.

---

## 1. Cấu hình

| | |
|---|---|
| OS | Ubuntu 24.04.4 LTS trên **WSL2**, kernel `6.18.33.2-microsoft-standard-WSL2` |
| CPU | Intel Core i5-12500H — 8 core / **16 thread** |
| RAM | **7,6 GiB tổng** — đây là hạn mức WSL, không phải RAM của máy Windows |
| Swap | 2,0 GiB |
| GPU | **NVIDIA GeForce RTX 3050 Laptop**, **4,00 GiB VRAM**, **sm_86** (Ampere), 16 SM |
| Python | 3.12.13, conda env **`nckh`** |
| PyTorch | **2.13.0+cu130**, CUDA 13.0, cuDNN 9.2.0, triton 3.7.1 |
| Thư viện | numpy 2.5.2, pyarrow 25.0.0, pandas 3.0.5, sklearn 1.9.0, wandb 0.29.0, kaggle 2.2.4 |

Đo lại RAM/VRAM còn trống trước mỗi đợt test: `grep MemAvailable /proc/meminfo`,
`nvidia-smi --query-gpu=memory.free --format=csv`.

Kích hoạt môi trường — **mọi lệnh Python và Kaggle CLI đều phải chạy trong env này**:

```bash
source /home/odixe/miniforge3/etc/profile.d/conda.sh && conda activate nckh
```

⚠ `conda run -n nckh python - <<'PY' … PY` **nuốt stdout của script**. Một script sửa file báo
thành công sẽ không in gì, và một assertion hỏng trông y hệt như im lặng. Dùng
`source … && conda activate nckh && python …` hoặc ghi script ra file rồi chạy file đó.

---

## 2. Quy tắc bắt buộc khi chạy test ở local

WSL chỉ được cấp 8 GB RAM. Vượt RAM ở đây **không** cho `MemoryError` sạch sẽ mà làm cả WSL đơ —
đã xảy ra một lần do chạy chồng hai cây test. Hàng rào vì thế phải **chủ động**, không phải bắt
exception:

- Chỉ chạy **một** cây tiến trình kiểm thử tại một thời điểm; không chạy test GPU hoặc đọc dữ
  liệu nặng song song, kể cả khi test kia đang chờ verifier.
- Local dùng **một worker**, fixture nhỏ và `compile=False`. Hai worker / hai GPU và compile
  sm_75 chỉ kiểm được trên Kaggle. Không chạy benchmark nguyên kích thước ở local.
- Mọi test nặng phải đi qua watchdog: khóa chống chạy chồng, tính RSS toàn bộ process group (cả
  cháu/chắt), dừng riêng nhóm test nếu RSS vượt trần hoặc hệ thống còn dưới 1.536 MiB
  MemAvailable. Trần RSS giảm thêm theo RAM đang rảnh.
- Đặt OMP/MKL/OpenBLAS thread = 1; stream parquet theo batch nhỏ. Không nạp toàn bộ train,
  không bật Inductor ở local, không dùng `ulimit -v` cho CUDA.
- Watchdog là hàng rào chủ động, không đảm bảo tuyệt đối trước tăng RAM đột biến giữa hai lần đo
  hoặc ứng dụng khác chiếm RAM. **Nếu watchdog dừng: giảm bài test hoặc chuyển lên Kaggle — không
  nới ngưỡng để cố chạy qua.**

```bash
# mọi lệnh nặng đều bọc qua watchdog; phần sau dấu \ là bài test của dự án
conda run --no-capture-output -n nckh python scripts/run_local_checked.py \
  python <script của bạn> [tham số]
```

`scripts/run_local_checked.py` thuộc từng dự án, không nằm ở `knowledge/`.

---

## 3. Ba ranh giới cứng — đọc trước khi viết bất kỳ script test nào

### 3.1 RAM 7,6 GiB

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

### 3.2 VRAM 4 GiB — chỉ chạy được fixture, không chạy được kịch bản thật

Kaggle có **2 × T4 16 GiB = 32 GiB**. Local có **1 × 3,99 GiB**. Đo được: forward+backward
DAGSNet ở batch 512 chiếm peak **1,58 GiB**, nên train một client thì được; nhưng:

* **không** giữ được train fp16 thường trú (5,29 GiB) như notebook Kaggle làm;
* **không** giữ được test fp16 thường trú (1,32 GiB) cùng lúc với nhiều model;
* mọi phép gộp nhiều model để eval song song đều OOM sớm hơn vùng cần đo — activation
  `(G, B, 192, 11)` fp16 đã là 1,1 GB ở G = 8. Muốn đo tỉ lệ thì phải giữ **G × batch cố định**.

### 3.3 sm_86 ≠ sm_75 — bốn khác biệt làm sai kết luận nếu bỏ qua

| | local RTX 3050 (sm_86, Ampere) | Kaggle T4 (sm_75, Turing) |
|---|---|---|
| bf16 | **có** (`is_bf16_supported() == True`) | **không** — rơi về đường emulate rất chậm |
| TF32 | có | **không** |
| số GPU | **1** | **2** |
| fp16 tensor core | có | có |

Hệ quả bắt buộc: **luôn ép `dtype=torch.float16` trong `autocast`**, không dùng bf16, không bật
cờ TF32 — nếu không, code chạy ngon ở local rồi hỏng hoặc chậm bất thường trên T4.

---

## 4. Số đo local — dùng làm **tỉ lệ**, không phải dự báo tuyệt đối

DAGSNet 395.024 tham số, đầu vào `(B, 66)`, AMP fp16, AdamW fused, clip 1.0.

**Train (eager, chưa `torch.compile`):**

| batch | ms/step | samples/s |
|---:|---:|---:|
| 256 | 17,77 | 14.405 |
| 512 | 18,28 | 28.006 |
| 1024 | 22,05 | 46.439 |

ms/step gần như **phẳng** từ 256 → 512: GPU đang **chờ CPU phóng kernel** (launch-bound), không
phải thiếu năng lực tính. Đây là lý do `torch.compile(mode="reduce-overhead")` là khoản lời lớn
nhất trên model này (đo trên T4: 18,52 → 6,49 ms/step, **2,86×**), và là lý do một kịch bản dùng
batch nhỏ hơn một nửa tốn ~1,9× thời gian chứ không phải 1,0×.

**Eval (inference_mode + AMP, model đơn, chưa fold BN):**

| batch | ms/batch | samples/s |
|---:|---:|---:|
| 16.384 | 120,99 | 135.420 |
| 65.536 | 317,14 | 206.650 |

Đối chiếu: T4 đo được **320.296 samples/s** ở batch 16.384. Tức **T4 ≈ 2,4× máy local cho eval**.
Dùng tỉ lệ này để ước lượng, rồi **đo lại trên T4 trước khi trích dẫn bất kỳ con số nào**.

Peak VRAM khi train batch 512: **1,58 GiB**.

---

## 5. Test được gì ở local

**✔ Kiểm chứng được đầy đủ (đúng/sai, không phải nhanh/chậm):**

* số tham số 395.024, `load_state_dict(strict=True)`, round-trip flat-vector ↔ state_dict
  (đo được **max|Δlogit| = 0,0**);
* dựng lại mô hình từ file trọng số của một round bất kỳ (weights-only, `weights_only=True`);
* 10 metric đối chiếu `sklearn` (đo được **max|Δ| = 1,1 × 10⁻¹⁶**);
* mọi công thức toán của phương pháp: hàm loss, quy tắc gộp, hàm regularization;
* hợp đồng commit/resume: mô phỏng crash, round dở bị làm lại, `history.csv` không trùng dòng;
* fold BatchNorm đúng (đo được max|Δlogit| = 8,7 × 10⁻⁵ trên model khởi tạo ngẫu nhiên);
* đọc parquet, thứ tự cột, áp scaler — trên **một** client nhỏ;
* validator notebook, `kernel-metadata.json`, slug.

**✘ KHÔNG kiểm chứng được ở local — bắt buộc đo trên Kaggle:**

* mọi con số thời gian tuyệt đối (GPU khác thế hệ, khác số lượng);
* topology **2 worker / 2 GPU** — local chỉ có 1 GPU, chạy được 1 worker;
* `torch.compile(mode="reduce-overhead")` trên **sm_75** (Triton sm_75 là rủi ro đã biết;
  local là sm_86 nên pass ở đây **không** chứng minh gì);
* mọi phép gộp nhiều model để eval song song (4 GiB OOM trước khi chạm vùng cần đo);
* prepack toàn bộ 43 M dòng (không đủ RAM lẫn VRAM);
* thời gian một round thật, và do đó ngân sách `max_hours`.

Khi báo cáo, luôn tách riêng: **"đã kiểm ở local"** và **"đã kiểm trên 2×T4"**. Cái trước không
chứng minh cái sau.

---

## 6. Kaggle — những ràng buộc không đổi

Giới hạn dịch vụ: **tối đa 2 batch GPU session mỗi tài khoản** chạy đồng thời (push thứ ba bị từ
chối thẳng), session dài tối đa **12 h**, `/kaggle/working` bị **xoá khi session mới bắt đầu**.

**Quota và roster tài khoản không ghi ở đây** — chúng đổi hằng tuần. Đọc bằng
`kaggle_account.py list|quota|health|plan` của skill `kaggle-training-notebook` trước mỗi lần
launch, và budget theo `total_time_allowed − time_used − time_reserved`.

Đổi tài khoản đi qua `kaggle_account.py use/ensure --confirm`, **không** copy tay
`credentials.json` — copy tay chỉ đổi CLI, để `.mcp.json`/host config giữ token cũ và hai bên
lệch nhau. Sau khi đổi, **reconnect MCP client** của host.

Probe MCP trực tiếp (`probe_kaggle_mcp.py`, hoặc gọi thẳng endpoint) đã trả `Unauthenticated`
cho **cả bốn** bearer đã lưu trong ngày 2026-09-08, kể cả hai tài khoản đang chạy tốt qua OAuth
cùng lúc đó. Vậy probe **không** dùng để nghiệm thu hay loại bỏ một MCP token; chỉ lời gọi
native của host mới kết luận được.
