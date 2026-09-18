# VeReMi NextGen — bộ dữ liệu federated α = 0,5 và tập test toàn cục

**Mục đích.** Mọi con số trong file này **đo thật** bằng `audit_dataset.py` và `fl_class_stats.py`
nằm cạnh nó, chạy ngày **2026-09-07** trên `/home/odixe/nckh/veremi/dataset`. Không có số nào chép tay từ
README hay sidecar mà không đối chiếu. Dùng file này thay cho việc quét lại dữ liệu ở phiên sau —
bộ dữ liệu này **cố định**, các build tiếp theo dùng đúng bộ này.

Dựng lại toàn bộ số liệu:

```bash
source /home/odixe/miniforge3/etc/profile.d/conda.sh && conda activate nckh
python knowledge/audit_dataset.py     # ~4 phút, ghi dataset_audit.json
python knowledge/fl_class_stats.py    # ~1 giây,  ghi fl_class_stats.json
```

Kèm theo: `dataset_audit.json` (kết quả máy đọc, đủ per-feature), `fl_class_stats.json`
(thống kê `n_{i,j}` cho prototype-based FL), `meta.json`, `scaler.json`.

---

## 1. Ba điều bắt buộc đọc trước khi viết code

### 1.1 `train/` ĐÃ chuẩn hoá, `test/` thì CHƯA — đo được, không phải nghe nói

| tập | `max |mean|` của 66 cột `f_*` | `std` (min … max) | kết luận |
|---|---:|---|---|
| train (cả 43.045.415 dòng) | **3,04 × 10⁻⁸** | 0,99999996 … 1,00000003 | **đã z-score** |
| test thô (10.761.343 dòng) | **3.248,2** | — | **chưa chuẩn hoá** |
| test sau khi áp `scaler.json` | 0,1983 | 0,6878 … 1,1051 | đúng như mong đợi |

Chuẩn hoá lại `train/` lần hai sẽ **không báo lỗi** — nó chỉ chia thêm cho 1,0 nếu bạn fit lại
trên chính nó, hoặc phá dữ liệu nếu bạn áp `scaler.json` lần nữa. Quên chuẩn hoá `test/` cũng
không báo lỗi. Dấu hiệu nhận biết đã đo:

```
mean(f_snd_spd) trong test thô = 7,4493      (nếu ≈ 0 thì bạn đã chuẩn hoá rồi)
```

Độ lệch train↔test sau chuẩn hoá là **thật, không phải lỗi**: `f_rcv_x_rel` / `f_snd_x_rel` có
mean +0,198 và std 1,104; `f_rcv_y_rel` / `f_snd_y_rel` có std 0,688. Split theo thời gian mô
phỏng nên hình học bản đồ dịch chuyển giữa hai nửa. Phải nêu điều này khi báo cáo.

### 1.2 Nhãn số nguyên nằm sẵn ở cột `label` — không cần giải mã chuỗi

Cả `train/` lẫn `test/` có **87 cột**: 66 cột `f_*` + 21 cột metadata. Trong đó:

* `label` — **`int8`, giá trị 0…15**, đúng bằng chỉ số lớp trong `meta.json["class_names"]`.
  Đã kiểm: `label == class_index(attack_type)` trên mọi dòng đã quét, và
  `bincount(label)` trên **đủ 10.761.343 dòng test** khớp từng con số với `meta.json["test_counts"]`.
* `attack_type` — cùng thông tin nhưng ở dạng `string`. Giải mã chuỗi trên 43 M dòng tốn hàng
  chục giây mỗi lần; **đọc `label` thay vì `attack_type`** trong bước prepack.
* `scenario` — `string`, 4 giá trị. Train có đủ `{highway_2, highway_7, urban_2, urban_7}`;
  **test chỉ có `{highway_7, urban_7}`**. Đây là hệ quả của split theo thời gian, phải công bố.

Trục lớp bị đóng băng theo `meta.json["class_names"]`, khớp `label_mapping.json["class_to_label"]`
và `["classes"]`. Không tự sinh lại thứ tự lớp ở phiên sau.

### 1.3 fp16 an toàn — đã đo, không suy đoán

| | max\|x\| sau chuẩn hoá | giới hạn fp16 | an toàn |
|---|---:|---:|---|
| train | **570,4353** | 65.504 | ✔ |
| test (sau `scaler.json`) | **570,4353** | 65.504 | ✔ |

Giá trị lớn nhất nằm ở `f_rcv_acl_noise` (−535,52 … +570,44); kế đó `f_sess_pos_pred_err` (292,17)
và `f_sess_dt` (178,07). Còn cách trần fp16 hai bậc độ lớn, nên lưu đặc trưng ở **float16** trên
GPU là an toàn — nhưng vẫn phải ghi vào báo cáo rằng giá trị đã bị lượng tử hoá về fp16.

**Không có giá trị non-finite nào** trong cả hai tập: `nan/inf` = 0 trên 43.045.415 dòng train và
10.761.343 dòng test (nguồn đã điền 0.0 trước khi cắt split).

---

## 2. Ba kịch bản federated — đo từ footer parquet

`fl_client/alpha05/{20,50,100}_client/train/client_id=NNN/part-*.parquet`, Dirichlet **α = 0,5**
trên label-skew. Cả ba kịch bản là **cùng 43.045.415 dòng** chia lại, không phải ba bộ dữ liệu khác nhau.

| client | file | dung lượng (GiB) | rows/client min | median | max | số lớp/client | client có đủ 16 lớp |
|---:|---:|---:|---:|---:|---:|---|---:|
| 20 | 446 | 7,44 | 870.217 | 1.817.459 | 5.890.990 | 16–16 | 20 / 20 |
| 50 | 1.069 | 8,25 | 199.063 | 713.022 | 2.630.929 | 15–16 | 44 / 50 |
| 100 | 1.992 | 8,86 | 98.180 | 383.001 | 1.333.839 | 14–16 | 86 / 100 |

Kiểm tra đã pass ở cả ba kịch bản:

* tổng số dòng từ footer = **43.045.415**, khớp `client_stats.json` **từng client một** (0 sai lệch);
* union số dòng theo lớp của mọi client = `meta.json["train_counts"]` **từng lớp một**;
* union phủ đủ **16/16 lớp**;
* thứ tự 66 cột `f_*` trong file client **và** trong file test = `meta.json["feature_cols"]`, đúng thứ tự.

⚠ **α = 0,5 ở đây là non-IID LỆCH TỈ LỆ, không phải THIẾU LỚP.** Vì mỗi client có hàng trăm
nghìn đến hàng triệu dòng, hầu như client nào cũng thấy gần đủ 16 lớp (tối thiểu 14). Đây là
khác biệt quan trọng so với các benchmark ảnh nhỏ (CIFAR-10 với Dir(0,1)) nơi nhiều client
**không có** lớp nào đó.

Hệ quả đo được: `|N_j| ≈ M` cho mọi lớp (20/20, 49–50/50, 97–100/100). Phương pháp nào tổng hợp
theo lớp trên tập client sở hữu lớp đó sẽ gần như luôn tổng hợp từ **tất cả** client — đừng thiết
kế dựa trên giả định có client thiếu lớp.

---

## 3. Thống kê phân mảnh theo client (`n_ij`)

Từ `fl_class_stats.json`. `n_{i,j}` = số dòng lớp `j` của client `i` (chỉ tính các cặp khác 0).

| client | K_i trung bình | Σᵢ(K_i + K) | n_ij min | median | mean | max | \|N_j\| min–max |
|---:|---:|---:|---:|---:|---:|---:|---|
| 20 | 16,00 | 640 | 4 | 35.384 | 134.517 | 3.451.254 | 20–20 |
| 50 | 15,88 | 1.594 | 1 | 13.756 | 54.213 | 2.177.693 | 49–50 |
| 100 | 15,84 | 3.184 | 2 | 7.094 | 27.175 | 878.656 | 97–100 |

`mean(n_ij)` là đại lượng hay dùng để đặt thang cho các hằng số phụ thuộc kích thước client
(ví dụ hệ số scale của prototype tổng hợp theo `n_ij`). Giá trị cụ thể cho từng phương pháp
**không** thuộc file này — chúng là lựa chọn của bản dựng, không phải thuộc tính dữ liệu.

---

## 4. Ngân sách tính toán — suy ra từ số dòng thật

Số bước optimizer mỗi round là `Σᵢ ceil(n_i / batch)`, **không** phải một epoch của một mô hình:

| client | batch | steps/round | steps cho 50 round |
|---:|---:|---:|---:|
| 20 | 512 | **84.083** | 4.204.150 |
| 50 | 512 | **84.098** | 4.204.900 |
| 100 | 256 | **168.200** | 8.410.000 |

Kịch bản 100 client tốn **2×** số bước dù cùng số dòng, vì batch nhỏ hơn một nửa. Trên model
launch-bound như DAGSNet, giảm batch **không** giảm giá mỗi bước tương ứng
(đo trên T4: 6,028 ms ở batch 256 so với 6,487 ms ở batch 512), nên kịch bản 100 client tốn
khoảng **1,9×** thời gian của kịch bản 20 client.

Dung lượng thường trú trên GPU nếu nạp hết vào VRAM ở fp16:

| | dòng | fp16 (GiB) | nhãn int8 (MiB) |
|---|---:|---:|---:|
| train (bất kỳ kịch bản nào) | 43.045.415 | **5,29** | 41,1 |
| test | 10.761.343 | **1,32** | 10,3 |

Cả hai vừa trong một T4 16 GB cùng model, Adam state và activation. Đây là lý do bỏ hẳn
`DataLoader` khỏi vòng lặp (xem `references/perf-federated.md` §4).

---

## 5. Vị trí dữ liệu

| | local | Kaggle |
|---|---|---|
| train FL | `/home/odixe/nckh/veremi/dataset/fl_client/alpha05/{20,50,100}_client/train/client_id=NNN/` | `odixe0502/veremi-fl-20client`, `-50client`, `-100client` |
| test toàn cục | `/home/odixe/nckh/veremi/dataset/centralized/test/part-0000{0..3}.parquet` | `odixe0502/veremi-nextgen2026-centralized`, thư mục `upload/test/` |
| scaler / meta | `knowledge/scaler.json`, `knowledge/meta.json` | `upload/scaler.json` trong dataset centralized |

Mount trên Kaggle **không** phải `/kaggle/input/<slug>/`. Với dataset centralized đã đo được
`/kaggle/input/datasets/odixe0502/veremi-nextgen2026-centralized/upload/`. Với ba dataset FL,
root tương đối là `{20,50,100}_client/`. Luôn phân giải bằng sentinel (tìm thư mục chứa
`train/client_id=000/`), đừng hard-code tiền tố mount.

---

## 6. Caveat bắt buộc kèm mọi con số công bố từ bộ dữ liệu này

Kế thừa từ `ARCHITECTURE.md` §8, cộng thêm ba điểm riêng của bản FL:

1. Split theo **thời gian mô phỏng**, không theo xe — 64 điểm cắt, mỗi (class × scenario) một điểm.
2. Lớp `benign` lấy từ luồng **không có tấn công** → nhóm đặc trưng `rate` mạnh bất thường.
3. **3.338.358 dòng nhập nhằng** (bị thao túng nhưng không gắn cờ) đã bị loại từ nguồn.
4. **Rò rỉ Sybil:** 100% dòng `trafficCongestionSybil` nằm trong flow mà mọi dòng cùng nhãn.
5. Mất cân bằng **41:1** trên cả hai tập — train `trafficCongestionSybil` 9.573.355 ÷
   `suddenConstantSpeed` 231.028 = 41,4; test 2.393.335 ÷ 57.757 = 41,4 → đọc `f1_macro`,
   không đọc `accuracy`.
6. **Client = receiver unit**, không phải một xe hay một RSU thật. Phân hoạch là mô phỏng lại
   non-IID, không phải phân bố triển khai thật.
7. **`scaler.json` fit trên toàn bộ 43 M dòng train** — tức trên dữ liệu của **mọi** client gộp
   lại. Trong FL thật client không truy cập được thống kê toàn cục. Đây là rò rỉ thông tin phải
   công bố; nó có trước bản FL này (dữ liệu được giao ở trạng thái đã chuẩn hoá).
8. **Test không được chia theo client.** Mọi client đánh giá trên cùng 10.761.343 dòng, trong đó
   chỉ có scenario `_7`. Điểm test vì thế đo **khả năng tổng quát hoá toàn cục** của mô hình cá thể
   hoá, không đo hiệu năng trên phân bố cục bộ của chính client đó.
