# DAGSNet — kiến trúc, hợp đồng vào/ra, và mô hình đã huấn luyện

**Mục đích của file này.** Đây là đặc tả đầy đủ của bộ phân loại DAGSNet dùng cho nghiên cứu:
đủ để dựng lại mô hình, nạp trọng số đã huấn luyện, và chạy suy luận **trong một project khác**
mà không cần mở lại notebook Kaggle hay đọc `report.md`. Mọi con số ở đây đọc thẳng từ artifact,
không chép tay.

| | |
|---|---|
| Nguồn kiến trúc | Khan et al. 2025, *Scientific Reports*, DOI `10.1038/s41598-025-94445-9`, §4.10, Eq. (38)–(48) |
| Bản dựng | build 4 — DAGSNet **trần**, không có §4.8 (DWT→ViT→GAT) và không có §4.9 (CMSO) |
| Dữ liệu | VeReMi NextGen, 16 lớp, 43.045.415 dòng train / 10.761.343 dòng test |
| Tham số | **395.024** learnable + 3.295 buffer (BatchNorm) |
| Checkpoint dùng | `checkpoints/ckpt_round_005.pt` — round đỉnh |
| Kết quả | `f1_macro` **0,853199**, accuracy **0,869709** trên đủ 10.761.343 dòng test |
| Thư mục gốc | `papers/build4-dagsnet-only/runs/edl_cmso_v4_dagsnet/` |

⚠ **Round 5 được chọn theo điểm trên tập test.** Đây là lựa chọn hậu nghiệm: không có tập
validation riêng, nên `f1_macro` 0,853199 là ước lượng **lạc quan** cho dữ liệu chưa thấy. Nếu
cần một con số không dính lựa chọn hậu nghiệm thì dùng round cuối (`last.pt`, round 49,
`f1_macro` 0,836477) — LR cosine đã về 0 ở đó nên nó là điểm hội tụ thật. Chênh lệch giữa hai
lựa chọn là 0,0167.

---

## 1. Hợp đồng vào/ra — phần bắt buộc đọc trước khi dùng lại

Sai ở mục này thì mô hình vẫn chạy và vẫn ra 16 số, chỉ là ra sai. Bốn ràng buộc:

### 1.1 Đầu vào: vector 66 chiều, đúng thứ tự, đã chuẩn hoá

```
x : (B, 66)  float32   — 66 cột đặc trưng, ĐÚNG thứ tự ở mục 5
```

Một cột là đặc trưng của mô hình **khi và chỉ khi** tên nó bắt đầu bằng `f_`. Bộ dữ liệu có
21 cột khác (`session`, `label`, metadata…) — chúng **không** được đưa vào mô hình.

**Chuẩn hoá bắt buộc:** z-score với `mean` và `std_used` đã fit **chỉ trên 43.045.415 dòng
train**, phương sai tổng thể (`ddof = 0`):

```python
x = (raw - mean) / std_used          # theo từng cột, giá trị ở mục 5
```

Giá trị non-finite đã được điền `0.0` từ nguồn, trước khi cắt train/test. Cột
`f_first_in_session` bằng 1 đúng ở những dòng bị điền, nên phép điền vẫn nhìn thấy được đối với
mô hình.

⚠ **Trong bộ dữ liệu gốc, `train/` đã chuẩn hoá sẵn còn `test/` thì chưa.** Đừng chuẩn hoá lại
`train/`, và đừng bao giờ fit bất kỳ thống kê nào trên `test/`.

### 1.2 Phép đổi hình dạng: (B, 66) → (B, 6, 11)

DAGSNet là mạng 1-D, nhận `(B, C, L)` — C kênh trên L vị trí. Vector 66 chiều được cắt thành
11 patch, mỗi patch 6 đặc trưng, rồi **hoán vị**:

```python
Fm = x.view(B, 11, 6).transpose(1, 2)      # (B, 6, 11): 6 kênh, 11 vị trí
```

Thứ tự hai bước này quan trọng: `view(B, 11, 6)` gom **6 cột liên tiếp** thành một patch,
`transpose` mới biến patch thành trục vị trí. Làm `view(B, 6, 11)` trực tiếp sẽ trộn sai các
cột và mô hình vẫn chạy — đây là lỗi im lặng nguy hiểm nhất khi tái sử dụng.

Vì sao 11 vị trí: đây đúng là lưới 11 patch mà `PatchEmbed` của §4.8 cắt đầu ra DWT thành, nay
đưa vào thô. Hai layout khác đã đo và loại: `(B, 1, 66)` chậm hơn pipeline đầy đủ 1,3× vì mọi
convolution phải chạy trên 66 vị trí thay vì 11; `(B, 66, 1)` làm mọi convolution suy biến
thành 1×1, biến DAGSNet thành một MLP.

### 1.3 Đầu ra: 16 logit, chưa softmax

```
logits : (B, 16)  float32     — Eq. (48), CHƯA qua softmax
```

Mô hình trả **logit**, không phải xác suất. Hàm loss lúc huấn luyện là
`nn.CrossEntropyLoss` (đã bao gồm log-softmax bên trong), nên head không có softmax.

```python
probs = torch.softmax(logits, dim=1)      # nếu cần xác suất
y_pred = logits.argmax(dim=1)             # nhãn dự đoán
```

### 1.4 Chỉ số lớp → tên lớp

Trục lớp bị **đóng băng** theo thứ tự dưới đây. Nếu project khác dùng thứ tự lớp khác thì mọi
metric per-class sẽ sai lệch mà không có cảnh báo nào.

| idx | lớp | support (test) | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | `benign` | 2.391.136 | 0.7201 | 0.8803 | 0.7922 |
| 1 | `accelerationMultiplication` | 157.490 | 0.9848 | 0.9736 | 0.9791 |
| 2 | `constantPositionOffset` | 442.475 | 0.8982 | 0.8933 | 0.8957 |
| 3 | `constantSpeedOffset` | 433.177 | 0.9433 | 0.9251 | 0.9341 |
| 4 | `dataReplay` | 475.410 | 0.7811 | 0.7235 | 0.7512 |
| 5 | `dosAttack` | 1.574.090 | 0.9950 | 0.9929 | 0.9939 |
| 6 | `feignedBraking` | 118.605 | 0.9683 | 0.9710 | 0.9696 |
| 7 | `positionMirroring` | 467.049 | 0.6669 | 0.5773 | 0.6189 |
| 8 | `randomPositionOffset` | 470.279 | 0.9688 | 0.9734 | 0.9711 |
| 9 | `randomSpeedOffset` | 542.163 | 0.9658 | 0.9697 | 0.9677 |
| 10 | `reversedHeading` | 269.999 | 0.9298 | 0.9656 | 0.9474 |
| 11 | `suddenConstantSpeed` | 57.757 | 0.9052 | 0.6761 | 0.7740 |
| 12 | `suddenStop` | 211.445 | 0.9759 | 0.8708 | 0.9204 |
| 13 | `timeDelayAttack` | 490.574 | 0.3530 | 0.1684 | 0.2281 |
| 14 | `trafficCongestionSybil` | 2.393.335 | 0.9835 | 0.9194 | 0.9504 |
| 15 | `zeroSpeedReport` | 266.359 | 0.9297 | 0.9868 | 0.9574 |
Support và F1 ở bảng trên là của **round 5**, đo trên đủ 10.761.343 dòng test.

---

## 2. Kiến trúc theo bài báo — Eq. (38)–(48)

Bốn backbone chạy **song song** trên cùng một feature map, đầu ra được nối lại rồi đưa qua một
head fully-connected. Đây là cấu trúc bài báo quy định; phần nào bài báo bỏ trống thì ghi rõ ở
mục 2.6.

```
x (B, 66)
  │  view(B,11,6).transpose(1,2)
  ▼
Fm (B, 6, 11)
  ├──► stem₀ Conv1d 1×1 → BN → ReLU ──► (B, 96, 11) ──► DenseNet1d    ──► (B, 192, 11)
  ├──► stem₁ ─────────────────────────► (B, 96, 11) ──► GoogleNet1d   ──► (B, 128, 11)
  ├──► stem₂ ─────────────────────────► (B, 96, 11) ──► AlexNet1d     ──► (B, 128,  3)
  └──► stem₃ ─────────────────────────► (B, 96, 11) ──► SqueezeNet1d  ──► (B,  96, 11)
                                                              │
                        global average pool trên trục vị trí ◄─┘
                                                              ▼
                       concat → (B, 544)   ◄── Eq. (47)
                                                              ▼
       LayerNorm → Dropout(0,1) → Linear 544→256 → ReLU → Dropout(0,1) → Linear 256→16
                                                              ▼
                                                    logits (B, 16)   ◄── Eq. (48)
```

### 2.1 Khối chung `cbr` — Conv → BatchNorm → ReLU

Mọi convolution trong mô hình đều nằm trong khối này. `bias=False` ở Conv1d vì BatchNorm ngay
sau đó đã có tham số dịch riêng — thêm bias là thừa và không đổi hàm số.

```python
def cbr(i, o, k):
    return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2, bias=False),
                         nn.BatchNorm1d(o), nn.ReLU(inplace=True))
```

`padding = k // 2` giữ nguyên chiều dài vị trí ở mọi kernel lẻ, nên trục 11 vị trí không bị co
lại ở ba nhánh Dense/Google/Squeeze.

### 2.2 DenseNet — Eq. (38)–(39)

Mỗi lớp nhận **toàn bộ** feature map của các lớp trước, nối theo trục kênh:

$$F_l = H_l([F_0, F_1, \dots, F_{l-1}]) \qquad (38)$$
$$F_{\text{DenseNet}} = F_L \qquad (39)$$

3 lớp, growth 32: `96 → 128 → 160 → 192` kênh. Đầu ra **192 kênh × 11 vị trí**.

### 2.3 GoogleNet — Eq. (40)–(41)

Inception module: bốn nhánh song song 1×1 / 3×3 / 5×5 / pool, mỗi nhánh 32 kênh, nối lại thành
128:

$$F_{\text{inception}} = \mathrm{Concat}(F_{1\times1}, F_{3\times3}, F_{5\times5}, F_{\text{pool}}) \qquad (40)$$
$$F_{\text{GoogleNet}} = F_{\text{inception-final}} \qquad (41)$$

2 module xếp chồng. Nhánh 3×3 và 5×5 đều có một 1×1 giảm chiều đứng trước, đúng kiểu Inception
gốc. Đầu ra **128 kênh × 11 vị trí**.

### 2.4 AlexNet — Eq. (42)–(44)

$$F_l = \mathrm{ReLU}(\mathrm{Conv}(F_{l-1}) + b_l) \qquad (42)$$
$$F_{\text{pooled}} = \mathrm{MaxPool}(F_l) \qquad (43)$$
$$F_{\text{AlexNet}} = F_{\text{pooled-final}} \qquad (44)$$

Ba khối conv 3×3 128 kênh, hai lần `MaxPool1d(2)` xen giữa. Đây là **nhánh duy nhất làm co trục
vị trí**: 11 → 6 → 3.

⚠ `ceil_mode=True` ở cả hai lần pool. Trục vị trí chỉ dài 11; với `ceil_mode=False` mặc định,
11 → 5 → 2 vẫn chạy, nhưng ở bất kỳ cấu hình nào có `k` nhỏ hơn thì trục có thể bị co về 0 và
mô hình sập. Đây là điều chỉnh **bắt buộc** khi đưa một kiến trúc ảnh 2-D sang chuỗi ngắn 1-D.

### 2.5 SqueezeNet — Eq. (45)–(46)

$$F_{\text{squeeze}} = \mathrm{Conv}_{1\times1}(F_{\text{input}}) \qquad (46)$$
$$F_{\text{Fire}} = \mathrm{Concat}(\mathrm{Conv}_{1\times1}(F_{\text{squeeze}}), \mathrm{Conv}_{3\times3}(F_{\text{squeeze}})) \qquad (45)$$

Fire module: squeeze xuống 32 kênh rồi hai nhánh expand 48 kênh, nối lại thành 96. 3 module xếp
chồng, mỗi module lại squeeze 96 → 32. Đầu ra **96 kênh × 11 vị trí**.

### 2.6 Hợp nhất và head — Eq. (47)–(48)

$$F_{\text{combined}} = \mathrm{Concat}(F_{\text{DenseNet}}, F_{\text{GoogleNet}}, F_{\text{AlexNet}}, F_{\text{SqueezeNet}}) \qquad (47)$$
$$y = \sigma(W F_{\text{combined}} + b) \qquad (48)$$

Bốn nhánh có **chiều dài vị trí khác nhau** (11, 11, 3, 11) nên không nối trực tiếp được. Giải
pháp: **global average pool trên trục vị trí** cho từng nhánh trước khi nối →
`192 + 128 + 128 + 96 = 544`.

**Những chỗ bài báo không quy định, và tôi đã chọn — mọi chỗ đều là deviation phải công bố:**

| chỗ | bài báo | bản dựng này | vì sao |
|---|---|---|---|
| stem 1×1 trước mỗi nhánh | không nhắc | Conv1d 1×1 → 96 kênh, 4 bản độc lập | bốn backbone cần cùng số kênh vào; không có stem thì mỗi nhánh phải tự xử lý 6 kênh |
| cách gộp 4 nhánh khác chiều dài | chỉ nói "Concat" | global average pool rồi mới concat | 11/11/3/11 không nối trực tiếp được |
| kích thước head | không nhắc | LayerNorm → 544→256 → ReLU → 256→16 | Eq. (48) chỉ nói một phép chiếu tuyến tính |
| σ ở Eq. (48) | "σ" | **không có** — trả logit | `CrossEntropyLoss` đã chứa log-softmax; thêm softmax nữa là sai |
| dropout | có nhắc, không định lượng | 0,1, hai vị trí trong head | |
| growth / số module / số kênh | không nhắc | growth 32, dense 3, inception 2, fire 3, stem 96 | |

Bài báo phân loại **nhị phân**; Eq. (50) cho phép đa lớp và bản này dùng nhánh đó với C = 16.

---

## 3. Trước khi huấn luyện — kiến trúc ở trạng thái khởi tạo

### 3.1 Ngân sách tham số

| khối | tham số | tỉ lệ |
|---|---:|---:|
| 4 × stem 1×1 | 3.072 | 0,8% |
| DenseNet — Eq. (38)–(39) | 37.056 | 9,4% |
| GoogleNet — Eq. (40)–(41) | 45.824 | 11,6% |
| AlexNet — Eq. (42)–(44) | 135.936 | 34,4% |
| SqueezeNet — Eq. (45)–(46) | 28.416 | 7,2% |
| head — Eq. (47)–(48) | 144.720 | 36,6% |
| **tổng learnable** | **395.024** | 100% |
| buffer BatchNorm (không học) | 3.295 | — |

Hai khối chiếm 71% tham số là **AlexNet** và **head**, dù AlexNet là nhánh đơn giản nhất về mặt
cấu trúc — vì nó là nhánh duy nhất giữ 128 kênh qua ba conv 3×3 liên tiếp.

### 3.2 Khởi tạo

Không có khởi tạo tuỳ chỉnh; toàn bộ dùng **mặc định của PyTorch**:

* `Conv1d`, `Linear` — Kaiming uniform với `a = √5`, tương đương uniform trong
  `±1/√fan_in`
* `BatchNorm1d` — `weight = 1`, `bias = 0`, `running_mean = 0`, `running_var = 1`
* `LayerNorm` — `weight = 1`, `bias = 0`

Seed cố định **42**, đặt trước khi dựng mô hình. Với cùng seed và cùng phiên bản PyTorch, trọng
số khởi tạo tái lập được từng bit.

### 3.3 Bảng từng lớp — hình dạng và tham số

Cột `in`/`out` là `(C, L)`, đã bỏ chiều batch. Trục vị trí `L` giữ nguyên 11 ở mọi nhánh trừ
AlexNet.

| lớp | kiểu | in (C, L) | out (C, L) | tham số |
|---|---|---|---|---:|
| `stems.0.0` | Conv1d | (6. 11) | (96. 11) | 576 |
| `stems.0.1` | BatchNorm1d | (96. 11) | (96. 11) | 192 |
| `dense.blocks.0.0` | Conv1d | (96. 11) | (32. 11) | 9.216 |
| `dense.blocks.0.1` | BatchNorm1d | (32. 11) | (32. 11) | 64 |
| `dense.blocks.1.0` | Conv1d | (128. 11) | (32. 11) | 12.288 |
| `dense.blocks.1.1` | BatchNorm1d | (32. 11) | (32. 11) | 64 |
| `dense.blocks.2.0` | Conv1d | (160. 11) | (32. 11) | 15.360 |
| `dense.blocks.2.1` | BatchNorm1d | (32. 11) | (32. 11) | 64 |
| `stems.1.0` | Conv1d | (6. 11) | (96. 11) | 576 |
| `stems.1.1` | BatchNorm1d | (96. 11) | (96. 11) | 192 |
| `google.net.0.b1.0` | Conv1d | (96. 11) | (32. 11) | 3.072 |
| `google.net.0.b1.1` | BatchNorm1d | (32. 11) | (32. 11) | 64 |
| `google.net.0.b3.0.0` | Conv1d | (96. 11) | (32. 11) | 3.072 |
| `google.net.0.b3.0.1` | BatchNorm1d | (32. 11) | (32. 11) | 64 |
| `google.net.0.b3.1.0` | Conv1d | (32. 11) | (32. 11) | 3.072 |
| `google.net.0.b3.1.1` | BatchNorm1d | (32. 11) | (32. 11) | 64 |
| `google.net.0.b5.0.0` | Conv1d | (96. 11) | (32. 11) | 3.072 |
| `google.net.0.b5.0.1` | BatchNorm1d | (32. 11) | (32. 11) | 64 |
| `google.net.0.b5.1.0` | Conv1d | (32. 11) | (32. 11) | 5.120 |
| `google.net.0.b5.1.1` | BatchNorm1d | (32. 11) | (32. 11) | 64 |
| `google.net.0.bp.0` | MaxPool1d | (96. 11) | (96. 11) | 0 |
| `google.net.0.bp.1.0` | Conv1d | (96. 11) | (32. 11) | 3.072 |
| `google.net.0.bp.1.1` | BatchNorm1d | (32. 11) | (32. 11) | 64 |
| `google.net.1.b1.0` | Conv1d | (128. 11) | (32. 11) | 4.096 |
| `google.net.1.b1.1` | BatchNorm1d | (32. 11) | (32. 11) | 64 |
| `google.net.1.b3.0.0` | Conv1d | (128. 11) | (32. 11) | 4.096 |
| `google.net.1.b3.0.1` | BatchNorm1d | (32. 11) | (32. 11) | 64 |
| `google.net.1.b3.1.0` | Conv1d | (32. 11) | (32. 11) | 3.072 |
| `google.net.1.b3.1.1` | BatchNorm1d | (32. 11) | (32. 11) | 64 |
| `google.net.1.b5.0.0` | Conv1d | (128. 11) | (32. 11) | 4.096 |
| `google.net.1.b5.0.1` | BatchNorm1d | (32. 11) | (32. 11) | 64 |
| `google.net.1.b5.1.0` | Conv1d | (32. 11) | (32. 11) | 5.120 |
| `google.net.1.b5.1.1` | BatchNorm1d | (32. 11) | (32. 11) | 64 |
| `google.net.1.bp.0` | MaxPool1d | (128. 11) | (128. 11) | 0 |
| `google.net.1.bp.1.0` | Conv1d | (128. 11) | (32. 11) | 4.096 |
| `google.net.1.bp.1.1` | BatchNorm1d | (32. 11) | (32. 11) | 64 |
| `stems.2.0` | Conv1d | (6. 11) | (96. 11) | 576 |
| `stems.2.1` | BatchNorm1d | (96. 11) | (96. 11) | 192 |
| `alex.net.0.0` | Conv1d | (96. 11) | (128. 11) | 36.864 |
| `alex.net.0.1` | BatchNorm1d | (128. 11) | (128. 11) | 256 |
| `alex.net.1` | MaxPool1d | (128. 11) | (128. 6) | 0 |
| `alex.net.2.0` | Conv1d | (128. 6) | (128. 6) | 49.152 |
| `alex.net.2.1` | BatchNorm1d | (128. 6) | (128. 6) | 256 |
| `alex.net.3` | MaxPool1d | (128. 6) | (128. 3) | 0 |
| `alex.net.4.0` | Conv1d | (128. 3) | (128. 3) | 49.152 |
| `alex.net.4.1` | BatchNorm1d | (128. 3) | (128. 3) | 256 |
| `stems.3.0` | Conv1d | (6. 11) | (96. 11) | 576 |
| `stems.3.1` | BatchNorm1d | (96. 11) | (96. 11) | 192 |
| `squeeze.net.0.squeeze.0` | Conv1d | (96. 11) | (32. 11) | 3.072 |
| `squeeze.net.0.squeeze.1` | BatchNorm1d | (32. 11) | (32. 11) | 64 |
| `squeeze.net.0.e1.0` | Conv1d | (32. 11) | (48. 11) | 1.536 |
| `squeeze.net.0.e1.1` | BatchNorm1d | (48. 11) | (48. 11) | 96 |
| `squeeze.net.0.e3.0` | Conv1d | (32. 11) | (48. 11) | 4.608 |
| `squeeze.net.0.e3.1` | BatchNorm1d | (48. 11) | (48. 11) | 96 |
| `squeeze.net.1.squeeze.0` | Conv1d | (96. 11) | (32. 11) | 3.072 |
| `squeeze.net.1.squeeze.1` | BatchNorm1d | (32. 11) | (32. 11) | 64 |
| `squeeze.net.1.e1.0` | Conv1d | (32. 11) | (48. 11) | 1.536 |
| `squeeze.net.1.e1.1` | BatchNorm1d | (48. 11) | (48. 11) | 96 |
| `squeeze.net.1.e3.0` | Conv1d | (32. 11) | (48. 11) | 4.608 |
| `squeeze.net.1.e3.1` | BatchNorm1d | (48. 11) | (48. 11) | 96 |
| `squeeze.net.2.squeeze.0` | Conv1d | (96. 11) | (32. 11) | 3.072 |
| `squeeze.net.2.squeeze.1` | BatchNorm1d | (32. 11) | (32. 11) | 64 |
| `squeeze.net.2.e1.0` | Conv1d | (32. 11) | (48. 11) | 1.536 |
| `squeeze.net.2.e1.1` | BatchNorm1d | (48. 11) | (48. 11) | 96 |
| `squeeze.net.2.e3.0` | Conv1d | (32. 11) | (48. 11) | 4.608 |
| `squeeze.net.2.e3.1` | BatchNorm1d | (48. 11) | (48. 11) | 96 |
| `head.0` | LayerNorm | (544.) | (544.) | 1.088 |
| `head.2` | Linear | (544.) | (256.) | 139.520 |
| `head.5` | Linear | (256.) | (16.) | 4.112 |
---

## 4. Sau khi huấn luyện — mô hình đã học

### 4.1 Cấu hình huấn luyện đã dùng

| | |
|---|---|
| Round × epoch | 50 round × 1 epoch = 50 lượt qua toàn bộ train |
| Batch | 2.048/GPU × 2 GPU × 1 = **4.096** global |
| Optimizer | `torch.optim.AdamW`, `lr = 1e-3`, `weight_decay = 1e-4` |
| Adam betas / eps / amsgrad | `(0.9, 0.999)` / `1e-8` / `False` (mặc định PyTorch) |
| Lịch LR | `LambdaLR`: warmup 1 round rồi cosine, về 0 ở round 49 |
| Loss | `nn.CrossEntropyLoss()` trên nhãn số nguyên |
| Precision | fp16 AMP + `GradScaler`; loss tính ở **fp32 ngoài autocast** |
| Clip | grad-norm 1,0 |
| Seed | 42 |
| Compile | `torch.compile` bật |
| Phần cứng | 2 × Tesla T4 (Kaggle), DDP |
| Thời gian | 4,61 h cho 50 round, 331,8 s/round, 129.764 mẫu/s |

Round 5 đạt sau **5 epoch**, tức khoảng 27,7 phút huấn luyện.

### 4.1.1 Đối chiếu với Table 1 của bài báo

Bài báo quy định một số hạng mục huấn luyện. Bảng dưới đối chiếu **từng hạng mục** với những
gì bản dựng này thật sự làm. Batch size và số epoch **cố tình không** đưa vào đây: bài báo ghi
batch 64 và 100 epoch, bản này dùng 4.096 và 50 — đó là deviation về ngân sách tính toán, đã
ghi riêng ở `report.md`, không phải hạng mục có thể tuyên bố là tuân thủ.

| Hạng mục bài báo nêu | Bản build 4 thực tế làm | Khớp? |
|---|---|---|
| **Learning rate 0,001, adaptive decay** | `lr = 1e-3`. "Adaptive decay" hiện diện ở **hai tầng**: (a) AdamW tự điều chỉnh bước cho từng tham số qua moment bậc 1 và 2; (b) lịch cosine giảm hệ số LR từ 1,0 về 0 qua 49 round. | ✔ giá trị khớp; **dạng** decay là tôi chọn |
| **Loss: Categorical Cross-Entropy** | `nn.CrossEntropyLoss()`. PyTorch nhận nhãn số nguyên và tự làm log-softmax bên trong — **tương đương toán học** với categorical cross-entropy trên nhãn one-hot, chỉ khác cách biểu diễn nhãn. | ✔ |
| **ReLU cho hidden layer** | 32 lớp `nn.ReLU`: một lớp sau mỗi `Conv1d + BatchNorm1d` trong cả bốn backbone, cộng một lớp giữa hai `Linear` của head. | ✔ |
| **Softmax cho output layer** | ⚠ Module **không** chứa `nn.Softmax` (đếm được: 0). Softmax **có** được áp dụng, nhưng ở hai chỗ khác: trong `CrossEntropyLoss` lúc huấn luyện, và gọi tường minh `torch.softmax(logits, 1)` khi cần xác suất lúc suy luận. Đặt thêm một `nn.Softmax` trong `forward` sẽ khiến loss áp softmax **hai lần** và làm hỏng gradient. | ✔ về mặt hàm số; khác về vị trí đặt |
| **Batch normalization** | 31 lớp `nn.BatchNorm1d`, một lớp sau mỗi convolution (khối `cbr`). Đây cũng là lý do mọi `Conv1d` đặt `bias=False`. | ✔ |
| **Dropout** | 2 lớp `nn.Dropout(p = 0,1)`, cả hai trong head: trước `Linear 544→256` và trước `Linear 256→16`. Bài báo có nhắc dropout nhưng **không định lượng**, nên 0,1 là tôi chọn. | ✔ có dùng; tỉ lệ là tôi chọn |
| **Adam để tăng tốc hội tụ** | `AdamW` — chính là Adam với weight decay tách rời (`decoupled_weight_decay = True`). Bài báo **không nêu** weight decay, nên 1e-4 là tôi chọn. | ✔ |
| **Pruning để nén trọng số** | ❌ **KHÔNG làm.** | ✘ |
| **Quantization để nén trọng số** | ❌ **KHÔNG làm.** Checkpoint là fp32 thuần. fp16 chỉ được dùng làm precision tính toán lúc huấn luyện (AMP), **không** phải quantization trọng số. | ✘ |

⚠ **Hai dòng cuối là deviation phải công bố kèm mọi con số.** Bài báo nêu pruning và
quantization như một phần của phương pháp nhưng **không định lượng** chúng — không có tỉ lệ
pruning, không có bit-width, không có bảng trước/sau. Vì vậy không có gì để tái lập, và bản
dựng này bỏ hẳn hai bước đó. Hệ quả thực tế: **mọi con số kích thước và tốc độ trong tài liệu
này là của mô hình chưa nén.** Đo được: `state_dict` có **161 tensor fp32** và 31 tensor int64
(`num_batches_tracked` của BatchNorm) — **không có tensor nào ở int8 hay fp16**, tức không hề
có quantization. 395.024 tham số fp32 = **1,507 MiB** trọng số; file checkpoint 4,70 MiB vì
còn chứa cả trạng thái optimizer.

Nếu sau này áp pruning/quantization thì **phải đo lại toàn bộ 10 metric**: nén trọng số thay
đổi kết quả chứ không chỉ thay đổi kích thước, và mọi con số ở mục 4.4 sẽ không còn mô tả mô
hình đã nén nữa.

⚠ **fp16 trong bảng 4.1 không phải quantization.** AMP dùng fp16 làm precision *tính toán*
trong lúc huấn luyện; trọng số chính (master weights) vẫn là fp32 và checkpoint lưu fp32. Hai
khái niệm này hay bị lẫn khi đối chiếu với câu "quantization" của bài báo.

### 4.2 Checkpoint chứa gì

`checkpoints/ckpt_round_005.pt` — `torch.save` một dict:

| khoá | nội dung |
|---|---|
| `round` | `5` |
| `model` | `state_dict`, **192 tensor**, không có tiền tố `module.` (đã gỡ DDP khi lưu) |
| `optim` | trạng thái Adam |
| `scaler` | trạng thái `GradScaler` |
| `sched` | trạng thái scheduler |
| `rng` | RNG của python / numpy / torch / cuda |
| `cfg` | 39 khoá cấu hình đã dùng |
| `fingerprint` | `b3d68bbddc06a00d` — băm của các trường ảnh hưởng hình dạng |
| `metrics` | 10 metric + train_loss + lr + seconds + peak_gb |

Chỉ cần `model` để suy luận. `optim`/`sched`/`rng` chỉ cần khi muốn huấn luyện tiếp.

**`fingerprint` là chốt an toàn:** nó băm `num_classes`, `n_features`, `batch_per_gpu`,
`run_name`. Nạp checkpoint vào một mô hình có hình dạng khác sẽ bị chặn thay vì im lặng chạy sai.

⚠ Bốn khoá trên là của **checkpoint DAGSNet cũ này**, không phải một hợp đồng chung. Bộ khoá đó
chỉ bao hình dạng: nó **mù** với `lr`, `mu`, `beta`, `lam`, `T`, `seed`, `local_epochs`,
`dropout` và toàn bộ preprocessing, nên hai run khác hẳn nhau về thuật toán vẫn cùng
fingerprint và vẫn resume chồng lên nhau. Một phương pháp mới phải tự định nghĩa bộ khoá của
mình — ví dụ `papers/fd-ids-2025/proj/ckpt.py::FINGERPRINT_KEYS`.

### 4.3 Trọng số đã học so với lúc khởi tạo

| khối | tham số | mean khởi tạo | std khởi tạo | mean r5 | std r5 | \|w\|max r5 |
|---|---:|---:|---:|---:|---:|---:|
| 4 × stem 1×1 | 3.072 | +0,12138 | 0,38887 | +0,06860 | 0,54380 | 7,6191 |
| DenseNet | 37.056 | +0,00267 | 0,05872 | −0,03699 | 0,26020 | 3,8712 |
| GoogleNet | 45.824 | +0,00873 | 0,10535 | −0,02053 | 0,27064 | 2,2689 |
| AlexNet | 135.936 | +0,00279 | 0,06132 | −0,02034 | 0,25814 | 4,8837 |
| SqueezeNet | 28.416 | +0,01338 | 0,13362 | −0,01372 | 0,28291 | 2,2191 |
| head | 144.720 | +0,00369 | 0,06613 | +0,00171 | 0,23874 | 2,6690 |

Ba điều đọc được:

1. **Độ lệch chuẩn tăng 2–4× ở mọi khối** — trọng số nở ra khỏi vùng khởi tạo hẹp, đúng như một
   mô hình thực sự học chứ không đứng yên.
2. **Mean chuyển từ dương nhẹ sang âm nhẹ ở cả bốn backbone.** Đây là dấu hiệu của ReLU +
   BatchNorm: một phần trọng số bị đẩy âm để tắt các kênh không hữu ích.
3. **`|w|max` toàn mô hình là 7,62 và nằm ở stem** — nhỏ, hữu hạn, không có dấu hiệu bất ổn.
   Để so sánh: run pipeline đầy đủ khi phân kỳ có `‖W_qkv‖` lên 526. DAGSNet trần không có khối
   attention nên không có cơ chế hỏng đó.

**Buffer BatchNorm** — thuần tuý là sản phẩm của huấn luyện, khởi tạo là 0 và 1:

| | khởi tạo | sau round 5 |
|---|---|---|
| `running_mean` | 0 | −9,830 … +7,630 |
| `running_var` | 1 | 0,0762 … 138,408 |

⚠ Phải nạp cả buffer, không chỉ trọng số. `load_state_dict` mặc định làm điều này; nhưng nếu
tự lọc key thì lọc mất `running_mean`/`running_var` sẽ khiến mô hình ở chế độ `eval()` chạy
chuẩn hoá bằng 0/1 và kết quả sai hoàn toàn mà không có lỗi nào.

⚠ **Luôn gọi `model.eval()` trước khi suy luận.** Mô hình có 31 lớp BatchNorm và 2 lớp Dropout;
ở chế độ `train()` chúng dùng thống kê batch và bật dropout, nên cùng một đầu vào sẽ ra kết quả
khác nhau giữa các lần chạy.

### 4.4 Kết quả đo được — đủ 10 metric, round 5

Trên **đủ 10.761.343 dòng test**, không lấy mẫu:

| metric | giá trị | | metric | giá trị |
|---|---:|---|---|---:|
| accuracy | 0,869709 | | **f1_macro** | **0,853199** |
| precision_macro | 0,873081 | | f1_micro | 0,869709 |
| precision_micro | 0,869709 | | f1_weighted | 0,863558 |
| precision_weighted | 0,865171 | | recall_macro | 0,841692 |
| recall_micro | 0,869709 | | recall_weighted | 0,869709 |

`precision_micro = recall_micro = f1_micro = recall_weighted = accuracy` là **đẳng thức** của
phân loại đơn nhãn đa lớp, không phải trùng lặp do lỗi.

Với mất cân bằng 41:1 thì **`f1_macro` mới là con số đáng đọc**; `accuracy` gần như vô nghĩa vì
hai lớp lớn nhất (`trafficCongestionSybil`, `benign`) đã chiếm 44,5% tập test.

### 4.5 Mô hình sai ở đâu — phải nêu kèm mọi con số trên

Lỗi **không** rải đều. Từ confusion chuẩn hoá theo hàng ở round 5:

| lớp thật | bị đoán nhầm thành | tỉ lệ |
|---|---|---:|
| `timeDelayAttack` | `benign` | **76%** |
| `positionMirroring` | `benign` | 34% |
| `suddenConstantSpeed` | `zeroSpeedReport` | 29% |
| `dataReplay` | `benign` | 23% |

Phần lớn sai sót là **không phát hiện được tấn công**, chứ không phải nhầm giữa các loại tấn
công — đúng kiểu lỗi tốn kém nhất trong một IDS. `timeDelayAttack` là lớp yếu nhất với F1 chỉ
0,2281 và recall 0,17.

---

## 5. Bảng 66 cột đặc trưng và tham số chuẩn hoá

Thứ tự dưới đây **là** thứ tự chiều của vector đầu vào. `mean` và `std_used` fit trên
43.045.415 dòng train, `ddof = 0`.

| # | cột | mean (train) | std_used |
|---:|---|---:|---:|
| 0 | `f_rcv_pos_noise_x` | -0.32229084 | 2.7170261 |
| 1 | `f_rcv_pos_noise_y` | 0.052102859 | 2.8360266 |
| 2 | `f_rcv_spd` | 7.5736565 | 9.8856121 |
| 3 | `f_rcv_spd_noise` | 7.1369936e-05 | 0.002071985 |
| 4 | `f_rcv_acl` | -0.043287912 | 1.17559 |
| 5 | `f_rcv_acl_noise` | -5.3480514e-08 | 0.00042314372 |
| 6 | `f_rcv_hed_noise` | -0.083993565 | 8.4686711 |
| 7 | `f_snd_pos_noise_x` | -0.1286987 | 2.782898 |
| 8 | `f_snd_pos_noise_y` | -0.22331546 | 2.9029515 |
| 9 | `f_snd_spd` | 7.2137361 | 9.9867292 |
| 10 | `f_snd_spd_noise` | 6.5251508e-05 | 0.0020403534 |
| 11 | `f_snd_acl` | -0.057533846 | 1.3625369 |
| 12 | `f_snd_acl_noise` | 3.3384239e-07 | 9.6364593e-05 |
| 13 | `f_snd_hed_noise` | 0.027575088 | 8.3364422 |
| 14 | `f_snd_dist_road_edge` | 1.4073348 | 9.526795 |
| 15 | `f_rcv_x_rel` | 1132.7659 | 858.00274 |
| 16 | `f_rcv_y_rel` | 2957.0531 | 1489.3207 |
| 17 | `f_snd_x_rel` | 1131.1197 | 856.0188 |
| 18 | `f_snd_y_rel` | 2958.1642 | 1487.6272 |
| 19 | `f_delay_s` | 0.001563238 | 0.0004321213 |
| 20 | `f_dx` | -1.6461433 | 101.21425 |
| 21 | `f_dy` | 1.1110922 | 148.0079 |
| 22 | `f_dist` | 154.36084 | 91.254248 |
| 23 | `f_bearing_sin` | -0.005745635 | 0.58016916 |
| 24 | `f_bearing_cos` | 0.00019984039 | 0.81447572 |
| 25 | `f_rcv_hed_sin` | -0.0069410829 | 0.58890351 |
| 26 | `f_rcv_hed_cos` | -0.034821813 | 0.80742301 |
| 27 | `f_snd_hed_sin` | -0.017015714 | 0.58687298 |
| 28 | `f_snd_hed_cos` | -0.029387493 | 0.80896659 |
| 29 | `f_hed_diff_cos` | 0.17828701 | 0.79824534 |
| 30 | `f_rcv_vx` | 0.0052198544 | 6.3576587 |
| 31 | `f_rcv_vy` | -0.51750925 | 10.695697 |
| 32 | `f_snd_vx` | -0.00087917273 | 6.2201021 |
| 33 | `f_snd_vy` | -0.38215771 | 10.627184 |
| 34 | `f_rel_speed` | 10.998055 | 13.284657 |
| 35 | `f_closing_speed` | 0.30107887 | 14.415484 |
| 36 | `f_spd_diff` | -0.35992034 | 12.223825 |
| 37 | `f_rcv_noise_mag` | 3.6981505 | 1.3621844 |
| 38 | `f_snd_noise_mag` | 3.7884973 | 1.3730877 |
| 39 | `f_first_in_session` | 0.24089553 | 0.42762703 |
| 40 | `f_sess_idx` | 54.580318 | 100.50522 |
| 41 | `f_sess_dt` | 0.70993388 | 1.3999639 |
| 42 | `f_sess_dt_send` | 0.70993384 | 1.3999635 |
| 43 | `f_sess_dt_skew` | 4.4553515e-08 | 0.00042355531 |
| 44 | `f_sess_dpos` | 10.116826 | 30.873939 |
| 45 | `f_sess_implied_spd` | 9.884142 | 29.011265 |
| 46 | `f_sess_spd_residual` | 4.5464952 | 27.72174 |
| 47 | `f_sess_dspd` | 0.0012256515 | 2.0020653 |
| 48 | `f_sess_acl_residual` | 0.050294939 | 1.9460357 |
| 49 | `f_sess_dhed` | 0.18283056 | 10.08776 |
| 50 | `f_sess_dmsgid` | 2386.8923 | 619913.81 |
| 51 | `f_sess_ddist` | -0.1996557 | 23.870133 |
| 52 | `f_sess_ddre` | 0.0023694589 | 5.8847587 |
| 53 | `f_sess_pos_pred_err` | 7.4354716 | 32.107792 |
| 54 | `f_alias_age_s` | 39.614839 | 55.747629 |
| 55 | `f_rx_rate_1s` | 75.116894 | 49.416556 |
| 56 | `f_rx_rate_5s` | 369.56792 | 247.46265 |
| 57 | `f_sender_rate_1s` | 1.6916893 | 1.0020296 |
| 58 | `f_sender_rate_5s` | 5.6860442 | 4.8419973 |
| 59 | `f_sender_share_5s` | 0.04223301 | 0.10781099 |
| 60 | `f_rcv_profile_normal` | 0.82807251 | 0.37731741 |
| 61 | `f_rcv_profile_cautious` | 0.098501432 | 0.29799144 |
| 62 | `f_rcv_profile_aggressive` | 0.073426055 | 0.26083456 |
| 63 | `f_snd_profile_normal` | 0.83610682 | 0.37017861 |
| 64 | `f_snd_profile_cautious` | 0.097530248 | 0.29667844 |
| 65 | `f_snd_profile_aggressive` | 0.066362933 | 0.24891544 |
---

## 6. Mã nguồn đầy đủ để dùng lại

Chép nguyên khối này vào project mới. Nó không phụ thuộc gì ngoài `torch` và `numpy`.

```python
"""DAGSNet — Khan et al. 2025 §4.10, Eq. (38)-(48). 395.024 tham số.
Vào: (B, 66) đặc trưng đã z-score.  Ra: (B, 16) logit (CHƯA softmax)."""
import torch
import torch.nn as nn


def cbr(i, o, k):
    """Conv → BatchNorm → ReLU. bias=False vì BatchNorm ngay sau đã có tham số dịch."""
    return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2, bias=False),
                         nn.BatchNorm1d(o), nn.ReLU(inplace=True))


class DenseNet1d(nn.Module):
    """Eq. (38)-(39): mỗi lớp nhận nối của toàn bộ feature map trước đó."""
    def __init__(self, cin, growth, layers):
        super().__init__()
        self.blocks = nn.ModuleList([cbr(cin + i * growth, growth, 3) for i in range(layers)])
        self.out_ch = cin + layers * growth

    def forward(self, x):
        for b in self.blocks:
            x = torch.cat([x, b(x)], dim=1)                     # Eq. (38)
        return x                                                # Eq. (39)


class Inception1d(nn.Module):
    """Eq. (40): bốn nhánh song song 1x1 / 3x3 / 5x5 / pool, nối lại."""
    def __init__(self, cin, c):
        super().__init__()
        self.b1 = cbr(cin, c, 1)
        self.b3 = nn.Sequential(cbr(cin, c, 1), cbr(c, c, 3))
        self.b5 = nn.Sequential(cbr(cin, c, 1), cbr(c, c, 5))
        self.bp = nn.Sequential(nn.MaxPool1d(3, 1, 1), cbr(cin, c, 1))
        self.out_ch = 4 * c

    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1)


class GoogleNet1d(nn.Module):
    """Eq. (40)-(41): các inception module xếp chồng."""
    def __init__(self, cin, modules_n, c=32):
        super().__init__()
        mods, ch = [], cin
        for _ in range(modules_n):
            m = Inception1d(ch, c); mods.append(m); ch = m.out_ch
        self.net = nn.Sequential(*mods); self.out_ch = ch

    def forward(self, x):
        return self.net(x)


class AlexNet1d(nn.Module):
    """Eq. (42)-(44). ceil_mode=True: trục vị trí chỉ dài 11, không được để pool co về 0."""
    def __init__(self, cin, ch=128):
        super().__init__()
        self.net = nn.Sequential(
            cbr(cin, ch, 3), nn.MaxPool1d(2, ceil_mode=True),
            cbr(ch, ch, 3),  nn.MaxPool1d(2, ceil_mode=True),
            cbr(ch, ch, 3))
        self.out_ch = ch

    def forward(self, x):
        return self.net(x)


class Fire1d(nn.Module):
    """Eq. (45)-(46): squeeze 1x1 nuôi hai nhánh expand 1x1 và 3x3."""
    def __init__(self, cin, sq, ex):
        super().__init__()
        self.squeeze = cbr(cin, sq, 1)                          # Eq. (46)
        self.e1 = cbr(sq, ex, 1)
        self.e3 = cbr(sq, ex, 3)
        self.out_ch = 2 * ex

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.e1(s), self.e3(s)], dim=1)       # Eq. (45)


class SqueezeNet1d(nn.Module):
    def __init__(self, cin, modules_n, sq=32, ex=48):
        super().__init__()
        mods, ch = [], cin
        for _ in range(modules_n):
            m = Fire1d(ch, sq, ex); mods.append(m); ch = m.out_ch
        self.net = nn.Sequential(*mods); self.out_ch = ch

    def forward(self, x):
        return self.net(x)


class DAGSNet(nn.Module):
    def __init__(self, cfg, n_features):
        super().__init__()
        self.patch_len = cfg["patch_len"]
        assert n_features % self.patch_len == 0
        self.k = n_features // self.patch_len                   # 11
        cin = self.patch_len                                    # 6 kênh

        s = cfg["stem_ch"]
        self.stems   = nn.ModuleList([cbr(cin, s, 1) for _ in range(4)])
        self.dense   = DenseNet1d(s, cfg["dense_growth"], cfg["dense_layers"])
        self.google  = GoogleNet1d(s, cfg["incep_modules"])
        self.alex    = AlexNet1d(s)
        self.squeeze = SqueezeNet1d(s, cfg["fire_modules"])
        comb = self.dense.out_ch + self.google.out_ch + self.alex.out_ch + self.squeeze.out_ch

        self.head = nn.Sequential(                              # Eq. (48)
            nn.LayerNorm(comb), nn.Dropout(cfg["dropout"]),
            nn.Linear(comb, 256), nn.ReLU(inplace=True),
            nn.Dropout(cfg["dropout"]), nn.Linear(256, cfg["num_classes"]))

    def forward(self, x):                                       # (B, n_features)
        # view rồi MỚI transpose: gom 6 cột liên tiếp thành một patch, sau đó patch
        # mới trở thành trục vị trí. Làm view(B, 6, 11) thẳng sẽ trộn sai các cột.
        Fm = x.view(x.shape[0], self.k, self.patch_len).transpose(1, 2)   # (B, 6, 11)
        feats = [gp(stem(Fm)) for stem, gp in
                 zip(self.stems, [self.dense, self.google, self.alex, self.squeeze])]
        pooled = [f.mean(dim=-1) for f in feats]                # global average pool
        return self.head(torch.cat(pooled, dim=1))              # Eq. (47) -> (48)


# Cấu hình ĐÚNG như checkpoint round 5 đã huấn luyện. Đổi bất kỳ giá trị nào ở đây
# thì state_dict sẽ không nạp được — đó là chủ ý.
CFG = {
    "patch_len": 6, "stem_ch": 96, "dense_growth": 32, "dense_layers": 3,
    "incep_modules": 2, "fire_modules": 3, "dropout": 0.1, "num_classes": 16,
}


def load_dagsnet(ckpt_path, device="cpu"):
    """Dựng mô hình và nạp checkpoint round 5. Trả về mô hình đã ở chế độ eval."""
    model = DAGSNet(CFG, n_features=66)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    sd = {k.removeprefix("module."): v for k, v in sd.items()}   # phòng khi lưu từ DDP
    model.load_state_dict(sd, strict=True)     # strict=True: thiếu buffer BN là LỖI
    n = sum(p.numel() for p in model.parameters())
    assert n == 395_024, f"kiến trúc lệch: {n:,} tham số thay vì 395.024"
    return model.to(device).eval()             # eval(): tắt Dropout, dùng running stats
```

### 6.1 Tiền xử lý và suy luận

```python
import json
import numpy as np

FEATURES = json.load(open("meta.json"))["feature_cols"]          # 66 tên, ĐÚNG thứ tự
SCALER   = json.load(open("scaler.json"))["features"]            # mean / std_used

def preprocess(df):
    """DataFrame thô -> (N, 66) float32 đã z-score, đúng thứ tự cột."""
    X = np.empty((len(df), len(FEATURES)), dtype=np.float32)
    for i, name in enumerate(FEATURES):
        s = SCALER[name]
        v = np.asarray(df[name], dtype=np.float64)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)    # điền như lúc dựng dữ liệu
        X[:, i] = (v - s["mean"]) / s["std_used"]
    return X


@torch.no_grad()
def predict(model, X, batch=16384, device="cpu"):
    """Trả (y_pred, probs). X là đầu ra của preprocess()."""
    preds, probs = [], []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        logits = model(xb)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


model = load_dagsnet("ckpt_round_005.pt", device="cuda")
y_pred, probs = predict(model, preprocess(df), device="cuda")
```

Ba file cần mang sang project mới:

| file | ở đâu | dùng để |
|---|---|---|
| `ckpt_round_005.pt` | `runs/edl_cmso_v4_dagsnet/checkpoints/` | trọng số |
| `meta.json` | `runs/edl_cmso_v4_dagsnet/` | `feature_cols`, `class_names` |
| `scaler.json` | `papers/build4-dagsnet-only/model/` | `mean`, `std_used` |

---

## 7. Những gì đã kiểm chứng cho tài liệu này

Mọi khẳng định ở trên đều đã chạy lại, không có số nào chép tay:

| kiểm tra | kết quả |
|---|---|
| `load_state_dict(strict=True)` với checkpoint round 5 | nạp được, không thiếu/thừa key |
| Số tham số | **395.024** learnable, khớp con số công bố; buffer 3.295 |
| Forward `(4, 66)` | ra `(4, 16)` |
| Tổng tham số theo khối | 3.072 + 37.056 + 45.824 + 135.936 + 28.416 + 144.720 = **395.024** ✓ |
| Confusion round 5 | cộng đúng **10.761.343** dòng |
| Cột confusion | bằng `bincount(y_pred)` ✓ |
| Hàng confusion | bằng `test_counts` trong `meta.json` ✓ |
| accuracy / f1_macro dựng lại từ confusion | 0,869709 / 0,853199 — khớp `metrics/round_005.json` ✓ |
| `argmax(y_prob)` so với `y_pred` | khác **8 / 10.761.343** dòng (0,0001%) |
| **Mã ở mục 6 chạy thật với checkpoint round 5** | nạp được, `eval()`, 395.024 tham số |
| **Logit của mã ở mục 6 so với `proj/model.py` gốc** | **max\|Δ\| = 0,0 — trùng khít từng bit** |
| Tính tất định trong `eval()` | hai lần forward cùng đầu vào cho kết quả hệt nhau |
| Cú pháp mọi khối code trong file này | `ast.parse` hợp lệ |

**Về 8 dòng lệch đó:** cả 8 đều là **hoà tuyệt đối ở fp16** — chênh lệch xác suất giữa lớp
top-1 và lớp đã chọn đúng bằng 0. Nguyên nhân: `y_pred` được tính bằng argmax trên logit fp32
trong lúc eval, còn `y_prob` mới bị ép xuống fp16 khi lưu; ở fp16 hai lớp trở nên bằng nhau và
argmax chọn chỉ số nhỏ hơn. Lành tính, và **`y_pred` mới là bản ghi có thẩm quyền**.

⚠ **`preds/round_005.npz` chỉ chứa `y_pred`, không có `y_true`.** Quy tắc hiện hành trong
`references/metrics.md` yêu cầu lưu cả hai; run này chạy trước khi quy tắc đó được ghi. Nhãn
thật vẫn lấy lại được từ `test/` (thứ tự cố định) hoặc suy ra từ hàng của confusion, nên không
mất bằng chứng — nhưng run sau phải lưu `y_true` kèm theo.

---

## 8. Caveat bắt buộc kèm mọi con số công bố từ mô hình này

1. Split theo **thời gian mô phỏng**, không theo xe — 64 điểm cắt, mỗi (class × scenario) một điểm.
2. Lớp `benign` lấy từ luồng **không có tấn công**, khiến nhóm đặc trưng `rate` mạnh bất thường.
3. **3.338.358 dòng nhập nhằng** (bị thao túng nhưng không gắn cờ) đã bị loại từ nguồn.
4. **Rò rỉ Sybil:** 100% dòng `trafficCongestionSybil` nằm trong flow mà mọi dòng đều cùng nhãn;
   F1 của lớp này phần lớn **không** đến từ kiến trúc.
5. `timeDelayAttack` là lớp khó nhất — F1 0,2281, recall 0,17 ở round 5.
6. Mất cân bằng **41:1** → đọc `f1_macro`, không đọc `accuracy`.
7. **88% flow trong test dài đúng 1 message**, hầu hết là Sybil.
8. Round 5 chọn theo điểm trên test → ước lượng **lạc quan**; xem cảnh báo ở đầu file.
9. **Một run, một seed.** Chưa có replication.
10. **Không đặt các con số này cạnh số của bài báo trong cùng một bảng.** Bài báo phân loại
    **nhị phân** trên CIC-IDS 2017 / CAN với bộ metric khác (specificity, NPV, MCC, FPR, FNR —
    đều cần TN, mà bài toán 16 lớp không định nghĩa được nếu chưa chốt cách trung bình). Cái
    được kế thừa từ bài báo là **phương pháp**, không phải con số.
