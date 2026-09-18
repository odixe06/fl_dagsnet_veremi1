# Lightweight FL for on-device NILM (Li, Yao, Qin, Wang) — trích xuất phương pháp

Nguồn: [`../../lightweight_fl_nilm.md`](../../lightweight_fl_nilm.md). Chỉ trích phần **học
tương hỗ liên bang** (§III.A bước 2–4, §III.C, Algorithm 1). Phần NAS (§III.B: MNAS, không
gian tìm kiếm nén, single-path, hardware-aware) **cố ý bỏ** theo yêu cầu chủ dự án. Mọi
chỗ bài báo để trống ghi ở §4; quyết định thay thế ở [`rebuild.md`](rebuild.md).

## 1. Bài toán của bài báo và bài toán của ta

| | bài báo | bản dựng này |
|---|---|---|
| tác vụ | hồi quy công suất thiết bị từ công suất tổng (NILM), REFIT/REDD | phân loại 16 lớp tấn công VeReMi NextGen, 66 đặc trưng |
| client | K hộ gia đình, mỗi hộ một model cá nhân hoá **kiến trúc khác nhau** (tìm bằng MNAS) | 20 / 50 / 100 client Dirichlet α = 0,5; mọi model **cùng kiến trúc DAGSNet** |
| loss cơ sở ℓ | L2 | CrossEntropy (xem `rebuild.md` §2 cho ℓ_d) |
| metric | MAE, SAE trên test riêng từng hộ | 10 metric phân loại trên test toàn cục cố định |

## 2. Khung 4 bước (§III.A)

1. **Search** — MNAS tìm kiến trúc cá nhân hoá per device. *(bỏ)*
2. **Mutual distillation** — mỗi device có model cá nhân hoá `w_s` và model proxy `w_r`
   (proxy **cùng kiến trúc trên mọi device**). Cả hai train trên dữ liệu local; tri thức
   truyền hai chiều qua chưng cất tương hỗ.
3. **Upload** — chỉ `w_r` được upload; dữ liệu thô và `w_s` ở lại device.
4. **Distribute** — server lấy trung bình `w_r` rồi phát lại cho proxy của mọi device; lặp
   2→4. **Cuối cùng mỗi device fine-tune `w_s` trên dữ liệu local.** *(bỏ — `rebuild.md` §2)*

## 3. Học tương hỗ liên bang (§III.C, Eq. 17–19, Algorithm 1)

Cho batch (x, y), `y_s = f_s(w_s, x)`, `y_r = f_r(w_r, x)`:

```
ℓ_s = ℓ(y, y_s),   ℓ_r = ℓ(y, y_r)                          (18)  label loss
ℓ_d = ℓ(y_s, y_r) / (ℓ_s + ℓ_r)                             (19)  distillation loss, trọng số thích nghi
L_s = ℓ_s + ℓ_d,   L_r = ℓ_r + ℓ_d                          (17)
```

"Distillation loss term will be weakened when the two models achieve poor performance" —
mẫu số là trọng số thích nghi để tránh chia sẻ tri thức xấu lúc đầu.

**Algorithm 1 (phần liên bang):**

```
for round t in R:
    for household k in K:                       # MỌI hộ, không lấy mẫu
        ∇_{w_s} L_train,s ; ∇_{w_r} L_train,r
        w_s ← w_s − ξ ∇_{w_s} ;  w_r ← w_r − ξ ∇_{w_r}
        upload w_r
    w̄_r ← (1/K) Σ_k w_r^k                       # trung bình ĐỀU
return global proxy w̄_r
```

## 4. Những chỗ bài báo để trống (phải tự chốt)

| # | chỗ trống | bài báo nói gì |
|---|---|---|
| 1 | dạng ℓ(y_s, y_r) cho **phân loại** | chỉ "ℓ is a basic loss function which can be L2 loss" (hồi quy) |
| 2 | gradient có chảy qua mẫu số (ℓ_s + ℓ_r) không | không nói; gọi nó là "weight of the distillation loss" |
| 3 | y_r trong L_s có bị detach không (và ngược lại) | không nói; Algorithm 1 tính ∇_{w_s} L_s và ∇_{w_r} L_r riêng |
| 4 | optimizer, learning rate ξ, batch, số epoch local, số round | không nói cho pha liên bang (chỉ ξ_α, ξ_w ký hiệu cho NAS) |
| 5 | khởi tạo `w_s`, `w_r` | không nói |
| 6 | fine-tune cuối: bao nhiêu epoch, LR | không nói |
| 7 | ℓ_s + ℓ_r = 0 (chia cho 0) | không nói |
| 8 | proxy trong bài báo: "three convolutional layers with a kernel size of 5 and a single dense layer" | ta dùng DAGSNet theo yêu cầu chủ dự án |
