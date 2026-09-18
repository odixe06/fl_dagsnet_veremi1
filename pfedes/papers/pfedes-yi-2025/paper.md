# pFedES — trích xuất phương pháp từ `00121-YiL.md`

Yi, Yu, Ren, Wang, Liu, Li. *pFedES: Generalized Proxy Feature Extractor Sharing for Model
Heterogeneous Personalized Federated Learning.* Bản markdown trong repo **không có phụ lục**
(Algorithm 1, Table 3–4, Figure 4, giá trị μ và E_fe đã chọn đều nằm ở phụ lục A–B).
Mục này chỉ ghi những gì **bài báo nói**; lựa chọn của bản dựng nằm ở [`rebuild.md`](rebuild.md).

## 1. Bài toán

- N client, mỗi client k có dữ liệu riêng D_k (non-IID) và **mô hình cục bộ riêng** F_k(ω_k)
  — cấu trúc có thể khác nhau (MHPFL). Mục tiêu Eq. (2)/(3): tối thiểu tổng loss của các
  mô hình cá thể hoá trên dữ liệu cục bộ của chính chúng.
- Server **không** giữ classifier. Thứ duy nhất được trao đổi là **proxy homogeneous feature
  extractor** G(θ) nhỏ, đặt **trước** F_k, với ràng buộc: x̂ = G(θ; x) **cùng chiều** với x
  (Figure 1(b): 2 lớp conv `padding = same`; "other feature extractor structures that
  satisfy this dimension condition can also be applied").

## 2. Một round t (Overview + Iterative Training)

1. Server chọn ngẫu nhiên K = |C·N| client → S^t, gửi G(θ^{t−1}).
2. Client k ∈ S^t, **bước ① — đóng băng G, train F_k**:
   - x̂ = G(θ^{t−1}; x); ŷ₁ = F_k(ω_k^{t−1}; x̂), ŷ₂ = F_k(ω_k^{t−1}; x) — Eq. (4)
   - ℓ₁ = ℓ(ŷ₁, y), ℓ₂ = ℓ(ŷ₂, y) (cross-entropy) — Eq. (5)
   - ℓ_ω = μ·ℓ₁ + (1−μ)·ℓ₂, **μ ∈ (0, 0,5]** — Eq. (6)
   - ω_k^t ← ω_k^{t−1} − η_ω ∇ℓ_ω — Eq. (7)
3. **Bước ② — đóng băng F_k(ω_k^t), train G**:
   - ŷ = F_k(ω_k^t; G(θ^{t−1}; x)) — Eq. (8); ℓ_θ = ℓ(ŷ, y) — Eq. (9)
   - θ_k^t ← θ^{t−1} − η_θ ∇ℓ_θ — Eq. (10)
4. Client gửi θ_k^t lên; F_k **không bao giờ rời client**.
5. Server: θ^t = Σ_{k∈S^t} (n_k / n) θ_k^t — Eq. (11). Ký hiệu n ở Preliminaries là tổng
   dữ liệu của **mọi** N client.
6. Suy luận: "**only each client's personalized heterogeneous local model F_k(ω_k) is used
   for inference**" — không dùng G lúc test.

## 3. Thiết lập thí nghiệm của bài báo

| hạng mục | bài báo |
|---|---|
| dữ liệu | MNIST / CIFAR-10 / CIFAR-100, non-IID 2/10 hoặc 10/100 lớp mỗi client; test **riêng từng client**, cùng phân bố với train (8:2) |
| mô hình | CNN-1…CNN-5 (Table 3–4, phụ lục); homogeneous = mọi client CNN-1 |
| N, C | N=10/C=100 %, N=50/C=20 %, N=100/C=10 % (Table 1–2) |
| optimizer | **SGD, η = η_ω = η_θ = 0,01** |
| E, batch | E ∈ {1, 10}, B ∈ {64, 128, 256, 512} — grid search |
| T | 100 hoặc 500 round |
| siêu tham số riêng | **μ** và **E_fe** (số epoch train G) — điều chỉnh, giá trị không có trong bản md |
| metric | accuracy trung bình của các client (mỗi client trên test của chính nó); chi phí truyền tin/tính toán tới accuracy mục tiêu |
| thí nghiệm | 3 lần, lấy trung bình |

## 4. Những chỗ bài báo để trống (phải tự chốt, xem `rebuild.md` §2)

1. Giá trị **μ** và **E_fe** đã chọn.
2. Cách khởi tạo F_k của từng client (giống nhau hay khác nhau).
3. n trong Eq. (11) khi C < 100 % (theo chữ là tổng toàn cục → θ bị co).
4. "Đóng băng" có ý nghĩa gì với BatchNorm/Dropout (bài báo dùng CNN nhỏ, không bàn).
5. Optimizer state giữa các round (SGD không có state nên bài báo không cần nói).
6. Hai forward F_k(x̂), F_k(x) ở bước ① là hai lời gọi riêng hay một batch ghép.
