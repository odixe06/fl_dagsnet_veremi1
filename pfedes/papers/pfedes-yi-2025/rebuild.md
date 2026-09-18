# pFedES trên VeReMi NextGen / DAGSNet — bản dựng

Mọi lựa chọn dưới đây là của **bản dựng** (chủ dự án chốt 2026-09-11 → 13-09 hoặc tôi chọn và ghi
rõ), không gán cho tác giả bài báo. Phương pháp gốc: [`paper.md`](paper.md). Mã: [`proj/`](proj/).

> **Lưu ý lịch sử.** Trước bản này có một bản (12-09) chạy **LR hằng 1e-3** đã bị **thay thế hoàn toàn** và
> không đóng góp con số nào cho report: LR hằng làm F_k overfit cục bộ rồi trôi xuống trên test toàn cục.
> Cấu hình hiện hành: **C = 100 % cho cả ba kịch bản, LR cosine 1e-3 → 1e-5**. Chi tiết chỉ còn trong nhật ký
> `CONTEXT.md` (§11–12), không lặp lại ở đây.

## 1. Cấu hình đã chốt (chủ dự án, 2026-09-11)

| hạng mục | giá trị | nguồn |
|---|---|---|
| Classifier F_k (mọi client) | **DAGSNet**, 395.024 tham số, init mặc định PyTorch, seed 42, **cùng một init** cho mọi client | `knowledge/ARCHITECTURE.md`; init chung là lựa chọn của tôi |
| Proxy G | **DAGSNet với head ra 66** (`build_proxy`, 407.874 tham số) để x̂ cùng chiều x | chủ dự án chọn (thay 2-conv CNN của bài báo) |
| Server | chỉ giữ và tổng hợp θ của G; không có classifier server | bài báo |
| Kịch bản | 20 / 50 / 100 client, α = 0,5 | chủ dự án |
| Tham gia C | **100 % cho cả ba kịch bản** → K = N = 20 / 50 / 100 | chủ dự án 2026-09-12 (bài báo Table 1–2 dùng 100 %/20 %/10 % — không so trực tiếp được, xem lưu ý đầu file) |
| Round × epoch | **50 × (E = 1, E_fe = 1)** | chủ dự án |
| Batch | **512 / 512 / 256** | `knowledge/DATASET.md` §4 |
| μ | **0,5** | chủ dự án (bài báo không cho giá trị) |
| Optimizer | **AdamW, wd 1e-4**, betas/eps mặc định, **tạo mới mỗi client mỗi round** (không giữ moment) | chủ dự án (bài báo: SGD 0,01) |
| Learning rate | **cosine theo round** `lr_min + (lr − lr_min)/2·(1 + cos(π(t−1)/(T−1)))`, t = 1…50, **hằng trong một round**, chung cho η_ω và η_θ; giá trị `lr`/`lr_min` xem `scripts/gen_notebook.py::PAPER` (chốt 2026-09-13 sau LR scan local, CONTEXT.md §12) | chủ dự án 2026-09-13: **một lịch LR thống nhất cho mọi dự án anh em** (công thức của `afpha`) |
| Clip | grad-norm 1,0 cho cả ω và θ | `ARCHITECTURE.md` §4.1 |
| Precision | fp16 AMP, loss fp32, GradScaler riêng cho ω và θ | như các bản dựng trước |
| Đánh giá | **mỗi round, mọi N client**: F_k(x) trên **đủ 10.761.343 dòng test**, 10 metric/client, mean/std/min/max trên N client | chủ dự án |
| Checkpoint | `weights/round_NNN.pt` = state_dict của G + state_dict của **mọi** F_k (tensor thuần, `weights_only=True`), không optimizer | chủ dự án |

## 2. Những chỗ tôi đã chốt thay cho bài báo — phải công bố kèm mọi con số

| # | chỗ | quyết định | vì sao |
|---|---|---|---|
| 1 | hai forward ở bước ① | **một** forward trên batch ghép `[x̂; x]` (2B dòng); loss vẫn đúng Eq. (6) (CE trung bình riêng từng nửa) | module compiled bằng CUDA graph không được gọi 2 lần trước backward (replay sau đè output trước); DAGSNet launch-bound nên batch ghép rẻ hơn ~1,7×. **Hệ quả:** BatchNorm của F_k thấy thống kê của cả x̂ lẫn x trong một batch |
| 2 | "đóng băng" | `eval()` + `requires_grad=False`: BN dùng running stats, Dropout tắt; gradient vẫn chảy **xuyên qua** F_k tới G ở bước ② | x̂ và F_k(·) là hàm tất định của đầu vào khi bị đóng băng |
| 3 | Eq. (11) | chuẩn hoá theo Σ_{k∈S^t} n_k (tập được chọn) | theo chữ (n toàn cục) thì θ bị co mỗi round khi C < 100 %; ở C = 100 % hai cách trùng nhau, giữ dạng tổng quát trong mã |
| 4 | BN buffer của G khi tổng hợp | `running_mean/var` trung bình theo n_k như trọng số; `num_batches_tracked` lấy max | như FD-IDS; bài báo không nói |
| 5 | chọn client | `select_clients` là hàm thuần của (seed, round); với C = 100 % luôn trả `list(range(N))` | resume vẽ lại đúng tập; không ăn RNG train |
| 6 | RNG train | `(seed, round, client)` → Dropout (default generator) và shuffle (Generator riêng), đặt ngay trước mỗi client | kết quả không phụ thuộc GPU nào nhận client hay thứ tự; đo được 1 worker ≡ 2 worker, max\|Δw\| = 0 |
| 7 | Init F_k | cùng một init seed 42 cho mọi client | bài báo không nói (homogeneous setting) |
| 8 | Eval cache | mã giữ cơ chế: client không đổi trọng số giữ nguyên confusion; mỗi `eval_all_every = 10` round và round cuối eval lại toàn bộ và so với cache (verifier chấp nhận ≤ `CACHE_TOL_ROWS = 100` dòng nhiễu fp16 chéo GPU, đòi `cache_mismatch` trong json = số tính lại). **Ở C = 100 % nhánh cache không bao giờ chạy** (mọi client đổi trọng số mỗi round) ⇒ eval-all mỗi round, `cache_mismatch ≡ 0`. **Client c luôn eval trên worker c % 2** để eval tất định theo client | eval tất định; chi phí eval/round tỉ lệ N (đo ở §4) |
| 9 | y_pred | chỉ lưu ở `preds_rounds = [50]` (mảng (N, 10.761.343) uint8 = 1,08 GB ở 100c) | lưu mọi round là 54 GB ở 100c |
| 10 | Adam reset | moment của F_k **bị bỏ** mỗi round (chủ dự án chọn phương án này) | không cần lưu optimizer; resume chỉ cần trọng số |
| 11 | lịch LR | cosine theo round (`proj/pfedes.py::lr_at`), hằng trong round, chung η_ω = η_θ; `lr`, `lr_schedule`, `lr_min` **và `rounds`** vào `FINGERPRINT_KEYS` (lịch trải theo T nên trọng số ở round r phụ thuộc T — phiên nối tiếp phải giữ đúng `rounds`); `logs/round_NNN.json` ghi `lr` từng client, verifier so với `lr_at(cfg, r)` | bài báo: SGD hằng 0,01. C = 100 % nghĩa là 50 epoch/client; F_k đạt đỉnh trên test toàn cục sau ~1–2 epoch local rồi trôi xuống, lịch cosine chặn đà trôi (LR scan local, CONTEXT.md §12.4) |

## 3. Hợp đồng artifact (`runs/pfedes_<K>c/`)

```
weights/round_NNN.pt     {round, global: sd(G), clients: {cid: sd(F_k)} x N, cfg, fingerprint, prev_sha, metrics(mean)}
resume/round_NNN.pt      {round, rng driver, fingerprint}  — truy vết; resume không cần
confusion/round_NNN.npy  (N, 16, 16) int64 — mỗi client một ma trận, tổng mỗi ma trận = 10.761.343
preds/round_050.u8.npy   (N, 10.761.343) uint8 — chỉ round trong preds_rounds
metrics/round_NNN.json   round, selected, evaluated, cache_mismatch, 10 metric mean + *_std/_min/_max,
                         loss_w/ce_orig/loss_theta/gnorm, steps/skipped (w, theta), train_sec, eval_sec,
                         vram, backend, seconds, clients: [{cid, selected, evaluated, 10 metric, per_class[16]}]
logs/round_NNN.json      {selected: [...], clients: [stats từng client được train]}
reports/manifest.json    cfg hiệu lực, fingerprint, data_id/content_id, số dòng từng client, torch/CUDA
reports/y_true.u8.npy    nhãn test, một lần
reports/calibration.json (chỉ probe) số đo T4
history.csv              DẪN XUẤT: một dòng/round (mean + stats)
clients.csv              DẪN XUẤT: một dòng/(round, client), 10 metric
complete/round_NNN.done  TUYỆT ĐỐI cuối cùng
```

Dung lượng trọng số: 20c **33 MB/round → 1,7 GB**; 50c **81 MB → 4,0 GB**; 100c **160 MB → 8,0 GB**
(50 round). `FINGERPRINT_KEYS` (24 khoá): kiến trúc, lr, **lr_schedule, lr_min, rounds**,
weight_decay, mu, clip, n_clients, participation, batch, local_epochs, proxy_epochs, seed,
data_id, run_name. `rounds` **có** trong đó từ 13-09 (lịch cosine trải theo T); phiên nối tiếp
vẫn hợp lệ vì cùng `rounds` = 50 và `lr_at` là hàm thuần của (cfg, round).

## 4. Chi phí một round — đo thật trên 2×T4, C = 100 % (production `_cos`, 13–15/09)

Ở C = 100 % mỗi round train đúng **một epoch trên toàn bộ 43 045 415 dòng** bất kể N (nên train/round gần
như không phụ thuộc N, chỉ phụ thuộc batch) và eval là eval-all mọi round (không có cache). Số dưới đây là
trung vị trên run đã kéo về (`report_data/summary.json`); round lớn nhất là round có compile (round 1 hoặc
round đầu mỗi phiên).

| kịch bản | train / round | eval / round | round (trung vị / lớn nhất) | 50 round | phiên Kaggle (trần 12 h) |
|---|---:|---:|---:|---:|---|
| 20c (B=512) | 1046 s | 273 s | 1320 / 1448 s | **18,2 h** | 2 (28 + 22 round) |
| 50c (B=512) | 1013 s | 687 s | 1716 / 1968 s | **23,9 h** | 3 (21 + 21 + 8) |
| 100c (B=256) | 1293 s | 1286 s | 2577 / 3041 s | ≈ **36 h** (28,6 h cho 39 round) | 4 (13 + 12 + 14 + …) |

Startup + prepack + compile 9–22 phút/phiên; VRAM 7,0–7,4 GiB/GPU; train compiled 12,06 ms/phase-step @512
(probe 20c, `runs/pulls/probe20/`), eval compiled-folded @16384 ≈ 414k rows/s/GPU. 100c tốn hơn vì batch 256
gấp đôi số bước (DAGSNet launch-bound) và nghẽn 4 vCPU.

## 5. Kiểm chứng đã có (local)

| bài | kiểm gì | kết quả |
|---|---|---|
| `tests/test_units.py` | tham số F/G, round-trip flat, fold BN (max\|Δ\| 1,9e-7), eval đủ dòng, Eq. (6)/(9) và hướng gradient, Eq. (11), chọn client, **lịch LR `lr_at` (8 ca)**, 10 metric vs sklearn (5,6e-17) | 32/32 |
| `tests/test_smoke_real.py` | driver thật, 4 client thật của 100c, K=2, kế hoạch **4 round cố định** (fingerprint có `rounds`) + crash-replay (bit-identical) + phiên tiếp theo mô phỏng bằng xoá round 4 (chạy lại **bit-identical**, cache dựng lại từ confusion) + chặn `rounds` khác ở cổng + phiên "không còn gì để train" + `require_resume`; `lr` mỗi round/client khớp lịch; verify full | pass |
| `tests/test_two_workers.py` | 1 worker ≡ 2 worker, max\|Δw\| = 0 | pass |
| `tests/test_compile_gate.py` | gate certify/reject/fallback, khôi phục dropout | 7/7 |
| `tests/test_gpu_local.py` | Inductor + CUDA graph thật trên sm_86: 4 graph, tail batch, AMP, compiled ≈ eager, eval template | pass (không phải bằng chứng sm_75) |
| `tests/test_ckpt_verify.py` | hợp đồng weights, fingerprint (14 khoá khoa học kể cả `lr_schedule`/`lr_min`/`rounds` đổi hash, 6 khoá vận hành không), **16 ca tamper** (gồm json `cache_mismatch` không khớp, CM của client không được chọn nhưng eval lại dịch hàng trăm dòng, **client train ở LR lệch lịch**) + 1 ca lệch 5 dòng được chấp nhận và báo cáo, import từ mount read-only, từ chối ghép hai lịch sử | 22/22 |
| `tests/test_notebook_sim.py` | exec đúng các cell notebook probe trên cây `/kaggle` giả (CPU: mọi cell trừ calibration; GPU: tới hết calibration) | pass |
| `scripts/validate_notebooks.py` | metadata, CFG qua AST, module ↔ nguồn từng byte, thứ tự luồng, tên biến, thư mục `_sN` phải `require_resume` + title ` sN` | pass 20c/50c/100c/20c-probe/20c-s2 |
| probe Kaggle `khanhmay0304/pfedes-veremi-20-clients-probe` | 2 round thật trên 2×T4: backend compiled cả train lẫn eval, 0 non-finite, `cache_mismatch=0`, `verify_run` pass trên output kéo về | pass (0,99 h) |
