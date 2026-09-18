# Lightweight-FL NILM (học tương hỗ liên bang) trên VeReMi NextGen / DAGSNet — bản dựng

Mọi lựa chọn dưới đây là của **bản dựng** (chủ dự án chốt 2026-09-15 hoặc tôi chọn và ghi
rõ), không gán cho tác giả bài báo. Phương pháp gốc: [`paper.md`](paper.md). Mã: [`proj/`](proj/).

## 1. Cấu hình đã chốt (chủ dự án, 2026-09-15)

| hạng mục | giá trị | nguồn |
|---|---|---|
| Mọi model (w_s của từng client, w_r của từng client, w̄_r của server) | **DAGSNet**, 395.024 tham số, init mặc định PyTorch, seed 42, **cùng một init** cho tất cả | chủ dự án; `knowledge/ARCHITECTURE.md` |
| NAS / backbone | **bỏ hẳn** §III.B (MNAS); chỉ dựng học tương hỗ giữa các classifier | chủ dự án |
| Kịch bản | 20 / 50 / 100 client, α = 0,5 | chủ dự án |
| Tham gia | **mọi client mỗi round** (Algorithm 1 lặp qua mọi hộ) | bài báo |
| Round × epoch | **50 × 1 epoch local** | chủ dự án |
| Batch | **512 / 512 / 256** | chủ dự án; `knowledge/DATASET.md` §4 |
| Optimizer | **AdamW, wd 1e-4**, betas/eps mặc định, **tạo mới mỗi client mỗi round**, một optimizer chung hai nhóm tham số (= hai optimizer cùng rate, đã kiểm bit-identical) | chủ dự án (Q6); bài báo không nêu |
| Learning rate | **cosine theo round 1e-3 → 1e-5, T = 50** (`proj/nilm.py::lr_at`), hằng trong round, chung cho w_s và w_r | chủ dự án — lịch thống nhất mọi dự án anh em |
| ℓ_s, ℓ_r | CrossEntropy (nhãn số nguyên) | phân loại |
| ℓ(y_s, y_r) | **KL trên softmax, 2 chiều** (Deep Mutual Learning): w_s học KL(p_r‖p_s) với p_r detach; w_r học KL(p_s‖p_r) với p_s detach; T = 1 | chủ dự án (Q1) |
| Mẫu số 1/(ℓ_s + ℓ_r) | **stop-gradient** (trọng số hằng theo batch), clamp ≥ 1e-6 | chủ dự án (Q2); clamp là của tôi |
| Tổng hợp | **w̄_r = (1/K) Σ_k w_r^k trung bình đều** (Algorithm 1), BN running stats trung bình như tham số, `num_batches_tracked` lấy max | chủ dự án (Q3) |
| Fine-tune cuối (§III.A bước 4) | **bỏ** | chủ dự án (Q5) |
| Clip | grad-norm 1,0 **riêng từng model** | `ARCHITECTURE.md` §4.1 |
| Precision | fp16 AMP, loss fp32, một GradScaler (skip là nguyên tử cho cả hai model) | như các bản dựng trước |
| Đánh giá | **mỗi round, mọi N client**: w_s^k(x) trên **đủ 10.761.343 dòng test**, 10 metric/client, mean/std/min/max trên N; **cộng w̄_r** (1 model, cột `global_*`, không trộn vào mean) | chủ dự án (Q4) |
| Checkpoint | `weights/round_NNN.pt` = state_dict của w̄_r + state_dict của **mọi** w_s^k (tensor thuần, `weights_only=True`), không optimizer, không module | chủ dự án |
| W&B | project `nilm-veremi`, run id = `run_name`; key nhúng theo uỷ quyền thường trực 2026-09-07 (notebook private, `.ipynb` không commit) | chủ dự án (Q7) |

## 2. Những chỗ tôi đã chốt thay cho bài báo — phải công bố kèm mọi con số

| # | chỗ | quyết định | vì sao |
|---|---|---|---|
| 1 | một kiến trúc cho mọi model | DAGSNet cho w_s, w_r, w̄_r; không MNAS | yêu cầu chủ dự án. **Hệ quả:** vì w_s và w_r cùng init, cùng dữ liệu, ở **round 1** hai model chỉ khác nhau qua mặt nạ dropout ⇒ ℓ_d ≈ 0 ở round 1; chưng cất chỉ thật sự có nghĩa từ round 2 (w_r ← w̄_r) |
| 2 | ℓ(y_s, y_r) = KL hai chiều, mỗi chiều detach phía kia | dạng chuẩn của mutual distillation cho phân loại; L2 của bài báo là cho đầu ra hồi quy | |
| 3 | mẫu số (ℓ_s + ℓ_r) là stop-gradient | bài báo gọi nó là "weight"; đạo hàm qua mẫu số sinh thành phần đẩy CE **tăng** để giảm ℓ_d | |
| 4 | clamp mẫu số ≥ 1e-6 | CE có thể = 0 chính xác ở fp32 khi logit bão hoà; chia cho 0 làm hỏng bước | không ảnh hưởng khi CE > 1e-6 (thực tế luôn) |
| 5 | một backward cho L_s + L_r | với các cross-term đã detach, ∇_{w_s} L_r = ∇_{w_r} L_s = 0 ⇒ đúng bằng hai gradient của Algorithm 1; tiết kiệm một backward | đã kiểm bằng autograd (`tests/test_units.py`) |
| 6 | một AdamW hai nhóm tham số | Adam là per-parameter ⇒ bằng hai optimizer cùng rate (kiểm bit-identical); GradScaler skip nguyên tử | |
| 7 | trung bình đều 1/K | Algorithm 1 viết vậy; FedAvg theo n_k là lựa chọn khác, chủ dự án chọn theo bài báo | client lớn nhất/nhỏ nhất ở 20c chênh 6,8× số dòng |
| 8 | Adam reset mỗi round | không cần lưu optimizer; checkpoint = trọng số; resume bit-identical | |
| 9 | lịch LR cosine theo round | bài báo không nêu; chủ dự án chọn lịch thống nhất | `lr`, `lr_schedule`, `lr_min`, `rounds` nằm trong fingerprint |
| 10 | bỏ fine-tune cuối | w_s đã học CE local mỗi round; không có test riêng client để đo lợi ích | |
| 11 | RNG train | `(seed, round, client)` → Dropout (default generator) và shuffle (Generator riêng) | 1 worker ≡ 2 worker bit-identical (`tests/test_two_workers.py`) |
| 12 | y_pred | chỉ lưu ở `preds_rounds = [50]` (N × 10,76 M uint8 + 10,76 M cho w̄_r = 1,09 GB ở 100c) | lưu mọi round là 54 GB ở 100c |
| 13 | eval w̄_r trên worker cuối | worker 1 (N chẵn ⇒ hai worker cùng số client; +1 model) | cân tải |

## 3. Hợp đồng artifact (`runs/nilm_<K>c/`)

```
weights/round_NNN.pt      {round, global: sd(w̄_r), clients: {cid: sd(w_s^k)} x N, cfg, fingerprint,
                           prev_sha, metrics (mean N client), global_metrics (w̄_r)}
resume/round_NNN.pt       {round, rng driver, fingerprint}  — truy vết; resume không cần
confusion/round_NNN.npy   (N, 16, 16) int64 — mỗi client một ma trận, tổng mỗi ma trận = 10.761.343
confusion/global_NNN.npy  (16, 16) int64 — w̄_r
preds/round_050.u8.npy    (N, 10.761.343) uint8 — chỉ round trong preds_rounds
preds/global_050.u8.npy   (10.761.343,) uint8
metrics/round_NNN.json    round, evaluated, lr, 10 metric mean + *_std/_min/_max, global_<10 metric>,
                          {loss_s,loss_r,ce_s,ce_r,kl_s,kl_r,gnorm_s,gnorm_r}_client_mean, steps, skipped,
                          train_sec, eval_sec, vram, backend, seconds,
                          clients: [{cid, 10 metric, per_class[16]}], global: {10 metric, per_class[16]}
logs/round_NNN.json       {clients: [{cid, n_k, lr, steps, applied, skipped, nonfinite, 8 giá trị trung bình, sec, vram}]}
reports/manifest.json     cfg hiệu lực, fingerprint, data_id/content_id, số dòng từng client, torch/CUDA
reports/y_true.u8.npy     nhãn test, một lần
reports/calibration.json  (chỉ probe) số đo T4
history.csv               DẪN XUẤT: một dòng/round (mean + stats + global_*)
clients.csv               DẪN XUẤT: một dòng/(round, client), 10 metric
complete/round_NNN.done   TUYỆT ĐỐI cuối cùng
```

Dung lượng trọng số/round: (N + 1) × 1,58 MB → 20c **33 MB → 1,7 GB**; 50c **81 MB → 4,0 GB**;
100c **160 MB → 8,0 GB** (50 round). `FINGERPRINT_KEYS` (21 khoá): 9 khoá kiến trúc, lr,
lr_schedule, lr_min, rounds, weight_decay, clip, n_clients, batch, local_epochs, seed, data_id,
run_name.

## 4. Chi phí một round — đo bằng probe 20c trên 2×T4 (2026-09-15, `runs/pulls/probe20/`)

Probe `minhtriethihi/nilm-fl-veremi-20-clients-probe` (2 round, 0,65 h session, 0,99 h quota):

| đại lượng | đo được |
|---|---:|
| bước mutual (2 forward + 1 backward, 2 DAGSNet) compiled, B = 512 | **13,11 ms** (calibration) / **13,96 ms** hiệu dụng (cả overhead per-client); eager 63,5 ms → ×4,8 |
| compile + gate (train + eval) | 90 s calibration; worker sẵn sàng **516 s** sau khi session bắt đầu (prepack 131 s) |
| eval compiled-folded, một model, đủ test | **355k rows/s @16384** (eager-folded 312k); @32768 354k → **giữ 16384** |
| round 20c steady | **892 s** = train 598 s + eval 293 s (21 model / 2 GPU = 27,9 s/model) + commit |
| VRAM | 7,12 GiB/GPU train, 7,10 eval |
| skip AMP | 24 / 18 trên 84.083 bước (warm-up GradScaler, ≤ 2/client) |
| gate compile sm_75 | w_s max\|Δlogit\| 5,5–7,3e-4, rel\|Δgrad\| 2,7–4,1e-3, 0 flip decisive; eval 7,3e-4 |

Kết quả 2 round (mean 20 model cá nhân hoá / proxy w̄_r): f1_macro 0,514 → **0,520** (std 0,10, min 0,30,
max 0,68) / 0,237 → **0,666**; accuracy 0,584 → 0,608 / 0,809. Round 1: kl_s = kl_r = 0,033, ce_s = ce_r
(w_s ≡ w_r về thống kê, đúng deviation #1); round 2: ce_r 0,397 > ce_s 0,219, kl_s 0,232 — proxy toàn
cục kém CE local hơn nhưng tổng quát hoá tốt hơn hẳn trên test toàn cục.

Ngoại suy cho production (train/round ≈ không đổi theo N vì mọi client train đúng một epoch trên
43 M dòng; eval = (N+1) × 27,9 s / 2; bước @256 ≈ 0,92 × bước @512 theo tỉ lệ đo ở `tinyproto`;
100c cộng 20 % nghẽn 4 vCPU đo ở `fd_ids`):

| kịch bản | train/round | eval/round | round | 50 round | phiên (11 h) |
|---|---:|---:|---:|---:|---:|
| 20c (B=512) | 598 s *(đo)* | 293 s *(đo)* | **14,9 phút** *(đo)* | **12,4 h** | 2 (11 h ≈ 43 round + ~2,5 h) |
| 50c (B=512) | ~590 s | ~725 s | ~22 phút | **~18,5 h** | 2 (11 h + ~8 h) |
| 100c (B=256) | ~1.290 s | ~1.425 s | ~45 phút | **~38 h** | 4 (3 × 11 h + ~6 h); `--max-hours 10.5` vì commit 8 GB |

Tổng ≈ **69 h round-time + 8 phiên × 9 phút startup ≈ 70 h**. Quota còn lại sau probe (đọc 12:00Z
15-09, refresh 2026-09-19T00:00Z): minhtran0601 28,2 + minhtrit06 18,9 + odixe0502 11,1 +
minhtriethihi ~8,6 + khanhmay0304 3,6 ≈ **70 h** — sát; 100c chắc chắn kéo qua kỳ refresh.
Dung lượng output: 20c 1,7 GB, 50c 4,0 GB, 100c 8,0 GB (+1,1 GB preds round 50 ở 100c).

## 5. Kiểm chứng đã có (local, 2026-09-15)

| test | phạm vi | kết quả |
|---|---|---|
| `tests/test_units.py` | 395.024 tham số; flat round-trip Δ=0; fold BN 2,4e-7; eval đủ dòng; **Eq. (17)–(19) vs tham chiếu tay**; **cấu trúc gradient** (một backward = ∇L_s ⊕ ∇L_r, mẫu số không đạo hàm); 1 AdamW ≡ 2 AdamW bit; aggregate 1/K; lr_at; 10 metric vs sklearn 5,6e-17 | **36/36** |
| `tests/test_smoke_real.py` | 4 client thật (100c), CPU, 3 round → verify → crash replay bit-identical → resume phiên sau bit-identical → `rounds` khác bị chặn → phiên "không có gì làm" → gate require_resume | pass |
| `tests/test_two_workers.py` | 1 worker ≡ 2 worker: max\|Δw\| = 0, metric bằng nhau | pass |
| `tests/test_compile_gate.py` | gate certify/reject/fallback, khôi phục dropout/trainable | 7/7 |
| `tests/test_ckpt_verify.py` | hợp đồng weights (8 khoá, tensor thuần, BN buffer), fingerprint 13 khoá khoa học/5 khoá vận hành, **22 tamper case bị bắt**, import read-only, từ chối ghép hai lịch sử | 27/27 |
| `tests/test_gpu_local.py` | `torch.compile(reduce-overhead)` thật trên sm_86: gate pass, 2 client qua graph, compiled vs eager Δw rel 1,8e-2, eval compiled > eager ở steady state | pass (không phải bằng chứng cho sm_75) |
| `tests/test_notebook_sim.py` (CPU + `--gpu`) | chạy từng cell notebook probe trên cây `/kaggle` giả (có sidecar `.stats.json`) | pass |
| `scripts/validate_notebooks.py` | metadata, CFG qua AST, module nhúng ↔ nguồn từng byte, luồng cell, tên biến | pass 20c/50c/100c/20c_probe |

Chạy: `CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/<file>` (GPU test bỏ
`CUDA_VISIBLE_DEVICES`). Test 2 worker và GPU local cần MemAvailable ≥ ~4,5 GB (prepack chạy trong
process con để parent không giữ pyarrow cạnh Inductor).
