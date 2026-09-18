# TinyProto — ngữ cảnh tiếp nối

> **📌 Note 2026-09-13 (chủ dự án, ghi từ phiên `~/nckh/pfedes`): thống nhất MỘT lịch learning rate cho mọi dự án anh em.**
> Lịch: AdamW, **cosine theo round** `lr(t) = lr_min + (lr − lr_min)/2·(1 + cos(π(t−1)/(T−1)))`, **lr = 1e-3, lr_min = 1e-5, T = 50**,
> hằng trong một round (công thức `afpha`; bản tham chiếu `~/nckh/pfedes/papers/pfedes-yi-2025/proj/pfedes.py::lr_at`).
> pFedES đã chạy lại với lịch này (kernel `*-cos`, 13-09). **Dự án này CHƯA sinh/push notebook mới** — chỉ làm khi chủ dự án yêu cầu;
> quota tuần 13→19-09 đã dành cho pFedES (~92 h). Chỗ cần sửa khi làm: `scripts/gen_notebooks.py` (AdamW 1e-3 hằng, wd 1e-4, **giữ optimizer giữa các round**) + driver/verifier — ngoài lịch LR còn phải quyết định có reset optimizer mỗi round như pfedes/afpha không (chủ dự án chưa chốt).
> Lý do và bằng chứng (đỉnh sớm sau ~2 epoch local, cosine chặn đà trôi nhưng không kéo lại đỉnh): `~/nckh/pfedes/CONTEXT.md` §12.

**Cập nhật: 2026-09-11, CẢ BA KỊCH BẢN ĐÃ XONG, báo cáo đã viết, Asia/Bangkok.**

Xây dựng lại phương pháp TinyProto-FP (AAAI-26 #02846, Lee & Choi) cho ba kịch bản FL
20/50/100 client trên VeReMi, mỗi client một DAGSNet riêng, chạy trên Kaggle 2 × Tesla T4.

**BA KỊCH BẢN ĐÃ XONG.** Mỗi kịch bản là **một thư mục run duy nhất** (đã ghép các phiên), chạy
đủ 50 round trên fixed test 10.761.343 dòng:

| Kịch bản | Phiên | `verify_run.py --require-complete` | Artifact |
|---|---|---|---|
| 20 client | 1 | **10.573 check / 0 fail** | `papers/tinyproto-lee-2026/runs/tinyproto_fp_20client/` |
| 50 client | 2, đã ghép | **21.133 check / 0 fail** | `…/runs/tinyproto_fp_50client/` |
| 100 client | 4, đã ghép | **38.733 check / 0 fail** | `…/runs/tinyproto_fp_100client/` |

**Báo cáo hoàn chỉnh: [`report.md`](report.md) ở thư mục gốc** — 23 bảng sinh tự động bằng
`scripts/make_report_tables.py --splice report.md`, CSV per-client ở
`papers/tinyproto-lee-2026/report/`. Xem §12 về cơ chế phân kỳ (τ) — đó là kết quả khoa học
chính của toàn bộ công việc này.

⚠ **Phát hiện phải đọc TRƯỚC khi diễn giải bất kỳ con số nào:** hai kịch bản k=0,3 (50 và
100 client) **phân kỳ**, và **đã tìm được cơ chế** (§12): APS có vòng lặp dương khi
τ = ‖μĉ_G‖/‖ĉ_L‖ > 1. τ đo ở round 1 dự báo đúng hai lớp sẽ nổ. **20 client hội tụ không phải
vì có 20 client** — ở k=0,3 nó cũng vượt ngưỡng; nó thoát nhờ sweep chọn k=0,1 hơn 0,0021.
Số client và k **bị lẫn**. Báo cáo nêu cả round đỉnh lẫn round 50.

⚠ **`kernel_sources` chéo tài khoản KHÔNG hoạt động** (§10). Bàn giao chéo tài khoản bằng
**dataset** — đã dùng hai lần (s1→s2, s3→s4), cả hai lần **byte-exact**.

---

## 0. Mục lục — cần gì thì đọc đâu

**Nguyên tắc: CONTEXT.md chỉ giữ quyết định, chính sách và trạng thái. Mọi thứ khác ở file riêng.**

| Cần biết | Đọc |
|---|---|
| **BÁO CÁO KẾT QUẢ HOÀN CHỈNH** (phương pháp, 23 bảng, cơ chế phân kỳ) | [`report.md`](report.md) |
| CSV per-client + mean-over-clients, `summary.json` | `papers/tinyproto-lee-2026/report/` |
| **Mọi kết quả test** (local + Kaggle), số đo, đính chính | [`papers/tinyproto-lee-2026/TESTLOG.md`](papers/tinyproto-lee-2026/TESTLOG.md) |
| Log thô từng lần chạy | [`papers/tinyproto-lee-2026/review-logs/INDEX.md`](papers/tinyproto-lee-2026/review-logs/INDEX.md) |
| Thuật toán, công thức, mọi lựa chọn triển khai và lý do | [`papers/tinyproto-lee-2026/rebuild.md`](papers/tinyproto-lee-2026/rebuild.md) |
| Chi phí truyền tin + ước lượng μ ban đầu (đã tách khỏi `knowledge/`) | [`rebuild.md` §8](papers/tinyproto-lee-2026/rebuild.md) |
| **Quy tắc: cái gì được ghi vào `knowledge/`** | [`knowledge/README.md`](knowledge/README.md) |
| Tóm tắt bài báo gốc | [`papers/tinyproto-lee-2026/paper.md`](papers/tinyproto-lee-2026/paper.md) |
| DAGSNet: kiến trúc, tham số khởi tạo, hợp đồng input | [`knowledge/ARCHITECTURE.md`](knowledge/ARCHITECTURE.md) |
| Dataset: cột, scaler, số dòng, phân bố lớp (đã audit thật) | [`knowledge/DATASET.md`](knowledge/DATASET.md) |
| **4 dataset VeReMi trên Kaggle đều public** (nên chạy chéo tài khoản được) | [`knowledge/KAGGLE_DATASETS.md`](knowledge/KAGGLE_DATASETS.md) |
| Máy local: RAM/GPU, sự cố đã gặp, cách chạy test an toàn | [`knowledge/LOCAL_ENV.md`](knowledge/LOCAL_ENV.md) |
| Image/runtime Kaggle đã kiểm chứng (digest + xuất xứ) | [`knowledge/runtime.json`](knowledge/runtime.json) |
| Quy trình làm việc với Kaggle nói chung | `.agents/skills/kaggle-training-notebook/SKILL.md` |
| **Bẫy thư viện/runtime đã gặp** (torch, vmap, AMP, image) | `.agents/skills/kaggle-training-notebook/references/library-runtime.md` |
| **Kết quả session 1 + phát hiện k=0,3 suy giảm + ngân sách suy lại** | §9 ngay trong file này |
| **Session 2: 50 client XONG, k=0,3 phân kỳ ở round 50, 3 lỗi pull** | §11 ngay trong file này |
| **`kernel_sources` chéo tài khoản KHÔNG chạy — bàn giao bằng dataset** | §10 ngay trong file này |
| **Artifact đã kéo về, ghép và verify — cả ba hoàn chỉnh** | `papers/tinyproto-lee-2026/runs/` |
| **Session 3+4 của 100 client, bàn giao s3→s4, cơ chế τ, báo cáo** | §12 ngay trong file này |
| **Kế hoạch production 4 tài khoản**, phân công, ranh giới session | §5d ngay trong file này |
| **Notebook production sẽ push** (7 bản chia session) | `papers/tinyproto-lee-2026/production/` |
| **Review trực tiếp bộ production, μ và các điểm cần sửa** | [`papers/tinyproto-lee-2026/PRODUCTION_REVIEW.md`](papers/tinyproto-lee-2026/PRODUCTION_REVIEW.md) + §8 |

**Mã nguồn:** `src/*.py` là nguồn thuật toán duy nhất. `scripts/gen_notebooks.py` nhúng nguyên
module vào notebook qua `%%writefile`. **Sửa `src/` → phải regenerate → phải validate.**
Không bao giờ sửa thuật toán trực tiếp trong `.ipynb`.

**7 notebook gốc** ở `papers/tinyproto-lee-2026/notebook/`: `calibration`; `mu-sweep`,
`mu-sweep-50client`, `mu-sweep-100client`; `train-{20,50,100}client`. Bộ này là **route A**
(cùng tài khoản `odixe0502`, μ đọc từ kernel output của sweep) — dùng cho sweep và tham chiếu.

**7 notebook production chia session** ở `papers/tinyproto-lee-2026/production/` — đây là bộ
**thực sự đẩy lên Kaggle** cho 50 round: `train-20client-s1`, `train-50client-s{1,2}`,
`train-100client-s{1,2,3,4}`. Mỗi cái đã gắn sẵn chủ sở hữu, dataset μ và session liền trước.
Notebook kiểm chứng riêng: `papers/tinyproto-lee-2026/validation/`.

---

## 1. Quyết định khoa học đã chốt

Chi tiết công thức ở `rebuild.md`. Tóm tắt để không phải mở file khi chỉ cần nhớ:

- **TinyProto-FP** (CPS + APS ghép lên FedProto). **Không** làm TinyProto-FT.
- Mỗi client một **DAGSNet** riêng, cùng kiến trúc/khởi tạo nhưng **giữ trọng số cá thể**
  suốt run. Không có global model. **Bỏ qua** feature extractor của bài báo — dùng thẳng 66 đặc trưng.
- 20/50/100 client, 100% tham gia, **50 round**, 1 local epoch.
- batch **512/512/256**; AdamW **lr=1e-3**, wd=1e-4, giữ optimizer giữa các round.
  (Bài báo ghi lr=0.01; dùng 1e-3 là **quyết định của người dùng**, phải báo cáo đúng như vậy.)
- λ=1, CPS **s=50**, d=256, seed=42; masked per-sample MSE; prototype tính **sau** epoch.
- **Hai quy tắc đánh giá:** `proto` (kết quả chính, nearest LOCAL prototype dày — Eq. 12) và
  `clf` (argmax). Mỗi round **mọi client chạy đủ fixed global test**; giữ 10 metric/client
  + mean/std/min/max + pooled.
- **Checkpoint chỉ lưu trọng số** (`state_dict` phẳng + manifest), **không** pickle `nn.Module`.
- **Sweep μ RIÊNG cho từng kịch bản 20/50/100.** Không chuyển k hay μ từ kịch bản này sang kịch
  bản khác. Grid k=[0.03, 0.1, 0.3, 1, 3], 4 round, validation **2%** phân tầng từ **từng** client,
  seed 42; μ=k/mean(n_ij). Chọn theo mean-client proto f1_macro ở round cuối. Production dùng μ
  tuyệt đối thắng của **chính** kịch bản đó và 100% train.
  Giao thức này được **đóng băng** trong `src/sweep.py::MU_PROTOCOL` và production kiểm lại nó.
  Bài báo chỉ nói "systematic parameter selection ... in the appendix" — PDF **không có** appendix
  và **không** dùng cụm "grid search". Đây là lựa chọn triển khai, **không được gán cho tác giả**.
- Không tự mở rộng sang FT, FedAvg, hay DDP đồng bộ gradient giữa các client.
- Khi không chắc về yêu cầu khoa học: **hỏi trước**.

---

## 2. Cấu hình vận hành (đều suy từ số đo — xem TESTLOG §2)

`compile=True`, `amp=True`, `eval_group=1`, `eval_batch=8192`; mặc định `max_hours=11.0`
(production 50c s2=7.5, 100c s3=8.0, xem §5d);
`expected_round_s` = 800/1300/2500 s cho 20/50/100 client.

---

## 3. Máy local — ràng buộc ưu tiên

**WSL 8 GB RAM (thấy 7.803 MiB), GPU RTX 3050 4 GB.** Đã từng tràn RAM làm sập WSL.

- **Mọi** test đi qua `scripts/run_local_checked.py` (khoá chống chạy chồng, trần RSS cây process,
  sàn MemAvailable, BLAS/OMP=1). Bị watchdog dừng thì **giảm fixture**, không nới giới hạn.
- Một cây kiểm thử tại một thời điểm; local 1 worker, `compile=False`.
- Không `ulimit -v` với CUDA; không đọc toàn bộ train vào RAM; không chạy Inductor local.
- Bộ test local **chạy trên CPU 1 worker** — nó **không** kiểm chứng đường hai GPU.
- Chi tiết và lệnh an toàn: `knowledge/LOCAL_ENV.md`.

---

## 4. Dữ liệu

- Local: `dataset/fl_client/alpha05/{20,50,100}_client/train/client_id=NNN`,
  test `dataset/centralized/test`.
- Kaggle: `odixe0502/veremi-fl-{20,50,100}client` + `odixe0502/veremi-nextgen2026-centralized`.
- Mỗi kịch bản 43.045.415 dòng train; fixed test 10.761.343; 66 cột `f_*`; 16 lớp.
- **Train đã z-score sẵn; test là raw và phải áp scaler fit-trên-train đúng một lần.**
  Không bao giờ chuẩn hoá lại train. Kiểm chứng số liệu: `knowledge/DATASET.md`.

---

## 5. Kaggle — tài khoản, quota, quyền chạy

- **Chỉ tin `scripts/verify_kaggle_identity.py <user>`** (introspection phía server) để biết CLI
  đang là tài khoản nào. Không tin ghi chép cũ, không tin helper tự báo cáo. Ghi chép phiên 1 từng
  ghi sai tài khoản đang hoạt động.
- Đổi tài khoản: dùng `.agents/skills/kaggle-training-notebook/scripts/kaggle_account.py`
  (`use <user> --confirm` hoặc `ensure --gpu-hours N --confirm`), **không** copy tay
  `credentials.json` nữa — copy tay chỉ đổi CLI, để `.mcp.json`/`.vscode`/`.codex` giữ token cũ
  và CLI/MCP lệch nhau. Không in token, không chép credential vào notebook/log/CONTEXT.
- **CHÍNH SÁCH XOAY TÀI KHOẢN (người dùng chốt 2026-09-09, THAY THẾ bản 2026-09-08):**
  agent **tự đổi tài khoản** khi việc đổi là phù hợp và cần thiết, tự truyền `--confirm`,
  không cần hỏi từng lần. Bản cũ (mỗi lần ghi đè `credentials.json` phải có người xác nhận)
  **đã hết hiệu lực**. Đã ghi vào skill để dùng cho các phiên sau:
  `.agents/skills/kaggle-training-notebook/references/multi-account.md` (và bản `.claude/`).
  Ràng buộc còn giữ: **báo rõ mỗi lần đổi** (tài khoản nào, vì sao, có đổi lại không);
  `kaggle auth login --force` **KHÔNG** nằm trong quyền này vì nó phá refresh token;
  danh tính lấy từ **server** (`verify_kaggle_identity.py <user>`), không lấy từ file config;
  đổi xong phải reconnect MCP client. Chọn tài khoản **theo kế hoạch huấn luyện**.
- **Roster 2026-09-08.** `khanhngoc0304` đã bị người dùng xoá; snapshot OAuth + MCP token local
  đã gỡ khỏi `~/.kaggle/accounts/`. Thêm `khanhmay0304` và `minhtrit06`: **mới có MCP token,
  chưa có OAuth** → chưa dùng/đo quota được cho tới khi người dùng browser-login (xem §7).
  Cả ba tài khoản mới/cũ nằm chung group `nckh_minhtriet`; **group không đảm bảo quyền đọc từng
  dataset** — phải kiểm bằng chính tài khoản sẽ chạy.
- **Bài học giữ lại từ `khanhngoc0304`: quota KHÔNG chứng minh có GPU.** Tài khoản đó vẫn báo
  30 h nhưng 3 lần thử đều `device_count()==0`, cùng notebook chạy trên odixe0502 thì được 2 × T4.
  Là vấn đề phạm vi tài khoản, **không phải code**. Không "sửa" bằng cách bỏ assert hay cài lại
  torch. Người dùng chốt 2026-09-08: **chỉ đọc quota OAuth** cho 2 tài khoản mới, không push
  notebook smoke để kiểm GPU — nên "đủ GPU" của chúng **chưa được chứng minh**, chỉ mới suy đoán.
- `minhtran0601` có notebook AFPHA từng chạy — **không dừng/sửa/ghi đè AFPHA**. Khi lập kế hoạch
  nhiều phiên, loại nó ra bằng `plan --exclude minhtran0601` trừ khi người dùng nói khác.
- Quota **30 h/tuần/tài khoản**, tối đa **2 phiên GPU đồng thời**, mỗi phiên tối đa **12 h**.
  **Sau khi mọi thứ xong (2026-09-11 ~00:00 UTC, `kaggle_account.py quota`):** khanhmay0304
  4,14 h, minhtran0601 1,33 h, odixe0502 0,66 h, minhtrit06 0,49 h — số này dao động giữa
  hai lần đọc cách nhau 1 h (minhtrit06 4,93 → 0,49 khi không chạy gì) nên **chỉ tin số đọc
  ngay trước khi push**. Refresh 2026-09-12T00:00:00Z.
  Đo lại 2026-09-08 sau 3 sweep: odixe0502 **18,92 h**, minhtran0601 **12,85 h**,
  khanhmay0304 **30 h**, minhtrit06 **30 h**; refresh **2026-09-12T00:00:00Z**. Tổng **91,8 h**
  cho **62 h** production. Số của odixe0502 là đọc sống qua MCP, ba số kia do `plan` báo cáo.
- Ghép nhiều phiên thành một lần chạy hoàn chỉnh: ranh giới **trong cùng một tài khoản** và
  **giữa hai tài khoản** đều đã hỗ trợ (2026-09-08). Generator có `--resume-from OWNER/SLUG` và
  `--mu-dataset OWNER/SLUG`; xem §5d. Ghi chú cũ "chưa hỗ trợ chéo tài khoản" **đã hết hiệu lực**.
- **QUYỀN CHẠY (người dùng chốt 2026-09-08):** agent **chỉ chạy khi được đồng ý từng lần**.
  Người dùng đã đồng ý **push 3 sweep μ**. Production 50 round **vẫn phải hỏi lại**.
- Ngân sách (đo thật, 2026-09-08): ba sweep đã tốn **~7,4 h**; production còn lại **~64 h**.
  Production 100 client **~33,9 h** (round ổn định đo được **40,7 phút**) → **vượt quota tuần của
  một tài khoản**, bắt buộc ~4 phiên. 20 client ~10,5 h (1 phiên), 50 client ~17,6 h (2 phiên).
- **Đo, đừng ngoại suy.** Tốc độ eval ở 100 client là **583.666** client-dòng/s, không phải 617k
  như ở 20 client — chi phí cố định mỗi client không co giãn tuyến tính. Phép chiếu cũ từ 20
  client tình cờ khá sát (quá cao ở train, quá thấp ở eval, triệt tiêu nhau), nhưng đó là may
  chứ không phải phương pháp đúng.
- **Nguồn có thẩm quyền về thời gian là `logs/timing_*.json` trong artifact**, không phải W&B
  summary. `wandb.Api()` **cache** — phải tạo mới mỗi lần poll, nếu không sẽ đọc ra số liệu cũ
  (lỗi này đã một lần dẫn tới ước lượng setup sai gấp 7 lần).

---

## 5b. Tham số μ thu được từ grid search — CẢ BA SWEEP ĐÃ XONG (2026-09-08)

Đây là **đầu ra chính** của giai đoạn sweep. Ba notebook đã chạy hết trên `odixe0502`, 2 × T4,
đều `status: complete`, `protocol v1`, đủ 5 ứng viên × 4 round, validation 2% / seed 42.

### Giá trị dùng cho production

| Kịch bản | k thắng | **μ tuyệt đối** | mean(n_ij) | val proto f1_macro | Notebook nguồn |
|---|---|---|---|---|---|
| 20 client | **0.1** | **7.585696e-07** | 131.827,1 | 0,6331 | `odixe0502/tinyproto-fp-mu-sweep` |
| 50 client | **0.3** | **5.646573e-06** | 53.129,6 | 0,5162 | `odixe0502/tinyproto-fp-mu-sweep-50client` |
| 100 client | **0.3** | **1.126461e-05** | 26.632,1 | 0,4297 | `odixe0502/tinyproto-fp-mu-sweep-100client` |

Quan hệ: `μ = k / mean(n_ij)` — kiểm lại được từ ba dòng trên.

**Hai điều phải nhớ khi đọc bảng này:**

1. **20 client chọn k khác (0.1) với 50/100 client (0.3)** theo đúng tiêu chí round 4.
   Chênh lệch ở 20 client nhỏ, chưa chứng minh đây là k tối ưu sau 50 round; xem §8.
2. **50 và 100 client cùng k=0.3 nhưng μ tuyệt đối lệch 2×** (5,65e-06 vs 1,13e-05), vì
   `mean(n_ij)` khác nhau. Cùng k không đồng nghĩa cùng μ; hai kết quả thắng giống k không
   tự chứng minh chuyển k luôn sai. Quyết định của dự án vẫn là sweep riêng, giữ μ tuyệt đối.

### Toàn bộ grid (val proto f1_macro ở round cuối)

| k | 20 client | 50 client | 100 client |
|---|---|---|---|
| 0.03 | 0,6302 | 0,5021 | 0,4111 |
| 0.1 | **0,6331** | 0,5045 | 0,4139 |
| 0.3 | 0,6310 | **0,5162** | **0,4297** |
| 1.0 | 0,3420 | 0,2736 | 0,2462 |
| 3.0 | 0,2392 | 0,1926 | 0,1679 |

### Phát hiện: regularizer APS tăng mạnh ở hai điểm đã thử k=1 và k=3

Quỹ đạo reg loss của kịch bản 100 client qua 4 round cho thấy rõ:

| k | reg loss r2 | r3 | r4 | kết cục |
|---|---|---|---|---|
| 0.03 | 0,0336 | 0,0031 | 0,0023 | reg giảm trong 4 round |
| 0.1 | 0,0862 | 0,0085 | 0,0027 | reg giảm trong 4 round |
| 0.3 | 0,0975 | 0,0177 | 0,0124 | reg giảm trong 4 round |
| 1.0 | 0,2224 | 0,9556 | **13,02** | phân kỳ |
| 3.0 | 1,6772 | **258,78** | **125.175,04** | phân kỳ mạnh |

Cả ba kịch bản cùng dạng ở **năm điểm đã đo**: k=0.03/0.1/0.3 có reg giảm, k=1/3 có reg
bùng lên và F1 giảm. Chưa suy được ngưỡng chính xác hay ổn định 50 round từ bốn round này.
Nếu sau này muốn tinh chỉnh thì mở rộng **xuống dưới/giữa** (ví dụ 0,03–0,3), không lên trên 1.

### KHÔNG chép các số này vào notebook bằng tay

Notebook trong `production/` gắn dataset μ tương ứng; bộ gốc `notebook/` gắn sweep qua
`kernel_sources`. Cell config đọc μ từ
`mu_sweep.json` và kiểm `MU_PROTOCOL` + chữ ký (gồm `scaler_fingerprint`, `data_fingerprint`)
trước khi train. Sửa placeholder không đổi μ hiệu lực vì cell này ghi đè; không tắt cổng kiểm
để chép số tay. Bảng trên chỉ để tra cứu và
đối chiếu. Đã kiểm thật: cổng nhận đúng sweep của kịch bản mình và **từ chối** sweep kịch bản khác.

### Các con số này KHÔNG phải kết quả nghiên cứu

Chúng là điểm trên **validation 2% tách từ train**, sau **4 round**. Kết quả nghiên cứu là
**50 round trên fixed test 10.761.343 dòng**, chưa chạy. Dữ liệu thô đầy đủ:
`review-logs/mu-sweep{,-50client,-100client}-result.json`.

---

## 5c. W&B — theo dõi tiến trình khi đang chạy

Kaggle chỉ publish output **khi kernel dừng**; W&B stream scalar **trong lúc chạy**.
Đã bật từ 2026-09-08 cho các notebook sweep/production.

- Project `tinyproto-fp`, mỗi ứng viên sweep là một run, group `musweep-<scenario>`.
- **Khóa W&B được nhúng thẳng vào notebook** vì Kaggle API push **không** đính được secret.
  Điều này nằm trong ủy quyền thường trực của chủ sở hữu cho notebook `is_private: true` của
  chính họ. **Cái giá:** khóa là **vĩnh viễn** trong lịch sử version riêng tư của Kaggle; chia sẻ
  hoặc publish notebook là làm lộ khóa; xoay khóa đồng nghĩa phải build lại và push lại **mọi**
  notebook mang nó. Muốn gỡ: `.agents/skills/kaggle-training-notebook/references/wandb.md` §1.1 — **revoke ở W&B TRƯỚC**, các bước sau
  chỉ là dọn dẹp.
- `.ipynb` sinh ra **không được commit hay chia sẻ** — đã có trong `.gitignore`.
- Build không nhúng khóa: `python scripts/gen_notebooks.py --no-embed-wandb-key`.
- W&B phải nằm ngoài đường train, nhưng **hiện `.log()` trong driver chưa bắt exception**;
  cần sửa P1 ở §8. Login/init có bắt lỗi chưa đủ bảo đảm training chạy tiếp.
  **Không bao giờ báo cáo một con số từ W&B mà không có artifact đã pull đứng sau.**

---

## 5d. Kế hoạch production song song trên 4 tài khoản (lập 2026-09-08)

**Mục tiêu:** chạy cả ba kịch bản đồng thời để wall-clock bằng kịch bản dài nhất chứ không
phải tổng. Với tốc độ đang dự báo, chuỗi 100 client khoảng 34 h cộng chi phí bàn giao quyết
định thời gian chung; đây không phải một cận dưới hiệu năng đã chứng minh.

### Phân công (tôn trọng yêu cầu: vắt kiệt minhtran0601 + odixe0502 trước)

| Kịch bản | Tài khoản | Quota snapshot | Cần (ước tính) | Session | max_hours trong file thật |
|---|---|---|---|---|---|
| 20 client | `minhtran0601` | 12,85 h | 10,5 h | **1** | 11 |
| 50 client | `odixe0502` | 18,92 h | 17,6 h | **2** | 11 → 7.5 |
| 100 client | `khanhmay0304` → `minhtrit06` | 30 h → 30 h | ~33,9 h tổng | **4** | 11 → 11 → 8 → 11 |

Tổng ~62 h dự báo / 91,8 h quota snapshot, chưa cộng đủ overhead. Quota refresh theo ghi
chép **2026-09-12T00:00:00Z**, review này không đọc lại quota.
**Không cam kết 31+19 hay 16+16+12→6 round.** Code dừng theo thời gian, có dự phòng 15%
round chậm nhất; lịch 100c mới có thể gần 15+15+11+9 theo mô phỏng, không phải số đo production.
Trước mỗi push tiếp theo: đọc quota thật, round đã commit và timing; regenerate riêng session
với ngân sách phù hợp, không đổi `rounds=50`/μ. Người dùng đã nêu rõ đây là việc cần làm.
Ba kịch bản nằm trên ba tài khoản khác nhau nên **không** đụng giới hạn 2 phiên đồng thời.

### Hai điều buộc phải giải quyết trước khi chạy

1. **Sweep μ nằm ở `odixe0502`, production chạy ở tài khoản khác.** Kernel output riêng tư
   không đọc được chéo tài khoản, và đây là vấn đề ngay từ **session 1**, không phải chỉ ở
   ranh giới resume. Giải: 3 dataset nhỏ `odixe0502/tinyproto-fp-mu-{20,50,100}client`
   (`papers/tinyproto-lee-2026/mu-datasets/`), mỗi cái đúng **một** file `mu_sweep.json`.
   Cổng chữ ký R1 **giữ nguyên** — chỉ đổi đường vận chuyển tài liệu, không đổi cách kiểm.
   **Tên file phải đúng `mu_sweep.json`:** cell config glob đúng tên đó; nới thành `mu_sweep*`
   sẽ hút luôn `mu_sweep_progress.json` (cũng có khoá `scenario`) → 2 match → assert nổ.
2. **100 client 33,9 h > 30 h quota/tài khoản** → bắt buộc đúng **một** ranh giới chéo tài
   khoản. Payload bàn giao ~**8 GB** (đo thật: 152 MB weights/round × 50 + 310 MB resume).

### Công cụ đã bổ sung cho việc này

`scripts/gen_notebooks.py` thêm hai cờ (trước đây `kernel_sources` luôn lấy theo `--owner`
nên không trỏ được về chủ sở hữu khác):

- `--mu-dataset OWNER/SLUG` — nhận `{n}`, tự thay 20/50/100. Dùng dataset thay kernel output.
- `--resume-from OWNER/SLUG` — notebook session trước; đồng thời bật `require_resume=True`.
  Bắt buộc đi kèm `--only trainN`: một nguồn resume chỉ thuộc **một** kịch bản, gắn cho cả ba
  thì `find_import_source` sẽ import nhầm run hoặc chết vì lệch fingerprint.

`scripts/validate_notebooks.py` kiểm production tới μ bằng **đúng một** trong hai đường, chặn
sweep của kịch bản khác, chặn >1 nguồn resume, và bắt `require_resume` phải khớp với việc có
gắn session trước hay không. 3 test mới trong `tests/review_regressions.py` (**20 test, pass**).

`max_hours` **không** nằm trong `FINGERPRINT_KEYS` → đổi theo từng session là an toàn.
`mu_value` **có** trong đó → μ buộc phải giống hệt qua mọi session, đúng như mong muốn.

### Mỗi session một notebook riêng (`--session N`)

Kaggle **chưa được xác nhận** là cho phép một notebook gắn chính output của nó làm
`kernel_sources` (phụ thuộc vòng). Không đánh cược 34 h vào giả định đó: `--session N` cho mỗi
phiên một slug riêng `…-sN`, phiên N gắn phiên N-1. Đúng trong cả hai trường hợp.

`run_name` **không** đổi theo slug, nên fingerprint giữ nguyên và `import_previous` vẫn nhận.
Đã kiểm bằng cách diff CFG của s1 và s2: **khác biệt duy nhất là `require_resume`**, và nó
không nằm trong `FINGERPRINT_KEYS`.

### Chuỗi lệnh (chạy sau khi được đồng ý)

```bash
MU='odixe0502/tinyproto-fp-mu-{n}client'
# B0 (odixe0502): tạo 3 dataset μ, rồi NGƯỜI DÙNG share — xem §7
for s in 20client 50client 100client; do
  kaggle datasets create -p papers/tinyproto-lee-2026/mu-datasets/tinyproto-fp-mu-$s; done
# B1: smoke test mỗi tài khoản mới (~5 phút) — chứng minh 2xT4 VÀ quyền đọc dataset
python scripts/gen_remote_validation.py --owner <acct>
# B2: session 1 của ba kịch bản, ba tài khoản, song song
python scripts/gen_notebooks.py --only train20  --owner minhtran0601 --mu-dataset "$MU" --session 1
python scripts/gen_notebooks.py --only train50  --owner odixe0502    --mu-dataset "$MU" --session 1
python scripts/gen_notebooks.py --only train100 --owner khanhmay0304 --mu-dataset "$MU" --session 1
# B3: session tiếp theo — gắn ĐÚNG session liền trước
python scripts/gen_notebooks.py --only train100 --owner khanhmay0304 --mu-dataset "$MU" \
  --session 2 --resume-from khanhmay0304/tinyproto-fp-train-100client-s1
```

Mỗi lần sinh xong **phải** `validate_notebooks.py` trước khi push. Dùng `--out` riêng cho từng
tài khoản: `--only trainN` ghi đè đúng thư mục con đó, sinh liên tiếp sẽ đè metadata của nhau.

### W&B trong production (đã sửa 2026-09-08)

**Lỗi phát hiện khi lập kế hoạch này:** cell launch của production gọi `D.run(cfg, paths, OUT_ROOT)`
**không truyền `wandb_run`** — chỉ notebook sweep mới truyền. Production vì thế login W&B rồi
**không stream gì cả**; muốn theo dõi vài round đầu là không thể. Đã sửa: production tạo run với
`id=cfg["run_name"]`, `resume="allow"`, `group=train-<scenario>`.

Chọn `id` theo **tên run** chứ không theo session: cả 4 session của 100 client nối vào **một**
biểu đồ liên tục, vì `driver.run` log `step=rnd` nên step vẫn tăng đơn điệu qua ranh giới
session. Việc truyền `wandb_run` đã sửa; **còn lỗi `.log()` chưa được bảo vệ trong driver**
(P1, §8), nên chưa được nói mọi lỗi W&B đều bị nuốt.
Đã chạy `notebook_sim` để chắc nhánh `WANDB is None` không làm hỏng gì.

**Cách theo dõi:** xem vài round đầu để xác nhận tiến trình tốt rồi thôi, **không** giữ tác vụ
ngầm dài hạn. Nhớ: `logs/timing_*.json` mới là nguồn có thẩm quyền về thời gian, W&B summary
thì không; và `wandb.Api()` **cache**, phải tạo mới mỗi lần poll.

### Ghép session thành một kết quả duy nhất

`src/ckpt.py::import_previous` đã lo sẵn: copy `weights/protos/confusion/metrics/client_log/
preds/reports/resume/logs`, **marker copy sau cùng**, `validate_integrity` trước khi nhận.
Khi chuỗi thực sự chạy hết, thư mục cuối sẽ giữ đủ artifact 50 round. Chưa chứng minh quỹ đạo
T4 bitwise giống chạy một mạch hoặc quyền đọc output chéo tài khoản. **Chỉ gắn output của session
liền trước** — gắn nhiều session cùng fingerprint sẽ làm `find_import_source` từ chối chọn.

---

## 6. Việc cần làm tiếp

**KHÔNG CÒN GÌ ĐANG CHẠY.** Cả ba kịch bản đã xong, đã ghép, đã verify, đã có `report.md`.

1. **Không sinh/push session nào nữa** cho cả ba kịch bản. Mọi notebook production đã hết vai trò.
2. **Mở khoá sửa `src/driver.py`** (đã hoãn ở §11.2): một session **chạy trọn** vẫn mang
   `logs/stopped_early.json` của session trước, vì `import_previous` copy `logs/` và driver chỉ
   ghi file này khi dừng sớm. Nay chuỗi 100 client đã kết thúc nên sửa không còn phá source
   manifest của run nào đang chạy. Sửa: driver xoá `stopped_early.json` ngay sau import. Sau đó
   `source_manifest.py write` lại.
3. **Nếu chạy thêm bất kỳ cấu hình nào của phương pháp này**, đo `τ_max` ở round 1 trước
   (§12, `scripts/make_report_tables.py::aps_scale`). `τ_max ≥ 1` thì đừng tốn 50 round.
4. Việc tách hai biến bị lẫn (số client vs k) cần ít nhất một run 20 client với k=0,3 hoặc
   50/100 client với k=0,1 — **chưa được yêu cầu**, chỉ ghi để không ai đọc nhầm bảng.
5. Các thư mục `_pull_100client-s{1,2,3,4}` (25,6 GB) đã bị xoá sau khi bản ghép verify
   38.733 / 0; kernel trên Kaggle vẫn còn nếu cần kéo lại. `resume-datasets/` là hardlink,
   không tốn đĩa.
6. Workspace **chưa có Git** — không tạo commit/PR giả.

---

## 7. Việc người dùng cần làm tay

**Hiện tại: không có việc gì.** Bốn dataset VeReMi vốn public; ba dataset μ đã tạo public nên
không phải share tay; cả 4 tài khoản đã có OAuth và đã được kiểm bằng chính server.

Chỉ còn hai việc chỉ người dùng làm được, khi cần:

1. **Dừng một phiên Kaggle đang chạy:** mở URL của đúng tài khoản sở hữu → **Stop session**.
   Agent dừng làm việc **không** làm worker Kaggle dừng. Không thay nguồn bằng notebook rỗng,
   không dùng kernel ID trong metadata làm session ID.
2. ~~Share output notebook riêng tư giữa hai tài khoản~~ — **không còn cần**: bàn giao chéo
   tài khoản đã đi bằng dataset (§10, §12), và không còn phiên nào để chạy.

Lưu ý vận hành:

- Sau mỗi lần đổi tài khoản CLI: **reconnect MCP client** (Claude: `/mcp`). CLI đổi ngay, MCP
  giữ bearer cũ tới khi reconnect. Token MCP tĩnh trong `~/.kaggle/accounts/*.mcp-token` **đã
  hết hạn**; đọc quota bằng `kaggle_account.py quota`.
- **`minhtran0601` có AFPHA.** Không dừng/sửa/ghi đè. Kịch bản 20 client dùng slot còn lại.

---

## 8. Đã thực hiện review P1–P6 và đã push production s1 — 2026-09-08

Review ở [PRODUCTION_REVIEW.md](papers/tinyproto-lee-2026/PRODUCTION_REVIEW.md) (P1–P6) đã được
thực hiện. Trạng thái từng mục:

| Mục | Đã làm | Bằng chứng |
|---|---|---|
| **P1** | `wandb_run.log()` nay nằm trong `try/except`, hỏng thì **tắt telemetry** và train tiếp; `finish()` production trong `finally` cũng được bảo vệ | test `TelemetryNeverStopsTraining`: logger ném lỗi round 1 → vẫn đủ 2 round commit + đủ `timing_*.json`, và chỉ gọi log **một** lần |
| **P3** | Validator kiểm quan hệ chuỗi session: sN phải nối sN−1, cùng scenario, khác slug, không tự gắn chính nó; s1 không được gắn session trước; μ dataset phải **đúng tên kịch bản**, không chỉ đếm `len(ds)>=3` | `SessionChainContract`: 6 biến thể sai + s1-gắn-prev đều bị từ chối; bộ 7 file thật pass |
| **P5** | Ngưỡng chốt **trước** probe: `AMP_MAX_DELTA=0.05`, `MIN_DECISIVE_FRAC=0.50`, `DECISIVE_MARGIN=0.01`. Logic tách thành hàm thuần `probe_verdict` để test được trên CPU | `AmpProbeGate`: phản ví dụ của review (delta=1000, decisive=0) nay **bị chặn**; thiếu coverage bị chặn; probe fp16 thực tế vẫn pass |
| **P6** | `scripts/source_manifest.py` ghim SHA256 của 10 module + generator + validator + 3 JSON μ + digest image; `verify` chạy trước mỗi session | `SourceManifestContract`: đổi 1 module hoặc đổi image đều bị bắt |
| **P2** | **ĐÃ NGHIỆM THU một phần** — xem dưới | smoke test thật trên 2 tài khoản |
| **P4** | max_hours 11/11/8/11 giữ nguyên; đọc quota thật trước mỗi push | quota đo 2026-09-08 dưới đây |

**Regression: 31 test pass**, peak 1.635 MiB, tuần tự qua watchdog, CPU, không Inductor.

### P5 — một phát hiện thêm ngoài review

Ngưỡng cũ `margin > 10*delta` khiến phép kiểm argmax **không bao giờ có thể kích hoạt**: muốn
đảo argmax thì nhiễu phải ≥ margin/2, tức `delta ≥ margin/2`, nên điều kiện `margin > 10*delta
≥ 5*margin` là bất khả. Tập "decisive" vì thế luôn loại sạch mọi hàng bị đảo. Đã đổi sang
**margin cố định** (`DECISIVE_MARGIN=0.01`), độc lập với sai số đo được, nên phép kiểm mới thật
sự có thể bắt lỗi. Có test dựng đúng tình huống đảo argmax trên hàng tách bạch.

### Smoke test 2 × T4 — cả hai tài khoản chưa chứng minh nay ĐÃ chứng minh

| Tài khoản | GPU | Đọc dataset VeReMi | Resume | Kết quả |
|---|---|---|---|---|
| `khanhmay0304` | **2 × Tesla T4 sm_75** | có | có | **335 check, 0 fail** |
| `minhtrit06` | **2 × Tesla T4 sm_75** | có | có | **335 check, 0 fail** |

Log: `review-logs/kaggle-smoke-{khanhmay0304,minhtrit06}.log`. Bài học `khanhngoc0304`
(quota đủ nhưng `device_count()==0`) **không** lặp lại ở hai tài khoản này.

### Bốn dataset VeReMi vốn đã PUBLIC

Kiểm 2026-09-08: cả bốn `veremi-*` đều public → mọi tài khoản mount được, **không cần share tay**.
Chi tiết: [`knowledge/KAGGLE_DATASETS.md`](knowledge/KAGGLE_DATASETS.md). Phần P2 lo về quyền đọc
dataset vì thế đã khép lại. **Còn lại chưa nghiệm thu:** bàn giao output notebook riêng tư
`khanhmay0304/…-s3` → `minhtrit06/…-s4` (chỉ tới lượt ở session 4).

### Ba dataset μ — đã tạo, PUBLIC

`odixe0502/tinyproto-fp-mu-{20,50,100}client`, mỗi cái đúng một file `mu_sweep.json`, đã `ready`.
Người dùng chốt để **public** để không phải share tay (Kaggle không có API thêm collaborator).
Nội dung không chứa credential. **Notebook thì vẫn luôn `is_private: true`** vì nhúng W&B key.

### Quota thật trước khi push (2026-09-08)

| Tài khoản | GPU còn | Dùng cho |
|---|---|---|
| `khanhmay0304` | 29,90 h | 100 client s1–s3 |
| `minhtrit06` | 29,91 h | 100 client s4 |
| `odixe0502` | 18,92 h | 50 client s1–s2 |
| `minhtran0601` | 12,85 h | 20 client s1 |

### ĐÃ PUSH — session 1 của cả ba kịch bản (đang chạy)

| Notebook | Tài khoản | max_hours | Trạng thái 21:25 |
|---|---|---|---|
| `tinyproto-fp-train-20client-s1` | minhtran0601 | 11,0 | **5 round**, khoẻ |
| `tinyproto-fp-train-50client-s1` | odixe0502 | 11,0 | **2 round**, khoẻ |
| `tinyproto-fp-train-100client-s1` | khanhmay0304 | 11,0 | **1 round**, khoẻ |

Cả ba đọc μ từ dataset public, `kernel_sources` rỗng (session 1 không nối gì).

### Theo dõi W&B — kết quả round đầu (2026-09-08, 21:25)

**20 client** — mọi chỉ báo lành:

| round | proto_f1 | clf_f1 | ce | reg | s |
|---:|---:|---:|---:|---:|---:|
| 1 | 0,5963 | 0,4987 | 0,1920 | 0,000000 | 780 |
| 2 | 0,6311 | 0,5222 | 0,0939 | 0,072265 | 739,3 |
| 3 | 0,6268 | 0,5190 | 0,0663 | 0,005475 | 739,4 |
| 4 | 0,6284 | 0,5187 | 0,0531 | 0,002443 | 738,1 |
| 5 | 0,6292 | 0,5184 | 0,0450 | 0,001722 | 738,3 |

**50 client** — hai round đầu cùng hình dạng: ce 0,2120 → 0,0841; reg 0 → 0,0898;
proto_f1 0,4866 → 0,5225; round 1.340 → 1.317,4 s.

**100 client** — round 1: proto_f1 0,4161, clf_f1 0,3397, ce 0,1887, reg 0 (đúng, round 1 chưa
có prototype toàn cục), 2.405,9 s. So với sweep 100c round 4 (validation) 0,4297 thì round 1 ở
0,4161 là đúng hướng.

Ba điều đáng ghi:

1. **`reg` giảm đều, không phân kỳ.** 20c: 0,0723 → 0,0055 → 0,0024 → 0,0017. Đây đúng là hành vi
   mong đợi của k = 0,1, và ngược hẳn với k ≥ 1 trong sweep (bùng lên 13,0 rồi 125.175). Bằng
   chứng trực tiếp rằng winner của giao thức v1 ổn định ở thang 50 round, không chỉ 4 round.
   `reg = 0` ở round 1 là **đúng**, chưa có prototype toàn cục để phạt.
2. **μ chọn trên validation chuyển được sang test cố định.** Sweep 20c round 4 (validation)
   0,6331 so với production round 4 (test cố định) 0,6284; 50c sweep round 4 0,5162 so với
   production round 2 đã 0,5225. Không có dấu hiệu μ chỉ hợp với tập validation.
3. **Thời gian round rất ổn định** — 20c 738–739 s sau round 1; round 1 đắt hơn ~41 s vì warm-up
   `torch.compile`, đúng như đã đo ở sweep.

### Ngân sách suy lại từ số đo THẬT của production (thay cho phép chiếu cũ)

| Kịch bản | s/round đo được | 50 round | Session |
|---|---:|---:|---|
| 20 client | **739** | **10,26 h** | **vừa 1 session** ở max_hours 11 |
| 50 client | **1.317** | **18,3 h** | s1 ~30 round, s2 cần ~7,3 h |
| 100 client | **2.405,9** (round 1, có warm-up) | **~32,9 h** | 4 session, xem dưới |

100c round 1 = **2.405,9 s (40,1 phút)**, đo lúc 21:31. Ngoại suy tuyến tính trước đó đoán
2.280 s — lệch **5,5%**, tức ngoại suy theo số client là xấp xỉ tạm được nhưng **không** thay
được số đo. Trừ warm-up `torch.compile` (20c: +41 s, 50c: +23 s) thì steady ≈ **2.370 s**,
nên 50 round ≈ **32,9 h**.

⚠ **`khanhmay0304` (29,90 h) KHÔNG đủ cho 100 client.** Phân bổ lại theo số đo:

| Session | Tài khoản | max_hours | Round dự kiến |
|---|---|---:|---:|
| s1 | khanhmay0304 | 11,0 | ~16 |
| s2 | khanhmay0304 | 11,0 | ~16 |
| s3 | khanhmay0304 | **7,5** (hạ từ 8,0) | ~10 |
| s4 | minhtrit06 | 11,0 | ~8 (còn lại) |

khanhmay0304 dùng ~29,5/29,90 h; minhtrit06 chỉ cần ~5,3/29,91 h. Đã sửa `PLAN` trong
`scripts/gen_production.py`. Số round mỗi session do **budget gate của driver** quyết định, bảng
này chỉ là dự kiến — đọc marker khi session dừng, đừng suy từ slug.

⚠ **50 client sát ngân sách:** `odixe0502` còn 18,92 h, mà 50 round cần 18,3 h → dư **0,6 h**.
Nếu round chậm đi hoặc setup lâu hơn thì s2 sẽ không đủ và phải mượn tài khoản khác cho vài round
cuối. Theo dõi và quyết định trước khi push s2, đừng giả định vừa.
> **Đã giải quyết (2026-09-09, §9):** s1 làm được 31 round chứ không phải ~30, nên s2 chỉ còn 19
> round ≈ 6,7 h trên 8,06 h quota còn lại. Hết cheo leo. Toàn bộ bảng ngân sách của §5d là **dự
> báo trước khi chạy**; số đo thật nằm ở §9 và phải ưu tiên §9 khi hai bên lệch nhau.

⚠ **Thời gian chờ trước round 1 lớn hơn "setup" đã đo ở sweep.** 20c: push → round 1 lên W&B mất
**~33 phút**, trong đó bản thân round chỉ 13 phút → ~20 phút là **queue của Kaggle + decode dữ
liệu**, thứ mà `timing_*.json` không nhìn thấy (nó chỉ đếm từ khi kernel chạy). Con số "setup
3,6 phút" ở TESTLOG là setup **trong** kernel, không phải độ trễ từ lúc push. Đừng dùng nó để
suy ra khi nào có kết quả.

**Source manifest (P6) thu về 15 file** — chỉ ghim thứ có thể đổi câu trả lời "session sau có
chạy cùng thuật toán không": `src/*.py`, `gen_notebooks.py`, ba JSON μ, digest image.
`gen_production.py` (chỉ owner/`max_hours`) và `validate_notebooks.py` (chỉ kiểm) **không** ghim,
nếu không mọi lần chỉnh ngân sách hợp lệ đều thành báo động giả.

⚠ **Fixture `/tmp/tinyproto_fixture` mất mỗi khi phiên khởi động lại** (nằm ở `/tmp`). 8 test sẽ
lỗi với `expected exactly one 4-client partition root`. Đó **không** phải regression — dựng lại:
`python tests/make_fixture.py /tmp/tinyproto_fixture --rows 2048 --test-rows 1024`.

**Session sau:** `python scripts/gen_production.py --only <scenario>` rồi validate rồi push;
chạy `source_manifest.py verify` trước, và đọc lại quota để chỉnh `max_hours`.


---

## 9. Session 1 đã dừng — kéo về, verify, sinh session 2 (2026-09-09)

### Kết quả pull + verify

| Kịch bản | Tài khoản | Round đã commit | Lý do dừng | Pull | `verify_run.py` |
|---|---|---:|---|---:|---|
| **20 client** | minhtran0601 | **50/50** | chạy hết, **không** có `stopped_early.json` | 2,0 GB | **10.573 check / 0 fail**, `--require-complete` |
| 50 client | odixe0502 | 31/50 | `wall_clock_budget` | 2,5 GB | 13.048 check / 0 fail |
| 100 client | khanhmay0304 | 16/50 | `wall_clock_budget` | 2,8 GB | 12.348 check / 0 fail |

`source_manifest.py verify` **OK, 15 file + runtime image khớp bản đã ghim** → session sau chạy
đúng thuật toán của session 1. Notebook đã thực thi (có output) lưu ở
`papers/tinyproto-lee-2026/notebook/train-{20,50,100}client.executed.s1.ipynb`.

Thư mục làm việc cuối cùng: `papers/tinyproto-lee-2026/runs/`
— `tinyproto_fp_20client/` là **run hoàn chỉnh đã ghép** (`merge_sessions.py`, 50 round liền
mạch, verify lại sau khi ghép vẫn **10.573/0**, nên phép ghép không mất gì); `_pull_50client-s1/`
và `_pull_100client-s1/` giữ nguyên để đối chiếu **byte-identical** với pull của session sau —
đó chính là bằng chứng resume hoạt động, đừng xoá trước khi ghép.

### ⚠ Phát hiện chính: k=0,3 đạt đỉnh sớm rồi suy giảm

| Kịch bản | k | proto F1 đỉnh | tại round | proto F1 cuối | Δ | reg chạm đáy | reg cuối |
|---|---:|---:|---:|---:|---:|---|---:|
| 20 client | 0,1 | 0,6350 | 13 | 0,6212 (r50) | −0,0138 | giảm đều tới r50 | 0,000348 |
| 50 client | 0,3 | 0,5249 | **4** | 0,4419 (r31) | **−0,0830** | r8 = 0,010943 | **0,045055** |
| 100 client | 0,3 | 0,4474 | **2** | 0,4088 (r16) | **−0,0386** | r9 = 0,010512 | 0,013400 |

Không NaN, không bùng nổ, `grad_norm` cỡ 0,13–0,54 (so với 125.175 của k=3 trong sweep). Đây là
**suy giảm trơn**, không phải hỏng. Hình dạng tách **đúng theo k**, không theo số client:
k=0,1 thì `reg` và `grad_norm` giảm đơn điệu tới round 50; k=0,3 thì cả hai **quay đầu tăng**
trong khi `train_loss` đã chạm sàn — tức số hạng CE bão hoà và regularizer APS trở thành gradient
chi phối, kéo prototype cục bộ về μ·global và làm hỏng chính phép phân loại theo prototype.

**Nguyên nhân là chân trời của sweep, không phải μ sai.** `MU_PROTOCOL` v1 chọn theo round 4;
ở round 4 thì 50 client đang **ở đúng đỉnh** của nó. Sweep 4 round không thể nhìn thấy khúc quay
đầu ở round 8–9. §5b đã ghi trước điều này ("chưa chứng minh đây là k tối ưu sau 50 round") —
nay đã đo được.

**Quyết định của người dùng (2026-09-09): chạy tiếp đủ 50 round, GIỮ NGUYÊN μ.** Lý do: đổi k lúc
này là chọn siêu tham số dựa trên quỹ đạo **test set**, làm hỏng toàn bộ tính hợp lệ của sweep.
Giao thức đã đóng băng **trước** khi chạy thì phải chạy hết. Suy giảm này tự nó là kết quả đáng
báo cáo, và vì giữ metric **từng round** nên phân tích "best round" làm được sau mà không chạy lại.
Khi viết báo cáo: **nêu cả round đỉnh lẫn round 50**, đừng chỉ nêu round 50.

### 20 client — kết quả nghiên cứu đầu tiên (round 50, fixed test 10.761.343 dòng)

Quy tắc `proto` (Eq. 12) — mean qua 20 client:

| metric | mean | std | min | max | pooled |
|---|---:|---:|---:|---:|---:|
| accuracy | 0,5639 | 0,0752 | 0,4105 | 0,6806 | 0,5639 |
| precision_macro | 0,6852 | 0,0315 | 0,6274 | 0,7528 | 0,6533 |
| recall_macro | 0,6349 | 0,0409 | 0,5222 | 0,7046 | 0,6349 |
| **f1_macro** | **0,6212** | 0,0413 | 0,5428 | 0,7049 | 0,6168 |
| precision_micro | 0,5639 | 0,0752 | 0,4105 | 0,6806 | 0,5639 |
| recall_micro | 0,5639 | 0,0752 | 0,4105 | 0,6806 | 0,5639 |
| f1_micro | 0,5639 | 0,0752 | 0,4105 | 0,6806 | 0,5639 |
| precision_weighted | 0,7283 | 0,0432 | 0,6159 | 0,7940 | 0,7096 |
| recall_weighted | 0,5639 | 0,0752 | 0,4105 | 0,6806 | 0,5639 |
| f1_weighted | 0,5972 | 0,0727 | 0,4682 | 0,6988 | 0,6057 |

`clf` (argmax): mean f1_macro **0,5120**, pooled 0,5369 → `proto` **cao hơn** `clf` 0,109 macro-F1,
đúng hướng bài báo mong đợi. Chi phí truyền tin: **32.000 tham số/round** so với 163.840 của
FedProto dày → nén **5,12×** (K_i = 16 lớp ở mọi client).

### Ngân sách suy lại từ `logs/timing_*.json` (nguồn có thẩm quyền)

| Kịch bản | s/round (mean 8 round cuối) | worst round | overhead trước round 1 | elapsed session 1 | quota đã tiêu |
|---|---:|---:|---:|---:|---:|
| 20 client | 788,5 | 816,3 | 172,9 s | 10,589 h | 10,60 h |
| 50 client | 1.234,7 | 1.343,1 | 209,8 s | 10,851 h | 10,86 h |
| 100 client | 2.336,3 | 2.411,9 | 188,4 s | 10,418 h | 10,42 h |

`elapsed` khớp quota tiêu thụ tới 0,01 h ở cả ba → **`max_hours` là thước đo quota đáng tin**.
Budget gate: dừng khi `elapsed + 1,15 × worst_round ≥ max_hours × 3600`, và `worst_round` được
mồi lại từ `timing_*.json` **đã import**, nên session 2 không bao giờ khởi động bằng phỏng đoán 60 s.

Quota đo lại 2026-09-09: `minhtrit06` 29,91 h · `khanhmay0304` 19,48 h · `odixe0502` 8,06 h ·
`minhtran0601` 2,25 h. Refresh 2026-09-12T00:00:00Z.

`PLAN` trong `gen_production.py` sửa theo số đo: **50c s2 7,5 → 7,9** (đủ chỗ cho cả 19 round kể
cả khi round chậm hơn 13%); **100c s3 7,5 → 8,0** (ở 7,5 thì biên chỉ 66 s, một chút overhead
import là mất một round sang s4 — 8,0 cùng số round nhưng hết cheo leo). Dự kiến 100c:
16 + 16 + 11 + 7 = 50.

### Hai lỗi thật đã sửa trong phiên này

1. **`merge_sessions.py` của skill sẽ ghép SAI cho dự án này.** `ROUND_DIRS` chỉ liệt kê
   `checkpoints/metrics/preds/confusion/reports`, mà dự án này để checkpoint ở **`weights/`** và
   còn có `protos/`, `client_log/`, `complete/`, `resume/`. Ghép xong sẽ ra một thư mục run
   **không có trọng số** mà vẫn báo thành công. Đã bổ sung đủ tên. Lỗi thứ hai che lỗi thứ nhất:
   phép kiểm liền mạch dùng `range(have[-1]+1)` nên luôn báo "gap ở round 0" với run đánh số từ 1
   → nó abort trước khi kịp báo thành công giả. Đã đổi sang `range(have[0], have[-1]+1)` cộng một
   phép kiểm riêng rằng round đầu phải là 0 hoặc 1. Sửa ở **cả hai** bản `.agents/` và `.claude/`.
2. **`.gitignore` thiếu `production/**/*.ipynb`.** Cả 7 notebook session đều mang
   `_INLINE_WANDB_KEY` trong source; chỉ `notebook/` và `validation/` được phủ. Workspace chưa có
   Git nên chưa lộ, nhưng `git init` là lộ ngay. Đã thêm, kèm `runs/` (vài GB trọng số).
   Bản `.executed.*.ipynb` kéo về **cũng** mang key — chúng nằm trong `notebook/` nên đã được phủ.

### Notebook session 2 — đã sinh, đã validate, CHƯA push

`gen_production.py --only 50client` và `--only 100client` → **7 notebook, 0 problem**.

| Notebook | Tài khoản | `kernel_sources` | max_hours | `require_resume` |
|---|---|---|---:|---|
| `…-50client-s2` | odixe0502 | `odixe0502/…-50client-s1` | 7,9 | True |
| `…-100client-s2` | khanhmay0304 | `khanhmay0304/…-100client-s1` | 11,0 | True |
| `…-100client-s3` | khanhmay0304 | `khanhmay0304/…-100client-s2` | 8,0 | True |
| `…-100client-s4` | minhtrit06 | `khanhmay0304/…-100client-s3` | 11,0 | True |

Tất cả `is_private: True`, GPU bật, μ đọc từ đúng dataset của kịch bản mình.
**Regression: 31 test pass**, peak 1.594 MiB.

---

## 10. P2 đã có câu trả lời: `kernel_sources` CHÉO TÀI KHOẢN KHÔNG HOẠT ĐỘNG (2026-09-09)

**Đo thật, không suy luận.** Notebook probe `minhtrit06/tinyproto-fp-handoff-probe` (CPU, không
tốn quota GPU) gắn `kernel_sources: ["khanhmay0304/tinyproto-fp-train-100client-s1"]`.

Kaggle phản ứng theo cách tệ nhất có thể:

```
The following are not valid kernel sources and could not be added to the kernel:
  ['khanhmay0304/tinyproto-fp-train-100client-s1']
Kernel version 1 successfully pushed.
```

Nó **cảnh báo một dòng, rồi báo push THÀNH CÔNG**, và kernel khởi động với `/kaggle/input`
**RỖNG HOÀN TOÀN** — probe in ra danh sách rỗng rồi thoát. Đây là kiểu hỏng nguy hiểm nhất: không
có exception lúc push, không có gì để một validator chỉ nhìn exit code bắt được. Nếu để tới
session 4 mới phát hiện thì đã tốn ~30 h quota cho ba session trước đó.

`require_resume=True` **sẽ** bắt được (driver `SystemExit` khi không tìm thấy round nào), nên
không mất dữ liệu — nhưng vẫn mất thời gian xếp hàng và một version notebook.

### Kế hoạch 100 client đã đổi hình

Ranh giới tài khoản là **không tránh được** (khanhmay0304 còn 19,48 h, mà 34 round cần ~22,1 h),
nên nó được **dời tới điểm sớm nhất và rẻ nhất**: `s1 → s2`, nơi payload chỉ là 16 round
(2,5 GB nén, **đã có sẵn trên đĩa local** từ lần pull s1) thay vì 43 round ở `s3 → s4`.

| Session | Tài khoản | max_hours | Nối bằng |
|---|---|---:|---|
| s1 | khanhmay0304 | 11,0 | — (xong, 16 round) |
| **s2** | **minhtrit06** | 11,0 | **dataset** `minhtrit06/tinyproto-fp-resume-100client-r16` |
| s3 | minhtrit06 | 11,0 | kernel output của s2 (cùng tài khoản) |
| s4 | minhtrit06 | 11,0 | kernel output của s3 (cùng tài khoản) |

minhtrit06 cần ~23 h trên 29,91 h. **Quota của khanhmay0304 (19,48 h) không dùng nữa** — quota
không phải ràng buộc, số ranh giới tài khoản mới là.

**Dataset resume thuộc sở hữu của chính tài khoản sẽ chạy nó**, nên không cần share, không cần
public: minhtrit06 vừa sở hữu dataset vừa sở hữu notebook.

### Hai cái bẫy khi tạo dataset resume

1. **`kaggle datasets create` mặc định BỎ QUA thư mục** (`--dir-mode skip`). Phải `-r zip`.
   Kaggle giải nén lại phía nó, nhưng **cắt mất một cấp**: local `runs/tinyproto_fp_100client/…`
   thành `tinyproto_fp_100client/…` trên dataset. Không sao — `find_import_source` dùng
   `rglob("config.json")` rồi lọc theo `run_name` + fingerprint, không cần thư mục `runs/`.
2. **Mặc định nó CHUYỂN ĐỔI file tabular sang CSV.** `client_log/round_*.csv` và
   `metrics/history.csv` sẽ bị viết lại → sha256 đổi → `validate_integrity` fail giữa chừng.
   Phải truyền `-t/--keep-tabular`. Đã đối chiếu kích thước sau khi upload: `config.json` 46.404,
   `client_log/round_001.csv` 7.135, `weights/round_016.pt` 159.350.497 — **khớp từng byte**.

### Công cụ đã bổ sung

- `gen_notebooks.py --resume-dataset OWNER/SLUG` — đường resume thứ hai, đặt `require_resume`
  giống hệt, không thêm `kernel_sources`. Không cho dùng cùng lúc với `--resume-from`.
- `validate_notebooks.py`: chấp nhận **đúng một** trong hai đường; và **chặn thẳng** trường hợp
  `kernel_sources` có chủ khác chủ notebook. Cấu hình cũ (s3 khanhmay0304 → s4 minhtrit06) nay
  bị từ chối với đúng lý do.
- `gen_production.py`: `PLAN` mang thêm `resume_dataset`, và **thoát 2** nếu một session định nối
  kernel output của tài khoản khác mà không khai dataset.
- `tests/review_regressions.py`: **34 test pass**. `CrossAccountResumeContract` trước đây
  **khẳng định đúng cái hành vi vừa bị bác bỏ** — đã viết lại theo số đo, cộng 3 test mới
  (đường dataset hợp lệ; kernel source chéo tài khoản bị chặn; hai đường loại trừ nhau).

### `gen_notebooks.py` đổi → manifest đã ghim lại, kèm bằng chứng

`gen_notebooks.py` nằm trong source manifest nên `verify` báo `CHANGED`. Bằng chứng thuật toán
**không** đổi: diff notebook s1 (đã chạy 16 round) với s2 — **25/25 cell, chỉ một cell khác, chỉ
một dòng**: `"require_resume": False → True`. Cả **10 cell `%%writefile`** chứa `src/` giống nhau
**byte-for-byte**. `require_resume` không nằm trong `FINGERPRINT_KEYS` nên fingerprint giữ nguyên.
Ghim lại sau khi có bằng chứng đó, không phải trước.

### NGHIỆM THU: cả hai session 2 đã resume ĐÚNG round, không chạy lại từ đầu (2026-09-09)

| Kịch bản | Đường resume | round cuối s1 | round đầu s2 | s/round |
|---|---|---|---|---:|
| 50 client | kernel output, **cùng** tài khoản | r31 proto 0,4419 · reg 0,045055 | **r32** proto 0,4427 · reg 0,049703 | 1.473 |
| 100 client | **dataset**, **chéo** tài khoản | r16 proto 0,4088 · reg 0,013400 | **r17** proto 0,4061 · reg 0,014246 | 2.516 |

**Đây mới là bằng chứng resume đúng, không phải việc kernel chạy được.** Nếu nó khởi động lại từ
đầu thì `reg` sẽ bằng **0** (round 1 chưa có prototype toàn cục) và proto F1 sẽ nhảy về vùng round
đầu (50c ~0,487; 100c ~0,416). Cả hai đều **nối liền quỹ đạo s1** ở cả ba đại lượng → trọng số,
prototype và trạng thái optimizer đều được khôi phục nguyên vẹn.

Round đầu mỗi session đắt hơn vì warm-up `torch.compile` trên worker mới: 50c +238 s, 100c +180 s,
đúng cỡ đã thấy ở round 1 của s1 (50c 1.340 s, 100c 2.412 s).

### Chiếu ngân sách từ số đo của chính session 2

| Kịch bản | overhead | s/round | dừng ở round | elapsed | quota tài khoản |
|---|---:|---:|---:|---:|---:|
| 50 client | 230 s | 1.387 | **50 — XONG TRỌN VẸN trong s2** | 7,41 h | 8,06 h |
| 100 client | 232 s | 2.336 | **32** | 10,50 h | — |

**50 client không cần session 3.** Biên tới ngưỡng break ở round 49 là ~1.455 s, tức khoảng một
round; chỉ hụt nếu round chậm hơn ~1.490 s, mà phương sai đang rất nhỏ (1.384/1.388/1.388).

100 client còn lại: **s3 rounds 33–48** (~16 round, 10,5 h), **s4 rounds 49–50** (~1,6 h). Cả hai
trên `minhtrit06`, nối bằng kernel output cùng tài khoản. Tổng minhtrit06 ~22,6 h / 29,91 h.

**Overhead import: hai đường NHANH NHƯ NHAU.** Đo bằng `_timestamp` của W&B trừ `startedAt`:
50c (kernel output) **230 s** trước round 32; 100c (**dataset 2,5 GB, chéo tài khoản**) **232 s**
trước round 17. Đường dataset **không** đắt hơn — nó chỉ *có vẻ* chậm vì history của W&B trễ vài
chục phút với run vừa resume (04:18 kernel khởi động, round 17 xong lúc 05:04 theo `_timestamp`,
nhưng API vẫn trả `last_step=16` mãi tới ~05:45). **Đừng suy ra thời gian chạy từ lúc W&B hiện
số**; lấy từ `_timestamp`, và sau khi pull thì lấy từ `logs/timing_*.json`.

`session_start` được đặt **trước** bước import nên phần này **có** bị tính vào `max_hours` — an
toàn về ngân sách.

**Round của session sau chậm hơn session đầu.** 50c: s1 ổn định 1.235 s, s2 ổn định **1.387 s**
(+12%). Ngân sách phải tính bằng số của **chính session đang chạy**, không phải của s1.

### Bài học vận hành: KHÔNG chẩn đoán một kernel còn sống hay không bằng W&B

Lúc 04:14 W&B báo `tinyproto_fp_50client` là **`state=crashed`**, heartbeat đứng ở 03:57:05, tức
2 giây sau `startedAt` — trông y hệt một tiến trình chết ngay sau `wandb.init()`. Thực tế
`kaggle kernels status` trả về **RUNNING**: kernel đang trong giai đoạn import 2,5 GB và decode,
chưa tới round nào để log. **Trạng thái sống/chết lấy từ Kaggle; W&B chỉ để đọc chỉ số.**

---

## 11. Session 2 đã xong — 50 client HOÀN THÀNH, 100 client tới round 32 (2026-09-10)

### Kết quả

| Kịch bản | Session 2 | Lý do dừng | `verify_run.py` |
|---|---|---|---|
| 50 client | round 32→**50** | chạy hết | pull: **21.133 / 0**; sau khi ghép: **21.133 / 0** |
| 100 client | round 17→**32** | `wall_clock_budget` | đang pull |

Dự báo trước khi chạy khớp gần như tuyệt đối: 50c "xong trọn 50 round, 7,41 h" → thực 7,40 h
quota; 100c "dừng ở round 32" → thực **32**.

**Phép ghép chính là bằng chứng resume.** `merge_sessions.py` từ chối ghép nếu bất kỳ artifact
nào xuất hiện ở cả hai pull mà **khác byte** — nghĩa là session sau đã train lại thay vì tiếp
tục. Ghép 50 client qua **31 round chồng lấn** không báo một khác biệt nào:
`new rounds per session: [31, 19]`.

### 50 client — kết quả round 50 (fixed test 10.761.343 dòng)

`proto` (Eq. 12), mean qua 50 client:

| metric | mean | std | min | max | pooled |
|---|---:|---:|---:|---:|---:|
| accuracy | 0,4062 | 0,0865 | 0,2184 | 0,5342 | 0,4062 |
| precision_macro | 0,4728 | 0,0625 | 0,3542 | 0,6107 | 0,4120 |
| recall_macro | 0,4557 | 0,0623 | 0,3451 | 0,5750 | 0,4557 |
| **f1_macro** | **0,4018** | 0,0606 | 0,2967 | 0,5117 | 0,3859 |
| precision_micro / recall_micro / f1_micro | 0,4062 | 0,0865 | 0,2184 | 0,5342 | 0,4062 |
| precision_weighted | 0,6180 | 0,0464 | 0,4468 | 0,7144 | 0,6004 |
| recall_weighted | 0,4062 | 0,0865 | 0,2184 | 0,5342 | 0,4062 |
| f1_weighted | 0,4321 | 0,0894 | 0,2316 | 0,5758 | 0,4553 |

`clf`: mean f1_macro **0,3984**, pooled 0,4198. Nén truyền tin **5,12×**
(79.700 vs 408.064 tham số/round; K_i trung bình 15,88).

### ⚠ ĐÍNH CHÍNH cách mô tả hiện tượng k=0,3: ở round 50 đây là PHÂN KỲ

Ở round 31 tôi mô tả là "suy giảm trơn, không phải hỏng". Với dữ liệu tới round 50 thì **mô tả đó
không còn đúng**:

| | round đỉnh | round 50 | tỉ lệ |
|---|---:|---:|---:|
| proto f1_macro | 0,5249 (r4) | 0,4018 | **−23%** |
| reg loss | 0,010943 (đáy, r8) | **0,489916** | **45×** |
| grad_norm | 0,2588 (đáy, r16) | **4,5224** | **17×** |
| train_loss | 0,008782 (đáy, r29) | 0,017222 | **2,0×** |

**`train_loss` đi lên là dấu hiệu phân kỳ**, không phải đánh đổi. Đây đúng là kiểu hỏng sweep đã
thấy ở k ≥ 1, chỉ đến muộn hơn nhiều. Không NaN, không bùng nổ kiểu 125.175 — nhưng regularizer
APS đã chiếm quyền chi phối hàm mục tiêu.

**Điểm sắc nhất: `clf` gần như KHÔNG ĐỔI suốt 50 round** (0,4051 ở r1 → 0,3984 ở r50), trong khi
`proto` rơi 0,5249 → 0,4018. APS **không** phá mô hình; nó phá **riêng hình học prototype**.
Lợi thế của quy tắc Eq. 12 so với argmax bị xoá sạch:

| Kịch bản | k | proto − clf (f1_macro, round 50) |
|---|---:|---:|
| 20 client | 0,1 | **+0,1092** |
| 50 client | 0,3 | **+0,0034** |

100 client (k=0,3) đang đi đúng đường: reg 0,0134 (r16) → 0,0659 (r32), grad_norm 0,256 → 0,741,
train_loss chạm đáy r24 rồi tăng. Ngoại suy nhịp hiện tại: round 50 sẽ có reg ~0,65, proto ~0,33.

Quyết định giữ nguyên μ và chạy hết 50 round **vẫn đúng** (giao thức đóng băng trước khi chạy;
đổi k theo quỹ đạo test set sẽ phá tính hợp lệ của sweep). Nhưng khi báo cáo: **kết quả round 50
của 50/100 client là kết quả của một run đã phân kỳ** — phải nêu kèm round đỉnh.

### Ba lỗi vận hành gặp trong lần pull này

1. **`kaggle kernels output` báo thành công khi thiếu file, và để lại file 0 byte.** Lần pull đầu
   của 50c s2: thiếu hẳn weights 36–50, **và** `round_035.pt` là **0 byte**. Đếm theo tên file ra
   "35/50" và **bỏ qua** file hỏng. Phải đếm `size > 0`. Máy người dùng lỗi giữa chừng lần pull
   thứ hai để lại thêm một file 0 byte nữa. Cách chữa: `find <pull> -type f -size 0 -delete` rồi
   pull lại với `-o --file-pattern '<regex>'` (thiếu `-o` thì CLI bỏ qua file nó tưởng đã có).
2. **Thư mục run HOÀN CHỈNH vẫn mang `stopped_early.json` của session TRƯỚC.** `import_previous`
   copy cả `logs/`, mà driver chỉ ghi file này khi dừng sớm → session chạy trọn 50 round không
   ghi đè nó. Thực tế: `_pull_50client-s2` có 50/50 round nhưng file ghi `last_round: 31`.
   **Chỉ `verify_run.py --require-complete` mới là câu trả lời.** *Không sửa `src/driver.py` lúc
   này* — chuỗi 100 client còn s3/s4 và source manifest tồn tại chính để bảo đảm chúng chạy cùng
   một mã. Sửa sau khi chuỗi kết thúc.
3. **`merge_sessions.py` ghép `resume/` theo hợp là SAI.** `resume/` là trạng thái cuốn chiếu:
   driver xoá blob của mọi round đã bị thay thế, nên run thật chỉ giữ blob của round cuối. Ghép
   hợp tạo ra `round_031.*` nằm cạnh `round_050.*` — thứ không run nào sinh ra được — và
   `verify_run.py` bắt đúng: `superseded resume blobs were not pruned`. Đã sửa: chỉ lấy `resume/`
   của session **cuối**. Ghép lại rồi verify: **21.133 / 0**.

### Ngân sách 100 client s3/s4

Đo từ s2: steady **2.345 s/round**, warm-up round đầu 2.516 s, overhead 232 s. 18 round còn lại
**không** vừa một session:

| max_hours | round cuối đạt được | elapsed |
|---:|---:|---:|
| 11,0 | 48 | 10,54 h |
| 11,5 | **49** | 11,19 h |
| 11,9 | 49 | 11,19 h |
| 12,0 | 50 | 11,84 h |

Chỉ `12,0` mới tới round 50, mà đó **đúng bằng trần nền tảng 12 h** — không còn biên để driver
dừng sạch trước khi Kaggle giết kernel, và cú giết cứng có thể mang theo cả output session. Vì
vậy: **s3 = 11,5 (round 33–49), s4 = 3,0 (round 50)**. minhtrit06 dùng ~11,95 h / 19,33 h.

---

## 12. Session 3 + 4 xong — 100 client HOÀN THÀNH, cơ chế phân kỳ, báo cáo (2026-09-11)

### Kết quả

| Phiên | Tài khoản | Round | Lý do dừng | `verify_run.py` |
|---|---|---|---|---|
| s3 | `minhtrit06` | 33→**49** | `wall_clock_budget` (11,05 h / 11,5) | **37.758 / 0** |
| s4 | `minhtran0601` | **50** | chạy hết | **38.733 / 0** `--require-complete` |
| ghép s1+s2+s3+s4 | — | 1–50 | — | **38.733 / 0** `--require-complete` |

Dự báo "s3 dừng ở round 49" đúng. s4 tiêu **0,80 h** quota (dự báo 1,1 h; setup+import 313 s,
round 50 là 2.503 s vì có lưu 2,15 GB dự đoán).

**Phép ghép:** `new rounds per session: [16, 16, 17, 1]`, **97 round chồng lấn** (16+32+49)
không một khác biệt byte; `history.csv` khớp trên mọi overlap. Nguồn gốc từng round ở
`runs/tinyproto_fp_100client/logs/sessions.json`.

**Cả 4 ranh giới phiên đều qua phép thử resume** (`reg` ≠ 0 và nối tiếp quỹ đạo ở round đầu
phiên mới; một run khởi động lại bắt buộc có `reg = 0`):

| ranh giới | nối bằng | round trước | round đầu phiên mới |
|---|---|---|---|
| s1→s2 | dataset | r16 `reg=0,013400` | r17 `reg=0,014246` |
| s2→s3 | kernel output | r32 `0,065881` | r33 `0,075387` |
| s3→s4 | dataset | r49 `0,927422` | r50 `1,097369` |

### s4 chạy trên `minhtran0601` thay vì `minhtrit06` — quyết định của người dùng

Tôi đã đọc quota thật: `minhtrit06` còn **5,35 h** (đủ thừa, nối bằng kernel output không cần
upload). Người dùng vẫn chọn `minhtran0601` (2,13 h). Chi phí: upload dataset **7,09 GB / 11
phút**, **353/353 file khớp kích thước từng byte** (`-r zip -t`), `max_hours=2,0` vì cổng ngân
sách trước round 50 cần `> 0,10 h + 1,15 × 2.544 s = 0,92 h`. PLAN trong `gen_production.py`
đã ghi lý do. Quota `minhtran0601` sau s4: 1,33 h. Không đụng AFPHA.

### Bẫy pull lặp lại y hệt 50c s2 — lần thứ hai, nên coi là hành vi mặc định của CLI

`kaggle kernels output` cho s4 **exit 0** với weights 1–5 đủ, **`round_006.pt` 0 byte**, 7–50
vắng mặt, mọi thư mục khác đủ 50 round. Chữa như §11.1: xoá file 0 byte, pull lại với
`-o --file-pattern 'weights/round_0(0[6-9]|[1-4][0-9]|50)\.pt$' --page-size 200` → 50/50.
**Quy trình pull chuẩn từ nay: pull → đếm `size > 0` theo từng thư mục → xoá 0 byte → pull lại
phần thiếu → verify.** Đừng tin exit code.

### Cơ chế phân kỳ đã tìm ra và kiểm chứng — ĐÂY LÀ KẾT QUẢ KHOA HỌC CHÍNH

Chi tiết đầy đủ ở `report.md` §7. Tóm tắt để không ai phải đọc lại từ đầu:

APS kéo `ĉ_L[i,j]` về `μ ĉ_G[j]`, mà `ĉ_G[j]` **được dựng từ chính `ĉ_L`** (Eq. 10). Định nghĩa
`τ_j = ‖μ ĉ_G[j] ⊙ m_j‖ / mean_i ‖ĉ_L[i,j] ⊙ m_j‖` đo ở **round 1** (round không regularization,
nên prototype giống hệt nhau với mọi k ⇒ τ tỉ lệ **chính xác** với k). `τ > 1` là vòng lặp
dương không có điểm bất động hữu hạn.

| | τ_max (r1) | lớp τ>1 | norm prototype hai lớp đó r1→r50 | 14 lớp còn lại |
|---|---:|---|---|---|
| 20c (k=0,1) | 0,420 | 0 | **co** 0,42× / 0,41× | 0,42–0,65× |
| 50c (k=0,3) | 1,283 | 2 (`benign`, `trafficCongestionSybil`) | **20,9× / 17,6×** | 0,75–4,1× |
| 100c (k=0,3) | 1,271 | 2 (cùng hai lớp) | **31,7× / 27,1×** | 0,95–2,0× |

`τ ≠ k`: μ là **một** vô hướng dùng chung, còn `ĉ_G[j] ∝ n̄_j`, nên hệ số hiệu dụng là
`k · n̄_j / n̄_all`, trải 41:1 trên VeReMi. Sweep **đã thấy** chế độ phân kỳ ở k=1,0 và k=3,0
(reg r4 lớn hơn nhóm ổn định 7 bậc) — chỉ không thấy phân kỳ **chậm** ở k=0,3 (τ=1,27, cần
8–9 round mới đảo chiều). **20 client ở k=0,3 sẽ có τ=1,260 → cũng phân kỳ.** Nó thoát nhờ sweep
chọn k=0,1 hơn k=0,3 đúng **0,0021** f1_macro. ⇒ số client và k **bị lẫn**, biến giải thích là k.

Hàm đo: `scripts/make_report_tables.py::aps_scale`. Đo được sau **một** round, trước khi tốn GPU.

### 100 client — round 50 (fixed test 10.761.343 dòng, mean qua 100 client)

proto f1_macro **0,3435** (đỉnh 0,4474 ở r2, **−23,2%**), clf 0,3289, lợi thế proto−clf
+0,0146 (đỉnh +0,0945). reg 0,010512 (r9) → **1,097369** (**104×**), grad_norm 0,2514 (r14) →
**7,2834** (**29×**), train_loss 0,006650 (r24) → 0,017671 (2,7×). Nén truyền tin 5,12×
(159.200 vs 815.104 tham số/round).

### Báo cáo

`report.md` (gốc): §1 phạm vi, §2 phương pháp, §3 bảng deviation, §4 thiết lập, §5 giao thức
đánh giá, §6 bảng 50 round × 3 kịch bản × 2 quy tắc + per-client tại round đỉnh/50, §7 cơ chế τ,
§8 truyền tin, §9 hạn chế, §10 artifact + chuỗi 7 phiên. **Không con số nào gõ tay**: bảng
sinh bởi `make_report_tables.py`, script assert mean tính lại từ `per_client` khớp
`aggregate.mean_over_clients` tới 1e-9, và bảng verify **chạy `verify_run.py` thật** lúc build.
Phần chữ dùng dấu phẩy thập phân, bảng sinh dùng dấu chấm (khớp CSV) — có ghi chú trong §6.2.

### Công cụ thêm trong phiên này

- `scripts/make_report_tables.py` — CSV per-client/mean, `summary.json`, 23 bảng markdown, splice
  idempotent vào marker `<!-- BEGIN TABLE x -->…<!-- END TABLE x -->`, `aps_scale` (τ).
- `gen_production.py` PLAN: s4 100client → `minhtran0601`, `max_hours=2,0`, dataset
  `minhtran0601/tinyproto-fp-resume-100client-r49`.
- Fixture `/tmp/tinyproto_fixture` mất lần nữa sau crash → 8 test lỗi; dựng lại → **34/34**.
