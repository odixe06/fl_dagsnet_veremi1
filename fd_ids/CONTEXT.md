# CONTEXT — FD-IDS trên VeReMi NextGen

> **📌 Note 2026-09-13 (chủ dự án, ghi từ phiên `~/nckh/pfedes`): thống nhất MỘT lịch learning rate cho mọi dự án anh em.**
> Lịch: AdamW, **cosine theo round** `lr(t) = lr_min + (lr − lr_min)/2·(1 + cos(π(t−1)/(T−1)))`, **lr = 1e-3, lr_min = 1e-5, T = 50**,
> hằng trong một round (công thức `afpha`; bản tham chiếu `~/nckh/pfedes/papers/pfedes-yi-2025/proj/pfedes.py::lr_at`).
> pFedES đã chạy lại với lịch này (kernel `*-cos`, 13-09). **Dự án này CHƯA sinh/push notebook mới** — chỉ làm khi chủ dự án yêu cầu;
> quota tuần 13→19-09 đã dành cho pFedES (~92 h). Chỗ cần sửa khi làm: `scripts/gen_notebook.py` (`lr`, Adam hằng 1e-3, wd 0) + `proj/` (optimizer trong worker) + `scripts/validate_notebooks.py` + `proj/ckpt.py::FINGERPRINT_KEYS` (thêm `lr_schedule`, `lr_min`, `rounds`).
> Lý do và bằng chứng (đỉnh sớm sau ~2 epoch local, cosine chặn đà trôi nhưng không kéo lại đỉnh): `~/nckh/pfedes/CONTEXT.md` §12.

Bối cảnh dùng lại qua nhiều phiên làm việc. Đọc file này **trước** khi làm gì khác.
Cập nhật lần cuối: **2026-09-11** (phiên 3, ~10:05Z).

**Trạng thái 2026-09-11 ~10:05Z — CẢ BA kịch bản ✅ 50/50 XONG, verify pass mode full, mỗi
kịch bản là MỘT run 1..50 ở `papers/fd-ids-2025/runs/fdids_{20c,50c,100c}_v2/` (20c và 100c
merge từ hai phiên); `report.md` HOÀN TẤT (649 dòng, 10 hình, cổng độc lập 0 lệch). Không còn
kernel nào chạy.** Đọc **§15 trước tiên**; §15.7 là phiên này, §15.6 việc còn lại (chỉ còn mục
tuỳ chọn). §14 là phiên trước.

Phiên 1 (v3): 20c **42**/50 (6,15 h), 50c **50/50 ✅** (7,44 h), 100c **25**/50 (5,86 h) — cả ba
`verify_run` **pass mode full**, output đã kéo về `papers/fd-ids-2025/runs/pulls/`. Hai run
dở được tiếp nối bằng cách upload cây run làm dataset của tài khoản sẽ chạy (vì
`kernel_sources` không qua được tài khoản khác), probe CPU chứng minh cổng resume import đúng
42/25 round rồi mới push GPU. Đường này nay là **thủ tục chuẩn** (§10.3).

Ba run trước chạy **eager** suốt vì một lỗi trong **gate compile của tôi** (lỗi **#39**), không
phải vì Triton hỏng trên sm_75: gate so hai **mask dropout** khác nhau rồi gọi đó là sai số
compiler. Đã sửa và đo trên chính T4: `certified=True` ở mọi cấu hình, **2,2–2,35×** ở
batch 512 và **2,63×** ở batch 256 ⇒ round **17,9 ph → ≈10,5 ph**, 50 round
**14,9 h → ≈8,8 h/kịch bản**. Tiến trình huấn luyện của ba run cũ vẫn **lành mạnh** trước khi
dừng (20c `f1_macro` 0,192 → 0,578 → 0,705; 50c 0,223 → 0,547 → 0,682).

⚠ §14.3 ghi **hai chẩn đoán sai** tôi đã mắc trong phiên này — đọc trước khi đo lại bất cứ thứ
gì qua W&B.

Rà soát lần 1–3 (§6, §11) đã đóng: **R01–R18** đều sửa xong, mỗi mục một ca chèn lỗi. Suite
local **13 file, 88 phép kiểm**, pass toàn bộ. Tổng **41** lỗi ghi ở [`TEST_LOG.md`](TEST_LOG.md)
§2 — trong đó #36, #37, #38 chỉ lộ ra khi chạm phần cứng thật, sau khi mọi cổng local đã xanh.

**Mục tiêu.** Dựng lại phương pháp huấn luyện của *FD-IDS* (Zhang et al., **Sensors** 2025,
25, 4309 — `sensors-25-04309.md`) trên bộ dữ liệu VeReMi NextGen và bộ phân loại **DAGSNet**
trong `knowledge/`, cho **ba cấu hình 20 / 50 / 100 client**, chạy trên Kaggle 2×T4.

---

## 0. Mục lục — ở đâu có gì

| đường dẫn | nội dung | ai sửa |
|---|---|---|
| [`sensors-25-04309.md`](sensors-25-04309.md) | bài báo gốc FD-IDS | chỉ đọc |
| [`knowledge/`](knowledge/) | **sự thật không đổi**: dữ liệu, kiến trúc, máy local, hạ tầng | sửa khi *đo lại*, không khi đổi phương pháp |
| ├ [`DATASET.md`](knowledge/DATASET.md) | 43.045.415 train / 10.761.343 test, 3 phân mảnh α=0,5, ngân sách bước | |
| ├ [`ARCHITECTURE.md`](knowledge/ARCHITECTURE.md) | DAGSNet 395.024 tham số, 66 cột, 16 lớp, mã nguồn đã kiểm | |
| ├ [`LOCAL_ENV.md`](knowledge/LOCAL_ENV.md) | máy local + **ranh giới** local↔Kaggle | |
| ├ [`KAGGLE_DATASETS.md`](knowledge/KAGGLE_DATASETS.md) | 4 dataset, đều **public** | |
| ├ [`runtime.json`](knowledge/runtime.json) | digest image Kaggle đã kiểm, `machine_shape` | |
| └ `meta.json` / `scaler.json` | thứ tự 66 cột, tên 16 lớp, mean/std | |
| [`papers/fd-ids-2025/`](papers/fd-ids-2025/) | **phương pháp này** — tham số đã chốt nằm ở §1 dưới đây | |
| ├ [`proj/`](papers/fd-ids-2025/proj/) | **7 module** — nguồn duy nhất của code notebook | sửa ở đây |
| └ [`notebook/`](papers/fd-ids-2025/notebook/) | `{20,50,100}c/` = run thật · `{20,100}c_probe/` = calibration 2 round. `.ipynb` **sinh tự động** | ❌ không sửa tay |
| [`scripts/`](scripts/) | công cụ | |
| ├ `gen_notebook.py` | sinh notebook từ `proj/`; `--probe`, `--require-resume`, `--kernel-source`, `--dataset-source`, `--max-hours`, `--run-tag` | |
| ├ `gen_resume_probe.py` | probe **CPU** chạy đúng `resolve_resume` production trên checkpoint dataset, assert `last == expected` — bắt buộc trước mỗi continuation (§10.3) | |
| ├ `validate_notebooks.py` | cổng kiểm tĩnh 5 lớp: metadata + `id`↔slug, CFG qua AST, **module ↔ nguồn từng byte**, thứ tự luồng thực thi, tên biến qua các cell | |
| ├ `verify_run.py` | kiểm artifact đã tải về, offline | |
| ├ `run_local_checked.py` | **watchdog RAM bắt buộc** cho mọi test local | |
| └ `kaggle_mcp_headers.py` / `sync_kaggle_mcp.py` / `probe_kaggle_mcp.py` | MCP | |
| [`tests/`](tests/) | **13 file, 88 phép kiểm** — mỗi lỗi đã sửa có ca chèn lỗi riêng | |
| ├ `test_ckpt` · `test_fdids` | checkpoint/fingerprint/marker · Eq. (2)–(6) và `client_update` | |
| ├ `test_amp_guard` · `test_resume_import` | R11 overflow AMP · R12 marker nguồn + chuỗi hash | |
| ├ `test_verifier` · `test_data_cache` | R13 12 ca tamper · R14 loader thật + cache | |
| ├ `test_teacher` · `test_budget` · `test_schedule` | R16 teacher cache · R17 đồng hồ · bất biến theo lịch | |
| └ `test_smoke_real` · `test_validator` | end-to-end dữ liệu thật · R18 7 mutation | |
| [`TEST_LOG.md`](TEST_LOG.md) | **mọi kết quả đo** (local + Kaggle) | ghi thêm, không xoá |
| [`report.md`](report.md) + [`figures/`](figures/) | **báo cáo** — sinh bởi `scripts/make_report.py` từ `runs/`; đủ 10 metric × 50 round/cấu hình, tự đối chiếu CSV↔JSON↔CM | ❌ không sửa tay: sửa generator rồi chạy lại |
| `.claude/skills/` + `.agents/skills/` | skill `kaggle-training-notebook` — `.agents` là bản gốc, `.claude` là bản mirror; **sửa thì sửa cả hai** | |

Dữ liệu local: train `~/nckh/dataset/fl_client/alpha05/{20,50,100}_client/train/client_id=NNN/`,
test `~/nckh/dataset/centralized/test/`.

**Vòng đời chuẩn:** sửa `proj/*.py` → `python scripts/gen_notebook.py --owner <acct>` →
`python scripts/validate_notebooks.py` → nhúng W&B key → **validate lại** → push.
Validator nay so **từng byte** module nhúng với `proj/*.py`, nên notebook cũ (stale) bị chặn.

**7 module `proj/`:** `model` (DAGSNet) · `data` (parquet → fp16, liệt kê part tường minh) · `fdids` (Eq. 2–6) ·
`driver` (vòng FL, 2 worker) · `ckpt` (weights/resume/marker/fingerprint) · `metrics`
(10 metric) · `verify` (dựng lại mọi con số từ artifact).

---

## 1. Phương pháp — cái gì lấy từ đâu

FD-IDS = **FedProx + Knowledge Distillation round-wise**. Client update, Eq. (6):

```
L = λ·L_hard + (1−λ)·L_soft + β·L_proximal
    L_hard     = CrossEntropy(logit student, nhãn nguyên)
    L_soft     = T²·KL( softmax(Z_t/T) ‖ softmax(Z_s/T) )      Eq. (4)
    L_proximal = (μ/2)·‖w_k − w_G^t‖²                          Eq. (3)
w_G^{t+1} = Σ_k (n_k/n)·w_k^{t+1}                              Eq. (2)
```

| tham số | giá trị | nguồn |
|---|---|---|
| Optimizer / lr | Adam / **0,001** | bài báo Table 3 |
| μ (regularization) | **0,01** | bài báo Table 3 |
| λ (distillation weight) | **0,5** | bài báo Table 3 |
| β (proximal weight) | **0,1** | bài báo Table 3 |
| T (temperature) | **3** | bài báo Table 3 |
| Loss cứng | CrossEntropy | bài báo Table 3 |
| KD interval | **round-wise** (mỗi round) | bài báo §4.3.2 — tốt nhất trong 3 lựa chọn |
| Tham gia | **toàn bộ** client mỗi round (m = K) | Algorithm 1 dòng 4 |
| **Batch** | **512 / 512 / 256** (20c/50c/100c) | `knowledge/DATASET.md` §4 |
| **Round × epoch** | **50 × 1** | chủ dự án chốt 2026-09-10; khớp `DATASET.md` §4 |
| Mô hình | **DAGSNet** 395.024 tham số | `knowledge/ARCHITECTURE.md` |
| Seed | 42 | `knowledge/ARCHITECTURE.md` |
| Clip grad-norm | 1,0 | `knowledge/ARCHITECTURE.md` §4.1 |
| Precision | fp16 AMP, loss ở fp32 ngoài autocast | `knowledge/ARCHITECTURE.md` §4.1 |

### 1.1 Deviation so với bài báo — phải công bố kèm mọi con số

| hạng mục | bài báo | bản này | vì sao |
|---|---|---|---|
| Bộ phân loại | DNN 5 lớp 32-64-128-64-32, 22.095 tham số | **DAGSNet, 395.024** | chủ dự án chỉ định |
| Dữ liệu | Edge-IIoT / N-BaIoT | **VeReMi NextGen**, 16 lớp, 66 đặc trưng | chủ dự án chỉ định |
| Số client | 9 | **20 / 50 / 100** | chủ dự án chỉ định |
| Non-IID | Dirichlet θ = 1 và θ = 0,1 | **α = 0,5 cố định** | phân mảnh đã dựng sẵn, không sinh lại |
| Round × epoch | 40 × 2 | **50 × 1** | chốt 2026-09-10, khớp ngân sách `DATASET.md` §4 |
| Batch | 128 | **512 / 512 / 256** | knowledge ưu tiên; batch 128 tốn ~75 h, vượt quota |
| Tiền xử lý | one-hot + MI chọn top-k đặc trưng | **không** — dùng đủ 66 cột `f_*` đã z-score | dữ liệu giao ở trạng thái đã xử lý |
| Metric | Accuracy/Precision/Recall/F1; FPR/FNR theo ma trận nhị phân ở §4.2 | **10 metric đa lớp** | vẫn định nghĩa được FPR/FNR bằng one-vs-rest hoặc gộp benign/attack, nhưng chưa chốt quy ước; không tự thêm |
| Đánh giá client (cột B/W Table 5) | có | **không** | xem §1.2 |

### 1.2 Vì sao chỉ đo global model — đã kiểm lại 2026-09-10

Algorithm 1 dòng 14: `Initialize w_k = w_G^t`. **FD-IDS không có bước cá thể hoá.** Mỗi round
client bị ghi đè bằng global model, train, gửi lên, rồi bị ghi đè lại. Không có trọng số client
nào tồn tại qua ranh giới round (khác FedPer / Ditto / per-FedAvg). Mô hình duy nhất tồn tại
liên tục là global model → **10 metric của nó, mỗi round, trên đủ 10.761.343 dòng test**.

Cột B/W của Table 5 là best/worst client accuracy. Diễn giải đó là điểm của local model sau
update phù hợp với Algorithm 1, nhưng bài báo không mô tả đầy đủ protocol đánh giá từng client.
Chỉ đo global model là phạm vi đã chọn của bản dựng, không tái lập toàn bộ Table 5.

⚠ Code **cố ý** reset Adam moment và GradScaler cho từng client, từng round; không có
optimizer toàn cục. Reset trọng số trong Algorithm 1 **không tự suy ra** phải reset Adam
moment: đây là quy ước triển khai hiện tại, cần công bố. Vì state này đã bỏ đi, không cần
lưu nó ở ranh giới round.

**Hợp đồng RNG (đã sửa đường eager CPU, R04):** nguồn ngẫu nhiên của một client được dẫn xuất từ
`(seed, round, client)` — `torch.manual_seed` cho Dropout và một `Generator` riêng cho shuffle,
đặt ngay trước mỗi client. Mục tiêu là kết quả không phụ thuộc worker nhận client hay số client
đã chạy trước đó. Đo được **trên CPU, compile=False**: 1 worker vs 2 worker cho `Δ trọng số = 0`,
và replay một round sau khi xoá marker cho kết quả **bit-identical**. Chưa suy rộng bằng chứng
này sang CUDA/compiled Dropout, hoặc hai rank chọn backend khác nhau; xem R16. Vì RNG tái dựng được từ
seed, `resume/round_NNN.pt` **không cần** để tiếp tục — nó chỉ lưu round + RNG của driver để
truy vết. Đây là hợp đồng có chủ đích, phải công bố khi báo cáo.

---

## 2. Hợp đồng artifact

Hợp đồng dự kiến: marker **tuyệt đối cuối cùng**. Code hiện ghi `seconds` trước metrics JSON,
CSV, marker và W&B, nên chưa bao trọn phần I/O; xem R17.

```
runs/fdids_<K>c/
  weights/round_NNN.pt      ← CHỈ trọng số: {round, model(tensor), cfg, fingerprint, metrics}
  resume/round_NNN.pt       ← round + RNG driver (truy vết; KHÔNG cần để tiếp tục — §1.2)
  confusion/round_NNN.npy   ← ma trận 16×16, tổng == 10.761.343
  preds/round_NNN.u8.npy    ← nhãn dự đoán, đúng thứ tự test (10,76 MB/round)
  metrics/round_NNN.json    ← row đầy đủ (10 metric + ce/kd/gnorm/steps/applied/skipped/
                               vram_gb/seconds) + 16 dòng per-class
  logs/round_NNN.json       ← từng client: cid, n_k, rank, steps, applied, skipped,
                               nonfinite, seed, ce, kd, gnorm, sec, vram_gb
  reports/manifest.json     ← cfg hiệu lực, fingerprint, feature order, class names,
                               số dòng từng client, phiên bản torch/CUDA
  history.csv               ← DẪN XUẤT từ metrics/*.json; ghi qua tmp + os.replace
  complete/round_NNN.done   ← TUYỆT ĐỐI cuối cùng
```

* **Marker không phải bằng chứng.** `ckpt.round_ok()` yêu cầu 4 file weights/resume/metrics/
  confusion tồn tại và không rỗng, nhưng chỉ đọc weights/metrics/confusion; chưa đọc/kiểm
  nội dung resume. Fingerprint được so với trường lưu trong weights khi có `fp` truyền vào.
  Chính `last_complete_round()` kiểm marker; `round_ok()` không kiểm marker. Hàm đầu trả
  round lớn nhất sao cho **1..r đều đủ** — một khoảng trống ở giữa kết thúc dải, không phải
  lấy marker lớn nhất.
* **Import qua staging.** `resolve_resume()` copy vào cây tạm, kiểm ở đó, rồi **publish từng
  round một** với marker sau cùng. Test cũ chứng minh retry một số vị trí crash ở đích;
  **chưa bảo vệ marker của nguồn hoặc lịch sử khác nhau cùng fingerprint** — R12.
  `preds/` và `logs/` **có** được import (`RESUME_SUBDIRS`, sửa sau bản ghi cũ ở đây): một
  run trải qua nhiều phiên vẫn phải tự kiểm được từ output **cuối cùng** của nó; giá là
  ~500 MB copy mỗi lần tiếp nối. Probe CPU 2026-09-11 đo cả 6 thư mục đều import đủ.
* **Fingerprint bao 21 khoá, gồm cả `run_name` và `data_id`** (`data_id` hash feature order,
  class order, K và scaler; chưa có identity nội dung parquet/partition — R14).
  `rounds` **cố ý không** nằm trong đó: lr là hằng, không có scheduler trải theo round,
  nên trọng số ở round r không phụ thuộc tổng số round — đó chính là điều làm continuation hợp lệ.
  Thiếu một khoá là `KeyError`, không mặc định `None`.
* Dựng lại mô hình tại **bất kỳ** round: `proj/ckpt.py::load_weights(path, build_model,
  expect_params=395_024)` → `strict=True` + assert số tham số + fingerprint + **kiểm hữu hạn**.
  Đã đo: **`max|Δ logits| = 0.0`**.
* `proj/verify.py::verify_run()` dựng lại 10 metric **từ ma trận nhầm lẫn** và đối chiếu một
  phần artifact. Chỉ dựng lại CM từ `preds/` khi file có mặt và được truyền `y_true_path`;
  thiếu file vẫn có thể pass. R13–R15 ghi các lỗ hổng đã tái hiện và phần cần lưu thêm.
* Dung lượng/scenario: trọng số 1,65 MB × 50 = **83 MB**; preds 10,76 MB × 50 = **538 MB**;
  confusion + metrics + logs < 5 MB. Tổng ≈ **630 MB**, dưới trần 20 GB output của Kaggle.

## 3. Trạng thái — 2026-09-10

**Xong:**
- MCP dựng lại; skill sửa hợp đồng checkpoint **và** chính sách tài khoản (§4) ở cả
  `.agents` và `.claude`; `knowledge/LOCAL_ENV.md` dọn sạch số liệu theo phiên.
- **R01–R10** (rà soát 1–2) và **R11–R18** (rà soát 3) đều đã sửa, mỗi mục có ca chèn lỗi
  riêng. Quá trình sửa sinh thêm 2 lỗi mới, cũng đã sửa. Tổng **35** lỗi ở `TEST_LOG.md` §2.
- Suite local: **13 file test, 86 phép kiểm, pass toàn bộ**. Validator kiểm 5 lớp (metadata,
  CFG qua AST, module ↔ nguồn **từng byte**, thứ tự luồng thực thi, tên biến qua các cell) và
  tự có **7 mutation** phải bị chặn.
- **3 notebook đã push và đang chạy** — §10.

**Chưa có (cập nhật 2026-09-11 phiên 2):**
- ✅ 20c và 50c đủ 1..50, `verify_run --require-rounds 50` pass mode full, đã merge phiên
  (`runs/fdids_20c_v2`, `runs/fdids_50c_v2`). 100c đang tiếp nối (§15.4).
- ✅ `report.md` (gốc repo) **INTERIM**: 20c + 50c đủ 50 round × 10 metric, per-class, hình; 100c là
  mục N/A có nhãn. Chạy lại `python scripts/make_report.py` sau khi merge 100c ⇒ tự thành bản đầy đủ.

**Ngân sách — vẫn là ƯỚC TÍNH**, suy từ ms/step T4 của một dự án khác:

| cấu hình | batch | steps/round | ước tính | `max_seconds` đã đặt | tài khoản |
|---|---:|---:|---:|---:|---|
| 20c | 512 | 84.083 | ≈5,0 h | **5,8 h** | `minhtrit06` |
| 50c | 512 | 84.098 | ≈5,0 h | **6,5 h** | `khanhmay0304` |
| 100c | 256 | 168.200 | ≈8,2 h | **9,5 h** | `khanhmay0304` |

`max_seconds` đặt theo **quota của tài khoản chạy nó**, không phải theo trần 12 h: run dừng
sạch ở ranh giới round và commit đủ, thay vì bị Kaggle cắt giữa chừng. Đồng hồ là
`time.monotonic()` bắt từ cell đầu (gồm spawn + prepack + compile), `seconds` mỗi round bao
trọn commit và W&B, và driver **từ chối bắt đầu** một round mà phiên không đủ chỗ commit.

## 4. Quyết định vận hành

* **Tài khoản (chính sách 2026-09-10, thay bản 09-08 và 09-09):** agent **tự đổi tài khoản,
  không hỏi**. `kaggle_account.py use/ensure --confirm` chạy trực tiếp. Việc duy nhất của chủ
  dự án là **reconnect MCP client khi thực sự cần** (`/mcp`) — CLI đổi ngay, còn MCP client
  đang chạy giữ bearer cũ và sẽ âm thầm trả lời như tài khoản trước. Agent **phải nói rõ**
  đã đổi sang tài khoản nào và vì sao; đổi im lặng vẫn là lỗi. Hai ngoại lệ giữ nguyên:
  `kaggle auth login --force` (huỷ refresh token không khôi phục được) và bất kỳ tài khoản
  nào chủ dự án đã giới hạn cho một việc cụ thể. Đã ghi vào skill ở cả hai cây.
  Roster và quota đổi hằng ngày — đọc bằng `kaggle_account.py quota` trước mỗi lần launch.
  **Tài khoản thứ 5 `minhtriethihi` (2026-09-11, 30 h):** chỉ có `KGAT_` token, chưa có
  OAuth snapshot (chờ chủ dự án `add-account` — §15.1). Token `KGAT_` **là API token của CLI
  2.2.4**: `KAGGLE_API_TOKEN=<đường dẫn file token>` (SDK đọc từ file, giá trị không vào
  dòng lệnh) đủ để push/kéo/tạo dataset. Helper `use/ensure/health/quota` vẫn cần OAuth.
* **W&B:** đã nhúng key vào 3 notebook `is_private: true` theo uỷ quyền của chủ dự án. Key nằm
  **vĩnh viễn** trong version history của Kaggle. `.ipynb` **không được** commit/chia sẻ
  (`.gitignore` đã chặn). Muốn thu hồi: revoke ở W&B **trước**, theo `wandb.md` §1.1.
* **Test local:** luôn bọc `scripts/run_local_checked.py`, một cây test tại một thời điểm,
  CPU, `compile=False`, fixture nhỏ. Watchdog dừng thì **giảm bài test, không nới ngưỡng**.
  ⚠ `test_smoke_real` nay đạt **2.992 MiB / trần 3.000 MiB** vì chạy thêm cấu hình 2 worker.
  Thêm bài vào file đó sẽ chạm watchdog — **tách file mới**.
* **Sửa code là sửa `proj/*.py`.** Validator so từng byte module nhúng với nguồn, nên một
  notebook chưa regenerate sẽ bị chặn thay vì âm thầm chạy code cũ.

---

## 5. Caveat bắt buộc kèm mọi con số công bố

Kế thừa `knowledge/DATASET.md` §6 và `ARCHITECTURE.md` §8:

1. Split theo **thời gian mô phỏng**, không theo xe; test chỉ có scenario `_7`.
2. Mất cân bằng **41:1** → đọc `f1_macro`, **không** đọc `accuracy`.
3. **Rò rỉ Sybil:** 100% dòng `trafficCongestionSybil` nằm trong flow toàn cùng nhãn.
4. `scaler.json` fit trên **toàn bộ** 43 M dòng train, tức trên dữ liệu của mọi client gộp
   lại — trong FL thật client không có thống kê toàn cục. **Rò rỉ phải công bố**, có trước bản FL này.
5. Đặc trưng lưu **fp16** — giá trị đã bị lượng tử hoá, là lựa chọn có chủ ý.
6. Test **không** chia theo client; điểm test đo tổng quát hoá toàn cục.
7. **Không** đặt số của bản này cạnh số của bài báo trong cùng một bảng: bài báo phân loại trên
   Edge-IIoT/N-BaIoT với bộ metric khác. Cái kế thừa là **phương pháp**, không phải con số.
8. Một run, một seed. Chưa có replication.

---

## 6. Rà soát notebook và bằng chứng test — 2026-09-10 (LỊCH SỬ)

> Giữ nguyên làm hồ sơ của phiên rà soát. Các bản sửa tiếp theo được ghi ở §7; kết luận mới
> nhất ở §11–§12. Phần §6.2 (phạm vi thực của test cũ) vẫn đáng đọc: nó giải thích *vì sao* một suite
> pass toàn bộ mà vẫn bỏ lọt 10 lỗi.

**Phạm vi:** đọc bài báo local `sensors-25-04309.md`, đặc tả `knowledge/`, 6 module, generator,
validator, 3 notebook và 3 test. Các phép thử mới chạy CPU trong `conda nckh`, một cây test
mỗi lần, qua watchdog; không compile thật, không chạy GPU, không quét lại toàn bộ dữ liệu.
Số dòng/phân bố toàn bộ vẫn lấy từ audit đã lưu trong `knowledge/`, không coi là số đo mới.

### 6.1 Những phần đã đối chiếu đúng

| Hạng mục | Kết luận và giới hạn |
|---|---|
| Notebook ↔ nguồn | Cả **18/18 module nhúng** (6 × 3 notebook) khớp `proj/*.py` sau bỏ newline cuối. Không phát hiện notebook stale ở lần rà soát này. |
| Kiến trúc | **8 định nghĩa hàm/class chung** khớp AST với mã đầy đủ trong `ARCHITECTURE.md`; layout `(B,11,6) → transpose`, 395.024 tham số. Driver khởi tạo mới, không nạp checkpoint pretrained round 5 trong `knowledge/`. |
| Cấu hình | 20c/50c: batch 512; 100c: 256; cả ba 50 round × 1 epoch, toàn bộ client. Metadata trỏ đúng FL dataset tương ứng + centralized test; notebook chưa chứa execution output. |
| Eq. (4), (6) | KL đúng chiều **teacher ‖ student**, có `T²`; CE/KD = 0,5/0,5; gradient proximal = `βμ(w−w_G)` = **0,001·(w−w_G)** trên learnable parameters. Không thiếu hay nhân thừa μ/β. |
| FL round | Student reset về global; teacher giữ global cố định ở `eval()`, không gradient; aggregation sau đủ client, trọng số `n_k/N`, thứ tự cộng theo client id. Hai worker không dùng DDP để trộn gradient giữa client. |
| Dữ liệu/eval | Đọc nhãn `label`, chỉ dùng 66 cột `f_*`; train không scale lại, test scale bằng `std_used`. Eval trên các shard không padding, cộng confusion nguyên rồi tính 10 metric. Cần kiểm thêm giá trị/identity và artifact như R05/R07/R08. |

### 6.2 Phạm vi thực của các test đã lưu

- **`test_ckpt.py`:** kiểm `C.save_round` và `C.load_for_resume`, trong khi FD-IDS chạy
  `C.save_round_weights` và nhánh resume tự viết trong `D.run`. Con số **resume 3,25 MB** ở
  `TEST_LOG.md` bao gồm **Adam state**, không phải dung lượng RNG-only của FD-IDS. Test
  `max|Δlogit|=0` là bằng chứng tốt cho dựng lại trọng số ở fixture, không chứng minh replay
  toàn bộ FL hay artifact của mọi round.
- **`test_fdids.py`:** kiểm số học proximal/KD bằng công thức viết lại trong test; gọi thật
  `aggregate`, `layout`, flatten/unflatten. **Không gọi `client_update`** để so toàn bộ bước
  optimizer với một reference; chưa kiểm AMP, cached teacher, BN counter hoặc tail batch.
- **`test_smoke_real.py`:** gọi driver thật trên **3 client cắt ngắn của 100c**, 20.000 dòng/client,
  40.000 dòng test, một worker, CPU, 2 round. Nó tự prepack bằng code trong test, **không gọi
  `load_clients`/`load_test` production**, và không thực thi các cell notebook. Recovery chỉ xoá
  marker sau khi round đã ghi xong; chưa gây crash giữa từng thao tác ghi/import.
- **`validate_notebooks.py`:** kiểm cú pháp, metadata và một số chuỗi. Không đối chiếu module
  với nguồn, chưa parse effective CFG hay kiểm dataset slug đầy đủ. Kiểm `require_resume`
  bằng vị trí chuỗi trong toàn notebook có thể bị thoả bởi khai báo CFG/định nghĩa hàm, dù
  cell không thực hiện kiểm tra đúng thứ tự. Cell hiện có kiểm *checkpoint tồn tại* trước
  decode, nhưng fingerprint chỉ được driver kiểm **sau prepack và worker startup**.

`TEST_LOG.md` giữ nguyên làm lịch sử. Phần rà soát này đính chính cách diễn giải các số đo cũ;
không phủ nhận những assertion mà test thực sự đã kiểm.

### 6.3 Các phép tái hiện lỗi trong phiên rà soát

Các probe tạm chạy trên fixture qua watchdog, không thay đổi code production:

| Probe | Kết quả quan sát | Chứng minh điều gì |
|---|---|---|
| Sửa độc lập 13 mục CFG | Fingerprint **không đổi** với `lr`, `mu`, `beta`, `lam`, `temperature`, `seed`, `local_epochs`, `batch`, `rounds`, `dropout`, `n_clients`, thêm `feature_cols`/`scaler_hash` khác | R05: fingerprint không bảo vệ cấu hình khoa học/preprocessing |
| Stub `client_update` trả `steps=1, skipped=1` vào `_worker` thật | Worker báo **`applied=1`** | R01: lỗi kế toán bước; đây là mô phỏng all-skipped, không phải đo overflow trên T4 |
| Ngắt `resolve_resume` tại copy `metrics/` | Marker round 1 đã xuất hiện; gọi lại vẫn trả 1 trong khi metric chưa có | R02: import dở được tin là đã hoàn tất |
| Import chạy hết vào working trống | **Không có `history.csv`** ở đích | R02: history ở root không nằm trong danh sách copy |
| Thay `torch.compile` bằng hàm trả chính eager model, chỉ chạy logic gate | Gate báo mismatch **0,214743**, rơi về eager; **62 BN running buffers** thay đổi | R03: ngay cả hai đường là cùng một model, gate vẫn loại nhầm vì warmup thay trạng thái; không kết luận khả năng compile thật trên sm_75 |

MCP native `get_accelerator_quota` trả thành công trong phiên này: chỉ chứng minh kết nối
được xác thực ở thời điểm kiểm. Chưa đối chiếu lại CLI/account identity; không dùng phép đọc
quota làm bằng chứng notebook chạy được trên 2×T4. Không launch hay đổi tài khoản trong audit.

### 6.4 Kết quả chạy lại suite trong audit

| Lệnh bên trong watchdog (`conda nckh`, `CUDA_VISIBLE_DEVICES=''`) | Kết quả mới |
|---|---|
| `python tests/test_ckpt.py` | **10/10 pass**, rebuild Δlogit=0; weights 1,65 MB, resume generic 3,25 MB; peak RSS 780 MiB |
| `python tests/test_fdids.py` | **5/5 pass**, proximal rel=1,721×10⁻⁸; aggregation Δmax=1,192×10⁻⁷; peak RSS 602 MiB |
| `python tests/test_smoke_real.py` | **pass**; r1 F1=0,025949, r2=0,027133, replay r2=0,025921; history vẫn [1,2]; peak RSS 1.987 MiB |
| `python scripts/validate_notebooks.py` | **pass** cả 20c/50c/100c; peak RSS 5 MiB |

Các giá trị trên là kiểm lại **code chưa sửa**. Kết quả pass đồng thời với lỗi ở §6.3 chứng
minh cần mở rộng phạm vi test; không chứng minh 20/50/100 client đã được train hoàn tất.

---

## 7. R01–R10 — các bản sửa vòng 2 và bằng chứng đã có

Mô tả đầy đủ từng lỗi (vị trí, cách tái hiện) nằm ở `TEST_LOG.md` §2, dòng #9–#18. Bảng này
là hồ sơ sửa ở đâu và bài nào đã chạy. **Không đọc “pass” ở đây như nghiệm thu toàn bộ
hợp đồng**: §11–§12 bổ sung các ca chưa được test cũ bảo vệ.

| ID | sửa ở đâu | bài kiểm chứng minh | số đo |
|---|---|---|---|
| **R01** client applied=0 vẫn được nhận | `driver.check_updates()` — hàm thuần, `applied = nsteps - skips` không còn `max(1,·)` | `test_fdids` #7 | chặn đủ 4 ca: applied=0, grad non-finite, trọng số NaN, thiếu client |
| **R02** import công bố marker sớm, mất history | `ckpt.resolve_resume/_import_from` — staging → kiểm → publish từng round → marker | `test_ckpt` #13 | ngắt ở round 3 → chỉ **2** được tin; retry hoàn tất **3** + history |
| **R03** gate compile so hai trạng thái BN | `driver._compile` — snapshot/restore weights+BN+RNG, so **logits và gradient** từ cùng trạng thái, mẫu **dữ liệu thật** | chờ T4 (§9) | ngưỡng chốt trước: `\|Δlogit\| ≤ 5e-2`, `rel\|Δgrad\| ≤ 5e-2`, 0 lật top-1 |
| **R04** Dropout/RNG phụ thuộc lịch | `driver._worker` — `manual_seed(seed, round, client)` ngay trước mỗi client | `test_smoke_real` | 1 vs 2 worker `Δ = 0`; replay **bit-identical** |
| **R05** fingerprint mù | `ckpt.FINGERPRINT_KEYS` 21 khoá, gồm `data_id`; bỏ bí danh `batch_per_gpu`/`total_rounds` | `test_ckpt` #11, validator | đổi trên **14** khoá khoa học, đứng yên trên **6** khoá vận hành; thiếu khoá → `KeyError` |
| **R06** history không atomic | `ckpt._write_csv` (tmp + `os.replace`), `rebuild_history()` từ metrics json | `test_ckpt` #12 | cắt CSV còn 1 dòng → dựng lại đủ **3** round, không mất cột |
| **R07** cache prepack dùng khi thiếu/sai nguồn | cell prepack — `manifest.json` ghi **sau cùng**, khoá theo `data_id`/`n_clients`/mount; `data._labels()` kiểm nhãn 0..15 trước `uint8` | validator + `test_smoke_real` | validator ép `MF.write_text` phải đứng sau file cache cuối |
| **R08** artifact không kiểm lại được | `preds/` ghi mỗi round; đếm logits non-finite trên device; `proj/verify.py` + `scripts/verify_run.py`; `write_manifest()` | `test_smoke_real` | `verify_run` pass: metrics == CM == weights == history, **preds dựng lại đúng từng ô CM** |
| **R09** ngân sách/quan sát thiếu | `run(t_origin=T0)`; `seconds` bao commit; `logs/` từng client; `vram_gb`; `try/finally` dừng worker | chờ T4 (§9) | driver in `session X.XXh` mỗi round; cell cuối in `projected: 50 rounds = X h` |
| **R10** summary không xác minh hoàn tất | cell tổng kết dùng `last_complete_round` + `verify_run`; round cuối là kết quả, best-test gắn nhãn *descriptive only*; `assert ok` cuối cell | `test_smoke_real` (`require_rounds=2`) | notebook **fail** nếu artifact không tự nhất quán |

Hai lỗi **mới sinh trong lúc sửa**, cũng đã sửa và có test:

| # | lỗi | bắt bởi |
|---|---|---|
| 19 | `round_ok` không kiểm marker → retry sau crash *giữa copy và mark* bỏ luôn bước mark, round kẹt vĩnh viễn | `test_ckpt` #13 (chính ca chèn lỗi vừa viết) |
| 20 | Cổng kiểm thứ tự soi **toàn** notebook, nên `C.resolve_resume` trong thân `driver.py` nhúng thoả mãn một khẳng định về **luồng thực thi** | chính cổng đó fail khi thêm check `data_id` |

Phần compile sm_75 và thời gian T4 của **R03/R09** vẫn cần Kaggle. Tuy nhiên các vấn đề logic
liên quan còn kiểm được ở local (R16–R17), cùng các ca AMP/import/verifier/cache ở §11.
Ngưỡng đã chốt không được nới để hợp thức hoá kết quả đo.

## 8. Những lựa chọn khoa học cần ghi rõ trước khi sửa thuật toán

1. **Nguồn tham số có phân cấp.** Giữ quyết định đã ghi ở §1: FD-IDS Adam/lr cố định từ
   Table 3; 50×1 và batch theo cấu hình đã chốt; DAGSNet từ knowledge. `ARCHITECTURE.md` §4.1
   có AdamW + weight decay + cosine của **run DAGSNet cũ**, không tự động biến thành mặc định
   FD-IDS. Adam hiện dùng betas `(0.9,0.999)`, eps `1e-8`, weight_decay `0`, amsgrad `False`;
   các giá trị mặc định không được bài FD-IDS liệt kê hết, cần ghi là lựa chọn triển khai.
   Không có class weighting/resampling, label smoothing hay scheduler trong driver hiện tại.
2. **Thời điểm KD:** §3.5.2 và Algorithm 1 tính CE+KD+proximal trong từng batch local với
   teacher global của round; §4.3.2 lại diễn đạt round-wise là “after each communication round”.
   Code hiện theo **Algorithm 1**: không có một pha KD bổ sung sau aggregation. Đây là cách
   diễn giải có căn cứ nhưng cần ghi rõ; không tự thêm epoch KD hoặc teacher cập nhật giữa
   client. Round đầu cũng dùng teacher mới khởi tạo; thêm warmup không KD là đổi phương pháp.
3. **BatchNorm là lựa chọn mới do đổi classifier.** Code average `running_mean/running_var`
   theo `n_k/N`, `num_batches_tracked=max`; proximal không đụng buffer. Trung bình các variance
   không phải pooled variance của dữ liệu toàn cục. Bài FD-IDS không quy định cách gộp BN cho
   DAGSNet. Giữ convention hiện tại làm baseline, kiểm bằng fixture khác mean/variance; nếu
   thử BN recalibration/FedBN, ghi là biến thể riêng và chỉ dùng train cho calibration.
4. **Teacher cache không “exact” về số học ở mọi backend.** `teacher_logits` lưu fp16, batch
   inference lớn; `eval()` làm BN/Dropout ổn định nhưng không loại bỏ rounding/batch-kernel
   differences. Test cached/online trên cùng weights/rows với nhiều batch size, đo sai số
   logits, KL và gradient. Student train/teacher eval có BN/Dropout khác nhau nên KD ban đầu
   không nhất thiết bằng 0 dù cùng trọng số; đó không tự nó là lỗi anchor.
5. **100c đổi cả K và batch.** Chênh lệch kết quả 20c/50c/100c không thể quy hoàn toàn cho số
   client: 100c còn gấp gần đôi optimizer steps. Bảng báo cáo phải ghi batch/steps; nếu muốn
   tách hiệu ứng K, đề xuất thêm ablation cùng batch, không thay ba cấu hình đã yêu cầu.
6. **Độ chính xác và tác dụng của phương pháp chưa được chứng minh.** Smoke F1 ≈0,026 chỉ
   kiểm pipeline. Sau khi đóng các lỗi kỹ thuật, có thể so FedAvg/FedProx/KD-only/FD-IDS cùng
   split/seed/budget để kiểm tác dụng từng thành phần; cần quy định hệ số CE khi tắt KD để
   không vô tình so hai thang loss. Replication nhiều seed là công việc bổ sung nếu muốn
   kết luận ổn định thống kê. Không dùng test để điều chỉnh λ/μ/T hay chọn BN policy.

Các điểm trên được ghi để làm rõ provenance/giới hạn, **chưa phải uỷ quyền thay thuật toán**.
Nếu phiên triển khai cần thay optimizer, thời điểm KD, BN policy hoặc ngân sách đã chốt,
hỏi chủ dự án trước khi sửa phần phụ thuộc lựa chọn đó.

---

## 9. Điều kiện nghiệm thu còn lại

Bước 1–4 của kế hoạch cũ đã có sửa và test, nhưng **chưa đạt đầy đủ** sau các phép chèn lỗi
ở §11. Cần làm lại cổng local ở §12 trước khi dùng hai bước Kaggle dưới đây để nghiệm thu.

| Bước | Việc cụ thể | Điều kiện đạt |
|---|---|---|
| 5 — Kaggle có đo thật | Chạy 2 notebook probe ở §10 trên đúng image đã pin, `world_size=2`, đúng T4 | `device_count()==2`, capability `(7,5)`; gate compile in ra `\|Δlogit\|` và `rel\|Δgrad\|` **đạt ngưỡng đã chốt trước** hoặc rơi eager kèm lý do; prepack đủ 43.045.415 dòng; peak VRAM/GPU đo được; ít nhất **2 round** để tách startup khỏi steady state |
| 6 — chạy dài và continuation | 3 run thật ở §10; nếu bị cắt thì attach output, regenerate với `--require-resume --kernel-source`, push tiếp | `verify_run --require-rounds 50` pass cho **từng** scenario; history không lặp/mất round; `assert ok` ở cell cuối không nổ |

Ngưỡng dự kiến đã chốt **trước** khi đo, không được nới sau: gate compile `\|Δlogit\| ≤ 5e-2` và
`rel\|Δgrad\| ≤ 5e-2` với 0 dòng lật top-1; proximal CPU `rel < 1e-6`; aggregation `atol 1e-6`;
`client_update` vs Eq. (6) autograd `rel < 2e-5`. Không dùng `max Δlogit = 0,05` của cái gate
hỏng cũ làm chuẩn. Lưu ý R16: code hiện chỉ kiểm top-1 trên tập con `decisive`, chưa thực hiện
đúng yêu cầu “0 dòng lật top-1” trên toàn batch trong bảng trên.

---

## 10. Vận hành — ba run đang chạy

**Ràng buộc cứng:** tối đa **2 batch GPU session đồng thời mỗi tài khoản**; phiên tối đa
**12 h**; `/kaggle/working` bị xoá khi phiên mới bắt đầu. Ba run song song ⇒ ít nhất 2 tài khoản.

### 10.1 Đã launch — 2026-09-10/11

| kernel | tài khoản | v1 | v2 | **v3 = phiên 1, XONG** | phiên 2 (§15.4) |
|---|---|---|---|---|---|
| `khanhmay0304/fd-ids-veremi-100-clients` | khanhmay0304 | ERROR | huỷ ở round 1 | 6,2 h → **25**/50 | `minhtriethihi/…-100-clients` |
| `khanhmay0304/fd-ids-veremi-50-clients` | khanhmay0304 | ERROR | huỷ ở round 3+ | 9,7 h → **50/50 ✅** | — |
| `minhtrit06/fd-ids-veremi-20-clients` | minhtrit06 | ERROR | huỷ ở round 3+ | 6,4 h → **42**/50 | `minhtriethihi/…-20-clients` |

v1 chết ở lỗi #36 (sidecar `.stats.json`), tốn 0,08 h. v2 chạy đúng nhưng **toàn bộ ở eager**
vì lỗi #39, 17,9 ph/round; người dùng huỷ để relaunch. **v3** bật được compile — chi tiết,
`run_name` và ngân sách ở **§14.6**.

⚠ **Kaggle lấy slug từ `title`, không từ `id`.** Push đầu khai `id` `fdids-veremi-100c` với
title *FD-IDS VeReMi 100 clients` → kernel ra đời ở `fd-ids-veremi-100-clients`, `id` đã khai
thành vô dụng cho `kernels status`, kéo output và `kernel_sources`. Generator nay **suy `id`
từ title**; validator chặn mọi cặp không khớp.

⚠ **Quota bị reserve trước, không tính theo tiêu thụ** — nhưng **huỷ session trả lại phần chưa
dùng**, và trả nhiều: huỷ hai session của `khanhmay0304` đưa nó từ 6,17 h lên **16,09 h**. Nên
một số quota đọc lúc đang chạy **không** phải là số quota thật khả dụng; đọc lại **sau** khi huỷ
trước khi đặt `max_hours`.

### 10.1b Số đo T4 thật — 2026-09-11

`f1_macro` và thời lượng round đọc từ W&B (`scan_history`, có `_timestamp`):

| run | r1 f1 / acc | r2 f1 / acc | ce r1→r2 | skip/steps | **round** | VRAM train/eval | teacher |
|---|---|---|---|---|---:|---|---:|
| 20c | 0,192371 / 0,434955 | **0,577886 / 0,773878** | 0,6490 → 0,5865 | 5, 6 / 84.083 | **17,9 phút** | 7,25 / 7,09 GiB | 221 s |
| 50c | 0,223400 / 0,479411 | **0,547154 / 0,752569** | 0,6503 → 0,5615 | 0, 0 / 84.098 | **18,0 phút** | 7,16 / 7,09 GiB | 230 s |
| 100c | chưa xong round 1 (≈2× số bước) | | | | | | |

**Tiến trình huấn luyện lành mạnh.** `f1_macro` tăng gấp ~2,5–3× sau một round; `accuracy`
lên 0,77; `ce_client_mean` và `kd_client_mean` đều giảm. So với smoke local (0,026) thì đây là
học thật trên dữ liệu thật.

**`skip` = 5 và 6 trên 84.083 bước là xác nhận trên T4 rằng bản sửa R11 là bắt buộc.** Theo
code trước bản sửa, 5 bước overflow đó bị đếm là `nonfinite` và `check_updates` sẽ **giết
round 1 của 20c** sau khi đã trả tiền cho prepack và một round train đầy đủ. Trần warm-up 16
được chốt trước khi đo, và số thật rơi vào 0–6 — biên hợp lý, không phải may.

⚠ **Bẫy đo:** `wandb.Api().run(...).scan_history()` **trả kết quả cũ**. Có lúc nó báo "1 round,
32,6 phút trước" trong khi round 2 đã log xong từ lâu — dẫn tôi tới một chẩn đoán sai
(">32 phút/round"). Luôn đối chiếu `gap` giữa hai `_timestamp` liên tiếp, đừng dùng
"thời gian kể từ round cuối" làm số đo; và `r.load(force=True)` **không** đủ để ép làm mới.

### 10.1c ĐÃ ĐÓNG 2026-09-11 — compile rơi eager vì lỗi #39, không phải vì sm_75

**LỊCH SỬ. Kết luận "compile đã rơi về eager" là ĐÚNG; đường đi tới nó thì SAI.** Nguyên nhân
thật, số đo trên chính T4 và bản sửa: **§14.1–§14.2** và `TEST_LOG.md` §3.3. Giữ lại mục này
vì hai giả thuyết bị loại bên dưới vẫn là số đo dùng được.

Lập luận hỏng ở chỗ nào: nó so **13,4 phút** (dự đoán từ **thời gian/step thuần**) với
**18,0 phút** (wall-clock **cả round**, trong đó có teacher 221 s và eval) rồi gọi phần chênh
là "1,34× chậm hơn". Trừ đúng hai phần đó: `(1074 − 221 − ~40)/42.041 ≈ 19,3 ms/step` —
**khớp gần như hoàn hảo với eager-local 19,07**, tức T4 không hề chậm hơn local, nó chỉ đang
chạy eager. Bài học: **đừng so một dự đoán per-step với một wall-clock có chứa việc khác.**

Hai giả thuyết khác đã **đo và loại** (vẫn đúng):

| giả thuyết | đo được (sm_86, batch 512, đủ AMP+KD+clip+Adam) | kết luận |
|---|---|---|
| proximal `cat` + vòng `add_` (§11.8 nghi) | 14,84 → 16,88 ms/step = **+2,04 ms (13,7%)** | **không phải nút cổ chai**. `torch._foreach_*` hạ còn 15,16 ms, **bit-identical** (`max\|Δ\|=0`) — vẫn đáng đổi (~10%), nhưng không phải lời giải |
| gather `X[idx]` trên tensor thường trú | 17,60 → 19,07 ms/step = **+1,47 ms** | **không phải** |

Lúc đó **không xác minh được backend từ xa**, do hai lỗ hổng quan sát chính bản sửa R17 tạo ra
(#37, #38). Cả hai nay **đã có trong notebook đã push** (§14.4), nên từ version 3 trở đi
`summary["backend"]` đọc được ngay khi run còn sống.

**Ngân sách sau khi bật compile** (thay cho ước tính ≈60 h ở đây): 20c/50c ≈ **8,8 h**,
100c ≈ **14,2 h** cho 50 round — xem §14.6.

### 10.2 Round đầu là phép calibration — đọc gì, dừng khi nào

Không có notebook probe riêng: round đầu của chính ba run này cho đúng những số đó, **và**
giữ lại checkpoint thay vì vứt đi. Đổi lại, phải theo dõi thật và dừng sớm nếu sai.

Cần đọc, theo thứ tự xuất hiện trong log:

| dòng | ý nghĩa | dấu hiệu phải dừng |
|---|---|---|
| `compile OK: max\|dlogit\| … rel\|dgrad\| … flips N of M` | gate compile trên **sm_75** | rơi `DISABLED -> eager` ⇒ chậm ~2,9×, phải tính lại ngân sách |
| `workers ready on <backend> (Ns …)` | startup + backend **thống nhất hai rank** | `forcing eager on all` ⇒ hai rank bất đồng |
| `prepack …s \| 43.045.415 train / 10.761.343 test` | decode đủ dòng | số khác ⇒ sai dataset mount |
| `content_id` | danh tính dữ liệu sau decode | khác giữa các phiên của cùng run ⇒ driver raise |
| `[rNNN] … skip=S/T` | skip warm-up của GradScaler | `S` chạm trần 16 ⇒ `check_updates` raise |
| `vram=…G` | peak VRAM **của một client** | sát 16 GiB |
| `seconds` mỗi round | đã bao gồm commit + W&B | round 2 ≫ round 1 ⇒ có rò rỉ |

Từ round 2 trở đi: `f1_macro` phải **tăng** và `ce_client_mean` phải **giảm**. Round 1 chưng
cất từ teacher khởi tạo ngẫu nhiên (§8.2), nên số của nó không phải chuẩn để so.

Cell cuối in `projected: 50 rounds = X h` tính **chỉ từ round của phiên hiện tại**; thay thẳng
vào bảng §3. Dưới 2 round hoàn tất thì nó **không** in projection — đó là chủ ý.

### 10.3 Nếu một run không đủ 50 round — thủ tục chuẩn (đo thật 2026-09-11)

Hai đường, chọn theo **tài khoản sẽ chạy phiên tiếp**:

| tài khoản tiếp nối | đường | ghi chú |
|---|---|---|
| **cùng** chủ kernel cũ | `--kernel-source <acct>/fd-ids-veremi-<K>-clients` | chưa từng chạy thật trong dự án này |
| **khác** chủ kernel cũ | `--dataset-source <acct>/<ckpt-dataset>` | **đã chạy thật** cho 20c/100c, §15 |

`kernel_sources` sang tài khoản khác **mount rỗng mà vẫn báo push thành công** (đo 2026-09-09,
skill `multi-account.md`). Đường dataset, từng bước, tất cả đã có script:

```bash
# 1. kéo output phiên trước (như chủ kernel cũ) và kiểm
kaggle kernels output <old>/fd-ids-veremi-<K>-clients -p papers/fd-ids-2025/runs/pulls/<run>.s<n> --page-size 200
python scripts/verify_run.py <pull>/runs/<run> --y-true <pull>/runs/<run>/reports/y_true.u8.npy --require-rounds <last>
# 2. dựng dataset: cây run nằm DƯỚI một cấp `runs/` (Kaggle bỏ đúng một cấp khi giải nén)
#    <ds>/dataset-metadata.json {"title","id":"<acct>/<slug>","licenses":[{"name":"CC0-1.0"}]}
#    <ds>/runs/<run>/{weights,resume,metrics,confusion,preds,logs,complete,reports,history.csv}
kaggle datasets create -p <ds> -r zip -t          # -t: KHÔNG chuyển đổi history.csv
#    đối chiếu tên + kích thước từng file với local (scratch ds_inventory.py) trước khi tin
# 3. probe CPU — 0 h GPU, ~35 s — chạy đúng resolve_resume production, assert last == expected
python scripts/gen_resume_probe.py --owner <acct> --clients <K> --expect <last> --run-tag _v2 \
       --dataset-source <acct>/<slug>   # push, đọc output, phải thấy RESUME PROBE OK
# 4. notebook continuation
python scripts/gen_notebook.py --owner <acct> --clients <K> --run-tag _v2 --max-hours <h> \
       --require-resume --dataset-source <acct>/<slug>
python scripts/validate_notebooks.py            # trước khi nhúng key
# nhúng W&B key (embed_wandb_key.py), validate lại, so fingerprint CFG ↔ manifest, rồi push
```

`--run-tag` phải **trùng** phiên trước (`run_name` nằm trong fingerprint và là W&B id).
`require_resume` **chết ở cổng** nếu không tìm thấy checkpoint đã xác minh — trước khi decode
parquet. W&B `resume="allow"` mở lại đúng run cũ, step tiếp tục đơn điệu (đã probe).
Validator: `dataset_sources[:2]` phải là hai mount dữ liệu; phần tử thứ 3 trở đi chỉ hợp lệ
khi `require_resume=True`, và `require_resume=True` bắt buộc phải attach gì đó.

⚠ **Cổng GPU chưa kiểm `last == expected`.** Một import dừng sớm (dataset hỏng một round)
vẫn qua cổng và **train lại** các round thiếu dưới cùng `run_name`; W&B vứt step không đơn
điệu mà không nói gì; `merge_sessions.py` sẽ từ chối ghép vì bytes khác — phát hiện được,
nhưng **sau** khi đã tốn GPU. Probe CPU ở bước 3 là hàng rào cho việc đó; **đừng bỏ bước 3.**

⚠ Không được attach output/dataset của **hai** run khác nhau cùng một `run_name`: `_import_from`
so SHA trọng số và raise kèm cả hai đường dẫn, nhưng đừng dựa vào đó thay cho việc attach đúng.

### 10.4 ⚠ Bảo mật — nhắc lại mỗi lần dùng

3 file `.ipynb` **chứa W&B key**, mode 600, đã `.gitignore`. Không commit, không chia sẻ,
không public hoá notebook. Key nay đã **vĩnh viễn** nằm trong version history của Kaggle cho
cả ba kernel. Muốn thu hồi: revoke ở W&B **trước**, theo `wandb.md` §1.1 — dựng lại notebook
ở local không xoá được gì trên Kaggle.

## 11. Rà soát lần 3 — 2026-09-10 (LỊCH SỬ)

> Giữ nguyên làm hồ sơ. **Mọi mục R11–R18 đã sửa** — trạng thái và bằng chứng ở §13.
> Các bảng tái hiện bên dưới vẫn đáng đọc: chúng nói *cách* bắt được từng lỗi.

### 11.1 Phạm vi và những phần xác nhận lại được

Đọc CONTEXT/TEST_LOG mới, bài báo local, đặc tả knowledge, 7 module, generator, validator,
verifier, 3 test và **cả 5 notebook** (3 run thật + 2 probe). Không thay đổi các quyết định
50×1, batch 512/512/256, Adam, BN aggregation, KD hay chính sách tài khoản. Không sửa module,
test hoặc notebook trong lần rà soát này.

* **35/35 module nhúng khớp nguồn** sau `rstrip()`. Cả 8 định nghĩa chung của mô hình khớp
  AST với `ARCHITECTURE.md`; 395.024 tham số. Các notebook chưa có execution output.
* Ba run thật vẫn 50 round, hai probe 2 round; tham số Eq. (2)–(6), batch và số client đúng
  cấu hình đã chốt. Metadata hiện trỏ đúng FL dataset của từng K và centralized dataset,
  image đã pin, notebook riêng tư và khai báo 2×T4. `kernel_sources=[]` ở cả 5 file:
  đây là bản khởi đầu, chưa phải notebook continuation.
* Không thấy thay đổi làm sai chiều KL, hệ số `T²`, gradient proximal `βμ(w−wG)` hoặc
  aggregation theo `n_k/N`. Bản sửa `applied = steps − skipped` là đúng; vấn đề mới nằm ở
  cách phân loại `nonfinite`, không phải lỗi `max(1,·)` cũ quay lại.
* Gate compile với **compiler giả trả lại chính eager model** nay cho `Δlogit=0`,
  `rel Δgrad=0`, giữ nguyên mọi tensor state. Xác nhận sửa lỗi BN cũ; không phải test
  Inductor/CUDA graph thật.

**Bằng chứng chạy mới** — CPU, `CUDA_VISIBLE_DEVICES=''`, `conda nckh`, từng cây test tuần tự,
bọc `scripts/run_local_checked.py`:

| Kiểm tra | Kết quả lần này | Peak RSS |
|---|---|---:|
| `tests/test_ckpt.py` | **14/14 pass**, rebuild Δlogit=0; weights 1,65 MB, resume 0,01 MB | 693 MiB |
| `tests/test_fdids.py` | **7/7 pass**; client update vs autograd: `4,606e-7` (E=1), `1,568e-7` (E=2) | 764 MiB |
| `scripts/validate_notebooks.py` trên workspace hiện tại | **pass đủ 5 notebook** | 586 MiB |
| Probe artifact/import/AMP/cache/compile identity | Đã tái hiện các ca trong §11.2–§11.7 | 702 MiB |
| Validator bản hiện tại + 3 mutation trong bản sao tạm | Baseline pass; **cả 3 bản sai vẫn exit 0** — R18 | 545 MiB |

Probe runtime **trích module từ chính notebook 20c** vào thư mục tạm rồi import để thử;
35/35 module đã khớp nên các lỗi này có mặt ở cả 5 notebook. Dùng fixture nhỏ, không đọc cả
43 M dòng, không bật compile thật/GPU. Hai script audit tạm là `/tmp/fdids_audit_v3.py` và
`/tmp/fdids_validate_audit_v3.py`; chúng chưa phải regression test được duy trì trong repo.
Cách chèn lỗi được ghi cụ thể bên dưới để chuyển thành test bền vững khi sửa.

Không chạy lại `test_smoke_real.py` nguyên bản: bài này sinh 2 worker CPU và log gần nhất đã
đạt 2.992/3.000 MiB. Bằng chứng smoke/bit-identical ở §7 vẫn lấy từ `TEST_LOG.md` §1b,
**không gọi là phép đo mới**. Các probe mới dùng một tiến trình, không nới watchdog.

MCP native `get_accelerator_quota` trả thành công ở phiên audit này. Chỉ xác nhận MCP có
xác thực; không đối chiếu CLI/account, không cập nhật roster §10 và không dùng quota làm
bằng chứng đã chạy được notebook trên T4.

### 11.2 R11 — P0: AMP overflow bình thường vẫn bị coi là client diverged

**Vị trí:** `proj/fdids.py::client_update`, `proj/driver.py::check_updates` (mở lại R01).
`nonfin += ~isfinite(gn)` chạy trước khi biết bước có bị scaler skip hay không. Gradient đã
overflow ở backward vẫn có norm Inf sau unscale; cộng proximal không phân biệt được nguồn
Inf này. Vì vậy comment “AMP overflow alone cannot cause this” trong driver không đúng.

**Đã tái hiện:** gọi `client_update` thật với `torch.amp.GradScaler('cpu')`, chèn Inf vào
gradient của **backward đầu tiên**, backward thứ hai bình thường. Nhận
`steps=2, skipped=1, applied=1, nonfinite=1`, trọng số cuối **hữu hạn**, nhưng `check_updates`
vẫn raise. Đây là mô phỏng overflow có kiểm soát, không phải số đo tần suất overflow trên T4.
Với scaler reset mỗi client, hiện chỉ cần một bước overflow được scaler xử lý đúng cũng có
thể khiến toàn round bị dừng sau khi đã train tất cả client.

**Hướng sửa/kiểm:** tách loss/logits/proximal không hữu hạn khỏi gradient overflow được scaler
skip; kiểm finite trước/sau khi thêm proximal trên những bước có gradient unscaled hợp lệ.
Không chỉ bỏ `nonfinite` guard: scaler đã kiểm overflow trước phần proximal, nên lỗi xuất
hiện sau đó vẫn phải bị chặn trước optimizer/aggregation. Ghi riêng `attempted/applied/skipped`
và cho phép một **ngân sách skip warm-up hữu hạn, chốt trước probe**. Regression test phải
chạy đường `client_update → worker stats → check_updates`, gồm một vài skip đầu rồi hồi
phục, all-skipped, skip kéo dài, loss NaN và proximal/gradient Inf. Ca hồi phục trong ngân
sách phải được nhận; các ca lỗi phải không sinh marker. Sau đó mới đo AMP thật trên T4.

### 11.3 R12 — P0: import không tôn trọng commit nguồn và có thể nối hai lịch sử khác nhau

**Vị trí:** `proj/ckpt.py::_import_from`, `round_ok`, `resolve_resume`, `rebuild_history`
(mở lại R02/R06).

**Đã tái hiện hai ca riêng:**

1. Nguồn có đủ artifact round 1–2 nhưng **chỉ có marker round 1**. Tại nguồn,
   `last_complete_round(...)=1`; import sang working trống lại trả **2** và tự tạo marker 2.
   `_import_from` không copy/kiểm marker nguồn, chỉ lặp `round_ok(stage,r,fp)` cho đến khi
   thiếu file. Nó biến round chưa commit thành round hoàn tất; lỗi crash này khác ca kiểm
   marker ở đích trong test #13 cũ.
2. Working có round 1 của lịch sử A, attached có round 1–2 của B, cùng cfg/fingerprint nhưng
   **trọng số round 1 khác nhau**. Import vẫn trả 2, giữ round 1 của A rồi lấy round 2 của B.
   `round_ok(d,r,fp)` chỉ cho biết cfg phù hợp, không chứng minh cùng lịch sử huấn luyện.

**Hướng sửa/kiểm:** xác định dải **đã commit ở nguồn** bằng marker và nội dung hợp lệ trước
khi staging, tuyệt đối không publish round vượt dải đó. Khi có overlap, so hash trọng số/
artifact của từng round và nhận diện run; khác nhau thì dừng với hai đường dẫn, không đoán.
Giữ `run_name` cho scenario nhưng thêm identity của lần train và liên kết checkpoint trước
nếu cần phân biệt fresh run với continuation. Publish bằng `.part → replace → marker`; xử lý
marker cũ trước khi thay bộ file mà nó đang xác nhận. Đừng chỉ kiểm kích thước file.

Thêm test nguồn thiếu marker cuối/giữa, nguồn bị cắt ở từng artifact, đích import bị ngắt,
overlap giống/khác trọng số, nhiều attachment, nguồn khác fingerprint. `rebuild_history`
phải chỉ lấy **dải commit hợp lệ liên tục**; hiện nó lấy mọi marker, có thể giữ dòng sau gap,
và không xoá CSV cũ khi không còn round hợp lệ. Không âm thầm khởi tạo fresh run trong một
thư mục đã có checkpoint không tương thích.

### 11.4 R13 — P1: verifier còn nhiều trường hợp báo pass sai

**Vị trí:** `proj/verify.py::verify_run`, `proj/ckpt.py::round_ok`, `scripts/verify_run.py`
(mở lại R08/R10). Fixture baseline 2 round ban đầu pass; mỗi ca dưới đây sửa **một bản sao
độc lập**, vẫn gọi verifier với `build_model`, `expect_params=395024`, `y_true_path` và
**`require_rounds=2`**:

| Chèn lỗi | Kết quả quan sát |
|---|---|
| Xoá dòng round 2 trong `history.csv` | **pass** |
| Thêm dòng round 1 trùng trong CSV | **pass** (dict theo round che mất duplicate) |
| Accuracy CSV hoặc metrics JSON = NaN | **pass** (`abs(NaN) > tolerance` là False) |
| Per-class F1 = 999, support giữ đúng | **pass**, chỉ support được đối chiếu |
| Xoá toàn bộ `preds/`, dù đã truyền y_true | **pass** |
| Xoá toàn bộ `logs/` | **pass** |
| Ghi bytes rác vào `resume/round_001.pt` | **pass**, file chỉ bị kiểm tồn tại/kích thước |
| CFG khai `n_test=999`, CM/preds thực có 16 dòng | **pass**, chỉ so tổng giữa các round |

**Hướng sửa/kiểm:** tách chế độ kiểm checkpoint tối thiểu với chế độ nghiệm thu đầy đủ.
Ở chế độ đầy đủ, bắt buộc tập round CSV đúng `1..last`, không duplicate/extra; mọi metric hữu
hạn và đủ khoá ở JSON/weights/CSV. Kiểm CM nguyên không âm, shape đúng `num_classes`, tổng
bằng `n_test`; đối chiếu đủ idx/tên lớp/support/precision/recall/F1. Kiểm log đủ và đúng tập
client, `n_k`, seed, `steps=E*ceil(n_k/B)`, `applied+skipped=steps`, số liệu hữu hạn.
Resume RNG chỉ là truy vết theo quyết định §1.2; nếu tiếp tục coi file này là artifact bắt
buộc thì phải đọc được, đúng round/fingerprint, không chỉ có tên file.

Predictions thiếu phải báo **chưa kiểm đủ**, hoặc lấy từ nguồn phiên trước đã định danh và
kiểm hash (R15). Chuyển từng ca trong bảng thành regression test bắt buộc fail. Giữ test
fixture hợp lệ pass để tránh verifier chỉ biết từ chối. `--require-rounds 50` hiện chỉ kiểm
độ dài dải round, **chưa đủ để chứng nhận chất lượng artifact**.

### 11.5 R14 — P1: data identity và cache vẫn chưa nhận diện đúng dữ liệu

**Vị trí:** cell prepack trong `scripts/gen_notebook.py`, `proj/data.py` (mở lại R05/R07).
Biểu thức `data_id` hiện chỉ chứa features, classes, K và scaler; không đọc parquet, phiên
bản dataset, hàng thuộc client nào hay thứ tự fragment test. Thay nội dung/hoán nhãn/đổi
partition nhưng giữ các mục này sẽ giữ nguyên fingerprint. Comment nói nhận diện “different
partition” đang vượt quá điều code làm được.

Cache chỉ xét `have == want`, không xác thực FILES ở nhánh hit. **Đã chạy chính nhánh If
của notebook** với manifest khớp nhưng không có `train_X`: vẫn in “cache reusable”. Sau đó
worker mới lỗi khi mở file; trường hợp file có mặt nhưng giá trị cũ/sai có thể khó phát hiện
hơn. Assert mang tên `X/y row mismatch` hiện chỉ kiểm chiều dài **y**, chưa kiểm shape/dtype X.

**Hướng sửa/kiểm:** lưu dataset slug/version và manifest file theo thứ tự ổn định, identity
partition/test vào cả fingerprint run và cache. Footer/schema/statistics giúp phát hiện
nhiều thay đổi với chi phí thấp nhưng **không phải checksum nội dung tuyệt đối**; cần ghép
version bất biến hoặc hash nội dung đã tính lúc audit/build dataset để bảo vệ cả trường hợp
rewrite giữ nguyên thống kê. Không quét lại 43 M dòng ở mỗi lần require-resume chỉ để tạo hash.

Cache hit phải kiểm đủ file, header/shape/dtype, X/y cùng số dòng, spans liên tục và đúng
client; manifest hỏng/incomplete thì rebuild an toàn. Decode đối chiếu `client_stats` và
phân bố lớp, rồi mới ghi marker. Viết fixture parquet nhỏ và chạy **load_clients/load_test
thật + cell prepack**: reorder feature, sentinel mơ hồ, nhãn ngoài miền, scaler đổi, rewrite
parquet giữ count, file cache thiếu/truncated, thứ tự test đổi. Smoke hiện tự dựng mảng bằng
code trong test, **không gọi hai loader production**, nên pass cũ chưa đóng phần này.

### 11.6 R15 — P1: output chưa đủ tự kiểm offline qua nhiều phiên

**Vị trí:** generator/prepack/summary, `driver.write_manifest`, `ckpt.RESUME_SUBDIRS`.
`y_true` chỉ nằm trong **`/kaggle/temp/veremi_cache/test_y.u8.npy`**, không được đưa vào
`/kaggle/working/runs`. Manifest có feature/class order nhưng **không lưu giá trị scaler**,
chỉ giữ `data_id` và đường mount nguồn. Vì vậy tải riêng run directory về chưa có đủ đầu
vào để gọi `--y-true` hoặc preprocess/dựng lại phép suy luận độc lập.

Không import preds/logs là **quyết định có chủ đích ở §2**, nhưng hiện chưa lưu liên kết
nguồn/hash cho chúng. Probe import xác nhận đích có **0 file preds, 0 file logs** ở các round
đã import. Qua continuation A→B→C, output cuối không tự chứa bằng chứng của A/B; verifier
hiện có thể vẫn nói pass vì bỏ qua chúng (R13).

**Hướng sửa/kiểm:** lưu y_true một lần trong output (≈10,76 MB), scaler thực dùng và manifest
test gồm thứ tự fragment/row identity. Code model đã được `%%writefile` vào
`/kaggle/working/proj`; khi đóng gói riêng run cần kèm code đó hoặc phiên bản/hash và recipe
phục dựng rõ ràng. Với preds/logs, hoặc import đầy đủ (tổng dự kiến vẫn khoảng 630 MB/scenario),
hoặc giữ cơ chế không copy nhưng xuất manifest liên kết từng round với output/version/hash
nguồn, rồi verifier truy xuất đủ chuỗi. Không đổi quyết định lưu trữ ngầm trong một lần audit.

Kiểm 3 phiên fixture 1→2→3, tải các artifact theo đúng quy trình dự định, bỏ cache tạm và
mount gốc, sau đó verify offline. CM từ preds chỉ chứng minh hai artifact phù hợp với nhau;
muốn nối bằng chứng về trọng số, thêm test load checkpoint rồi inference lại trên tập dòng
thật cố định với tolerance/backend đã ghi, thay vì chỉ strict-load một mô hình hữu hạn.

### 11.7 R16 — P1: gate compile và hợp đồng RNG chưa đủ đại diện đường train

**Phần đã đạt:** identity compiler giữ nguyên state và cho sai số 0 — lỗi BN cũ đã sửa.
**Phần thấy từ code, chưa phải kết luận compile T4 hỏng:**

* `_compile.probe` chỉ forward + CE backward với **nhãn ngẫu nhiên**, chưa chạy KD,
  proximal, GradScaler, clipping và Adam step như production. Warmup 3 lần cũng không update
  optimizer. Chưa kiểm vòng reset weights/moments giữa nhiều client và chuyển compiled batch
  đầy → eager tail → compiled batch kế tiếp.
* `n_bad` chỉ đếm top-1 ở những dòng có margin `> max(10*dz,1e-3)`. Tập này có thể rỗng;
  gate vẫn có thể đạt điều kiện `n_bad=0`. Đây không phải “0 dòng lật top-1 trên cả batch”
  như ngưỡng §7/§9. Cần in cả số flip toàn batch và số dòng được kiểm, thực hiện đúng ngưỡng
  đã chốt; nếu muốn convention margin khác thì phải ghi rõ trước khi đo, không nới hậu nghiệm.
* Hai worker tự gate/fallback độc lập; rank có thể chọn compiled hoặc eager khác nhau.
  `compile` không có trong fingerprint và log client không ghi backend thực dùng. Seed CPU
  cho kết quả đúng lịch chưa chứng minh CUDA/compiled Dropout tương đương hay replay được
  qua thay đổi backend/image. Không kết luận compiler sai chỉ vì RNG hai backend khác nhau.
* Test Eq. (6) dùng ZT dựng trong test, chưa bao phủ `teacher_logits` production. Cache fp16
  và batch teacher lớn cần được so với online teacher trên cùng dòng; “Exact” trong docstring
  chỉ đúng về teacher cố định, không bảo đảm số học mọi backend — §8.4.

**Hướng kiểm:** fixture CPU kiểm teacher/anchor bất biến và cached/online logits→KL→gradient;
T4 so nhiều bước client update đầy đủ, bao gồm tail, reset giữa client và round, dữ liệu thật
từ nhiều client cùng các giá trị biên. Tách phép kiểm số học có cùng randomness với phép kiểm
Dropout/RNG production; không đổi dropout của run thật để làm gate pass. Ghi backend thực,
torch/CUDA/image và chỉ chốt tính độc lập lịch sau khi đổi thứ tự giao client giữa hai rank
và replay continuation đạt tolerance đã thống nhất. Có thể buộc cùng backend cho mọi worker
nếu chưa có bằng chứng an toàn khi trộn backend.

### 11.8 R17 — P1 cho budget/probe: thời gian và projection còn sai phạm vi

**Vị trí:** `driver._rounds`, cell tổng kết của generator (mở lại R09).

1. `row['seconds']` được chốt sau weights/preds/logs nhưng **trước** metrics JSON, CSV,
   marker và W&B. Dự báo lấy worst của giá trị này nên vẫn thiếu đuôi round. Kiểm budget
   chỉ sau round, không trước round đầu; spawn/prepack/compile có thể đã hết budget mà vẫn
   bắt đầu train. Đây là giới hạn dừng dự báo, không phải hard deadline.
2. Khi continuation, `rows` gồm toàn bộ history import; `overhead=(now−T0)−sum(sec)` lại lấy
   thời gian **phiên này trừ tất cả phiên**. Ví dụ giả định 50 round × 600 s trong history,
   phiên tiếp nối chỉ chạy round cuối và kéo dài 1.000 s: overhead thành **−29.000 s**, và
   projection 50 round thành 1.000 s. Đây là phản ví dụ tính toán, không phải benchmark T4.
3. Summary tính overhead sau verifier nên lẫn thời gian verify/shutdown/W&B vào nhãn
   “startup+prepack”. `vram_gb` lấy max-memory từ đầu worker và ghi trước eval round đó;
   chưa phải peak riêng từng phase/từng GPU. Log `sec` gộp teacher và train; chưa dùng nó để
   “đo trực tiếp” chi phí cố định mỗi client rồi ngoại suy chính xác cho 50c.
4. Có thể chỉ có 1 round probe nhưng summary vẫn in `steady/projected`. `_collect` đợi đến
   client hoàn tất, chưa có heartbeat tiến độ batch; shutdown có thể chờ 60 s mỗi worker.

**Hướng sửa/kiểm:** dùng monotonic deadline từ đầu session; kiểm trước startup/round đầu và
mỗi round, ghi các phase startup/prepack/compile/train/teacher/eval/commit/verify rõ ràng.
Phần đo tổng round dùng cho budget phải bao gồm commit và có dự phòng finalization. Tính
projection từ **round thuộc session hiện tại**; tách thời gian lịch sử, không sinh overhead
âm khi resume. Hai round hoàn tất là tối thiểu cho calibration; thiếu thì báo chưa đủ số đo,
không dùng projection làm điều kiện duyệt run dài. Ghi heartbeat thưa theo client/batch.

Test bằng fake clock cho hết budget trước round đầu, I/O chậm, continuation có history dài
và probe thiếu round. Sau đó đo T4 đủ hai batch 256/512; 50c vẫn có sai số ngoại suy cần được
ghi rõ hoặc đo thêm trước khi chốt ngân sách sát quota. `train_ce/train_kd` hiện là trung
bình **đều giữa client** của trung bình bước đã áp dụng, không phải loss trung bình theo mọi
mẫu train; ghi đúng tên/phạm vi để tránh đọc nhầm khi so K khác nhau.

### 11.9 R18 — P2: validator chưa kiểm effective CFG và luồng như mô tả

**Đã thử trong bản sao tạm của repo**, giữ source module khớp, sửa độc lập notebook 20c:

| Mutation | Validator |
|---|---|
| Thêm `CFG['lr'] = 0.25` sau `CFG = dict(...)` | **exit 0** |
| Dời `MF.write_text(...)` lên trước `load_clients`/ghi các mảng | **exit 0** |
| Thay dataset centralized trong metadata bằng dataset không liên quan | **exit 0** |

Parser chỉ đọc khai báo CFG đầu tiên, không kiểm assignment về sau. Kiểm marker dùng
`flow.index('test_y.u8.npy')`, nhưng lần xuất hiện đầu là chuỗi tên file trong `FILES`,
không phải lệnh `np.save`. Dataset chỉ kiểm số lượng và substring của FL dataset thứ nhất.
**Các notebook hiện tại chưa bị các mutation này**; đây là lỗ hổng cổng kiểm cho lần regenerate
tiếp theo, không phải bằng chứng CFG hiện đang là lr=0,25.

**Hướng sửa/kiểm:** whitelist các cập nhật CFG hợp lệ (`data_id`, `n_test`), chặn ghi đè
tham số khoa học ngoài khai báo đã parse; kiểm AST của lệnh ghi marker so với các lệnh ghi
file thật trong cùng nhánh. So chính xác cả hai dataset slug, metadata accelerator/internet
và tập notebook bắt buộc 20/50/100. Duy trì ba ca mutation trên làm test của validator.
Tiếp tục kiểm byte module; không dùng kiểm tĩnh thay cho test cell prepack hoặc driver.

## 12. Thứ tự sửa và nghiệm thu sau rà soát lần 3 (LỊCH SỬ)

> Bước 1–3 (local) **đã đạt** — §13. Bước 4 (probe T4) được gộp vào bước 5: round đầu của
> chính ba run thật là phép calibration, xem §10.2. Bước 5–6 đang chạy.

Nguyên tắc vẫn giữ nguyên và vẫn áp dụng: fixture tái hiện phải thành **file test riêng**
chạy qua watchdog, không phình `test_smoke_real.py`; test phải gọi đường production; số đo cũ
trong TEST_LOG chỉ được **ghi thêm**, không sửa cho trông như đã pass ca mới.

---

## 13. R11–R18 — đã sửa, kèm bằng chứng

Mô tả đầy đủ từng lỗi ở `TEST_LOG.md` §2 dòng #21–#35; số đo ở §1c.

| ID | sửa ở đâu | bài kiểm chứng minh | số đo |
|---|---|---|---|
| **R11** overflow AMP bị coi là divergence | `fdids.client_update` — chỉ đếm non-finite trên bước scaler **đã áp dụng**; `driver.check_updates` thêm ngân sách skip và kiểm `applied+skipped==steps` | `test_amp_guard.py` **8/8** | overflow+recovery: `nonfinite` **1 → 0**, `check_updates` REJECTED → ACCEPTED; biên ngân sách 16 nhận / 17 từ chối |
| **R12** import nối hai lịch sử | `ckpt.save_round_weights` ghi `prev_sha`; `_import_from` copy và tôn trọng marker **nguồn**; `round_ok` kiểm chuỗi | `test_resume_import.py` **7/7** | nguồn marker 1/2 → import **đúng 1**; overlap khác trọng số → `RuntimeError` nêu cả hai đường dẫn |
| **R13** verifier báo pass sai | `verify.py` viết lại: `isfinite` **trước** phép so, recompute đủ per-class, bắt buộc preds/logs/manifest, CSV phải đúng `1..last` | `test_verifier.py` **13/13** | 12/12 ca tamper bị chặn; fixture hợp lệ vẫn pass |
| **R14** data/cache identity | `data.cache_ok` (kiểm file+shape+dtype+spans); `content_id` sau decode; `write_manifest` là cổng thứ hai | `test_data_cache.py` **8/8** | 8/8 ca cache hỏng bị chặn; nhãn −1 và 16 đều bị chặn |
| **R15** chưa tự kiểm offline | `y_true` copy vào `reports/`; scaler vào manifest; `RESUME_SUBDIRS` thêm `preds`, `logs` | `test_smoke_real.py` | `verify_run` đọc y_true **từ run**, không từ cache tạm |
| **R16** gate compile/teacher | probe chạy đúng loss production (CE+KD+proximal+clip); in flip trên **cả** batch; từ chối nếu tập decisive < 10%; driver ép hai rank cùng backend | `test_teacher.py` **3/3** | cache teacher **không** exact: `max\|ΔZt\| = 1,2×10⁻⁴`; KD loss `Δ = 0`, gradient `rel 1,1×10⁻⁵` |
| **R17** đồng hồ/projection | `time.monotonic()`; kiểm budget **trước** round đầu; `finalize_reserve_seconds`; projection từ `hist` phiên hiện tại; VRAM reset mỗi phase; đổi tên `ce_client_mean`/`kd_client_mean` | `test_budget.py` **5/5** | hết budget trước round 1 → **0 round, không commit gì**; budget đủ 1/5 round → dừng sạch, round đã commit nguyên vẹn |
| **R18** validator | chặn ghi `CFG` ngoài whitelist; thứ tự marker kiểm bằng **AST**; so chính xác cả hai dataset slug; thêm kiểm `id` ↔ slug-từ-title | `test_validator.py` **8/8** | 7/7 mutation bị chặn, gồm cả 3 ca của §11.9 |

**Hai lỗi lộ ra *sau* khi các cổng local đều xanh**, và cả hai chỉ lộ khi chạm thật:

| # | lỗi | lộ ra ở đâu |
|---|---|---|
| 36 | sidecar `.stats.json` làm `ds.dataset(dir)` chết ở `load_test` | **lần push đầu lên T4** — local pass vì smoke test đọc lazy rồi `break` trước khi chạm sidecar |
| — | Kaggle lấy slug từ `title`, bỏ qua `id` | **lần push đầu** — kernel ra đời ở `fd-ids-veremi-100-clients`, `id` đã khai thành vô dụng |

Bài học chung, ghi để phiên sau không lặp: **một fixture do mình dựng chỉ kiểm được điều mình
đã nghĩ tới.** Cả hai lỗi trên đều nằm ngoài trí tưởng tượng của 76 phép kiểm local, và cả hai
đều rẻ khi gặp sớm (0,08 h quota) — đó là lý do round đầu phải được theo dõi thật (§10.2)
chứ không phải push rồi bỏ đó.

---

## 14. Phiên 2026-09-11 — compile đã được mở khoá, relaunch cả ba

**Kết quả lớn nhất của phiên: ba run T4 chạy chậm gấp đôi vì một lỗi trong GATE của tôi,
không phải vì Triton hỏng trên sm_75.** Chi tiết ở lỗi **#39** (`TEST_LOG.md` §2) và số đo ở
`TEST_LOG.md` §3.3.

### 14.1 Vì sao compile bị tắt suốt — lỗi #39

`_compile` so một bước train của model eager với model compiled. `restore()` đặt lại RNG của
torch nên **eager** lặp đúng mask dropout cũ, nhưng **Inductor functionalise RNG** và tự rút
offset Philox riêng ⇒ mask dropout hai bên không bao giờ trùng. Gate đo chênh lệch giữa hai
mask rồi gọi đó là lỗi compiler:

| dropout | `max\|dlogit\|` | kết quả |
|---|---:|---|
| 0,1 (production) | **6,07e-01** | `10*dz = 6,07` > mọi margin ⇒ `n_dec = 0` ⇒ `cannot certify` ⇒ **rơi eager** |
| 0,0 | **7,32e-04** | **compile OK** |

**Sửa:** tắt dropout (`m.p = 0`) cho **cả hai bên** chỉ trong lúc so sánh, khôi phục trong
`finally` **và** ở nhánh `except`. Warm-up phía trên đã capture graph ở p production nên
production dùng lại đúng entry đó — đã kiểm trên đường CUDA-graph thật ở local (probe chạy
`client_update` với `comp` ở p=0,1 sau gate, 11,6 ms/step, không lỗi).

Bắt được ở **local sm_86**, nơi Triton không hề bị nghi ngờ — đó chính là điều loại bỏ giả
thuyết "sm_75 hỏng". `tests/test_compile_gate.py` (6 phép kiểm) fail trên code chưa sửa và
pass sau khi sửa; đã **kiểm chứng chèn lỗi** bằng cách revert tạm đúng đoạn guard.

### 14.2 Số đo T4 và ảnh hưởng ngân sách

`certified=True` ở **mọi** cấu hình trên T4 (`minhtran0601`, ~0,15 h quota):

| cấu hình | eager | compiled | speedup |
|---|---|---|---:|
| B=512, 2 proc (20c/50c) | 21,5–23,7 ms/step | 9,7–10,6 | **2,22–2,35×** |
| B=256, 2 proc (100c) | 23,1–24,5 ms/step | 8,8–10,1 | **2,63×** |

Round **17,9 ph → ≈10,5 ph**; 50 round **14,9 h → ≈8,8 h/kịch bản**. Teacher (221 s) và eval
(~40 s) **không** co lại — teacher chạy batch 16.384, compute-bound. Phần lợi của
`torch._foreach_*` (proximal, bit-identical, `max|Δ| = 0`) **cộng thêm** vì nó nằm **ngoài**
graph compiled; probe đã push mang `fdids.py` bản cũ nên chưa có trong số trên.

### 14.3 ⚠ Hai chẩn đoán SAI của tôi trong phiên này — đừng lặp lại

1. **"Round 3 chậm thật."** Sai. Round 3 xong lúc 17:04:40Z, `gap` = 17,9 ph, đúng nhịp.
   Lý lẽ hỏng: *"system stream tươi ⇒ sender không tắc ⇒ history cũng tươi"*. Hai stream được
   **nạp riêng**; endpoint history của W&B trễ **25–30 phút**. Tín hiệu đáng tin duy nhất là
   **một hàng MỚI thực sự xuất hiện** — "chưa có hàng" không phân biệt được *trễ* với *chậm*.
2. **"T4 chậm 1,34× so với eager-local."** Sai. So dự đoán **thời gian/step** với wall-clock
   **cả round** (có teacher + eval). Trừ đúng: `(1074−221−40)/42.041 ≈ 19,3 ms/step`, khớp
   eager-local 19,07. Kết luận "đã rơi eager" thì đúng, lập luận thì không.

Thêm một bẫy API: `scan_history(keys=[...])` chỉ trả hàng có **đủ mọi key**; hỏi `round`
(key bị loại khỏi log, lỗi #38) ⇒ **0 hàng**, trông y hệt run chưa chạy gì. Đọc không lọc key.

### 14.4 Trạng thái repo — build và push ĐÃ đồng bộ

`proj/driver.py` (#37, #38, #39) và `proj/fdids.py` (`_foreach`) đã vào cả ba notebook.
`scripts/validate_notebooks.py` **XANH** cả trước và sau khi nhúng key. Notebook đều `0600`;
`.gitignore:6` loại `papers/**/notebook/**/*.ipynb`. **Key W&B nhúng trong notebook là vĩnh
viễn trong version history của Kaggle — không commit, không chia sẻ các file `.ipynb` này.**

Suite local: **13 file, 86 phép kiểm, pass toàn bộ** (thêm `test_compile_gate.py` 6 và
`test_proximal.py` 4). `scripts/gen_compile_probe.py` sinh notebook probe khi cần; thư mục
`notebook/compile_probe/` đã **xoá** sau khi probe trả lời xong (regenerate được, kết quả đã
ghi ở `TEST_LOG.md` §3.3). Validator **không** quét thư mục đó.

**`gen_notebook.py --run-tag`** (mới): hậu tố cho `run_name`. Cần cho mọi lần relaunch, vì
`wandb.init(resume="allow", id=run_name)` sẽ **mở lại đúng run cũ**; run cũ đã có history tới
step 3 nên `log(step=1..3)` là **không đơn điệu** và W&B **vứt bỏ** — run trông vẫn sống mà
không báo gì. `run_name` cũng nằm trong `FINGERPRINT_KEYS` nên bản tag không thể resume từ,
hoặc bị ghép vào, checkpoint của bản không tag. Cả hai đều là điều một lần relaunch cần.

**`validate_notebooks.py`** nay lấy tên notebook từ `kernel-metadata.json` (`code_file`) thay
vì tự suy ra `fdids_{K}c.ipynb` — nó **crash** ở lần relaunch đầu vì file thật là
`fdids_20c_v2.ipynb`. Có thêm phép kiểm hình dạng `run_name` để không nới lỏng gì.

### 14.5 Cancel: chỉ người dùng làm được

**Cả bốn MCP bearer đã chết** (`Unauthenticated` với mọi tài khoản), và Kaggle CLI **không có**
lệnh `cancel` — chỉ có `list/files/get/init/push/pull/output/status/logs/update/delete/topics`.
Nên huỷ session là việc **người dùng** làm trên UI. Mint lại bearer cần đăng nhập trình duyệt,
thứ mà chính sách tài khoản dành riêng cho người dùng.

### 14.6 Relaunch 2026-09-11 — đã push, version 3

Người dùng đã dừng ba session v1; huỷ **trả lại** phần reserve: `khanhmay0304` 6,17 → **16,09 h**.
Quota lúc push: `khanhmay0304` 16,09 · `minhtrit06` 6,58 · `minhtran0601` 2,13 · `odixe0502` 0,66.

| kernel | `run_name` | `max_hours` | startup | round | **đủ trong phiên** | 50 round cần |
|---|---|---:|---:|---:|---:|---:|
| `minhtrit06/fd-ids-veremi-20-clients` | `fdids_20c_v2` | 6,4 | 12,6 ph | **521,1 s** | ~41/50 | 7,6 h |
| `khanhmay0304/fd-ids-veremi-50-clients` | `fdids_50c_v2` | 9,7 | 13,1 ph | **531,5 s** | **50/50 ✅** | 7,8 h |
| `khanhmay0304/fd-ids-veremi-100-clients` | `fdids_100c_v2` | 6,2 | 16,9 ph | **737,9 s** | ~28/50 | 10,7 h |

⚠ **Startup thật chỉ 12,6–16,9 phút, KHÔNG phải ~40 phút** như ước tính cũ. Con số 40 phút là
suy từ v2 (`_runtime` của round 1 = 41,4 ph) mà **quên trừ chính round 1** — vốn tốn ~18 ph ở
eager. Dùng `_runtime − seconds` của round 1 để lấy startup, đừng dùng `_runtime` trần.

⚠ **100c BỊ NGHẼN CPU và round trôi lên** — 738 → 903 s qua 10 round, trong khi 20c/50c đứng
yên trong 3 s suốt 20 round. **Không phải nhiệt** (cả ba cùng 67 W, 77 °C) và **không phải rò
bộ nhớ** (`vram_train_gb` = 7,13 cố định, train+eval **không đơn điệu**). Bằng chứng quyết
định là đọc **util cùng với clock**: 20c/50c **100% util ở 1245–1290 MHz** (bão hoà, bị trần
công suất ghì), còn 100c **58% util ở 1575/1590 MHz** — *boost được vì đang rảnh*, tức **đói
CPU**. Batch 256 ⇒ **168.200 bước/round** (gấp đôi 84.083) trên cùng **4 vCPU**. Vì vậy 100c
được **~23 round**, không phải 28. Chi tiết: `TEST_LOG.md` §3.4; bài học đã ghi vào skill
`references/perf-federated.md` §2.

### ⚠⚠ PHÁT HIỆN KHOA HỌC: cả ba run ĐẠT ĐỈNH SỚM rồi THOÁI LUI

| run | `f1_macro` tốt nhất | ở round | tại round 26 | accuracy đỉnh → hiện tại |
|---|---:|---:|---:|---|
| 20c | **0,7822** | **7** | 0,7676 | 0,8161 (r4) → 0,7401 |
| 50c | **0,7205** | **6** | 0,6722 | 0,7642 (r3) → 0,6669 |
| 100c | **0,6870** | **9** | 0,6746 (r16) | 0,7192 (r3) → 0,6729 |

`ce_client_mean` là **loss huấn luyện phía client**, và nó giảm **đơn điệu** suốt (20c:
0,649 → 0,229). Loss train giảm trong khi metric test giảm ⇒ **overfitting**, nói thẳng như vậy.
`f1_weighted` rơi mạnh nhất (20c 0,8083 → 0,7421); `recall_macro` lại **tăng** (0,7685 → 0,7822)
còn `precision_macro` giảm ⇒ model toàn cục đang **đánh đổi độ đúng lớp đa số lấy recall lớp
hiếm**, tức tích luỹ false positive.

**Giả thuyết (CHƯA kiểm chứng, đừng chép vào báo cáo như kết luận):** DAGSNet có **BatchNorm**,
và Eq. (2) FedAvg **trung bình cả running stats của BN** cùng với trọng số. Với phân hoạch
non-IID Dirichlet α = 0,5, các thống kê đó phân kỳ giữa client và việc trung bình chúng được
biết là gây thoái lui dần. **Model của chính bài báo là DNN 5 lớp, KHÔNG có BatchNorm** — nên
tương tác này đến từ **phép thay thế bộ phân loại của chúng ta**, không phải từ phương pháp
của bài báo. Muốn khẳng định thì phải làm thí nghiệm riêng trên checkpoint đã lưu (ví dụ:
đánh giá lại với BN stats tính lại trên dữ liệu giữ riêng, so với BN stats trung bình).

**Hệ quả cho báo cáo:** round cuối **không phải** round tốt nhất. Vẫn chạy đủ 50 round theo
đúng bài báo, nhưng khi báo cáo phải neo vào **đường cong**, kèm round đạt đỉnh, chứ không
trích một con số ở round 50. Đây chính là lý do hợp đồng artifact bắt buộc lưu checkpoint và
đủ 10 metric **mỗi round**.

✅ **Skip đã bão hoà rồi đảo chiều** (20c: 5→…→40→38→31→32): `GradScaler` chia đôi mỗi lần
overflow nên skip/client là **logarit** theo độ lớn gradient. Ngoại suy tuyến tính là sai mô
hình. 50c skip = 0 suốt. Mối lo này đóng lại.

✅ **`backend=compiled` trên cả ba**, đọc được khi run **còn sống** (nhờ bản vá #38).
**17,9 → 8,7 ph/round = 2,06×**, tốt hơn dự đoán 10,5 ph. `ce` khớp đường eager tới 4 chữ số
ở cả round 1 và 2 ⇒ thay đổi compile **không** làm lệch bài toán tối ưu. Chi tiết và bảng đối
chiếu eager-vs-compiled: `TEST_LOG.md` §3.4.

`minhtran0601` (2,13 h) **để trống có chủ ý**: một session 2 h mất ~0,7 h cho prepack, gộp
100c vào `khanhmay0304` (2 session đồng thời, 9,7 + 6,2 = 15,9 ≤ 16,09) cho nhiều round hơn.

⚠ **100c tốn gấp đôi mỗi round**: batch 256 ⇒ 43 M/256 ≈ **168.000 bước** (84.000/GPU) so với
84.000 (42.000/GPU) của 20c/50c. Compiled ~9,0 ms/step ⇒ train 756 s + teacher 221 s + eval
≈ **17 ph/round**, tức 50 round ≈ **14,2 h** — không phải 8,8 h như 20c/50c. Tính ngân sách
continuation theo con số này.

### 14.7 Việc tiếp theo

1. Theo dõi round đầu qua W&B. **Kiểm `summary["backend"]`** — nhờ bản sửa #38 nó xuất hiện
   ngay sau khi worker ready, và đó là cách duy nhất nhìn từ ngoài xem gate có certify trong
   production không. Nhớ độ trễ 25–30 phút của endpoint history (§14.3).
2. ~~Continuation sau quota refresh~~ — **đã làm sớm hơn** trên `minhtriethihi` (§15), không
   cần chờ refresh.
3. Chưa làm, chưa đo: teacher chiếm ~33% mỗi round sau khi compile bật (221 s trên ~630 s).
   Chạy batch 16.384 nên compute-bound, không chắc compile giúp được — cần đo trước khi sửa.
4. ~~MCP bearer cả bốn tài khoản đều chết~~ — **SAI**, xem §15.1: cả 4 bearer
   `IntrospectToken` ⇒ `active=True`. "Chết" là kết luận từ probe trực tiếp, thứ
   `kaggle-credentials.md` đã ghi là không dùng được để chấp nhận/loại token. Cancel qua MCP
   native sau `/mcp` reconnect **chưa thử lại**.

### 14.8 Đã ghi vào skill cho các bài báo sau

Bài học #39 là **liên dự án**, không riêng FD-IDS, nên đã ghi vào skill (cả hai cây
`.agents/` và `.claude/`, `references/` **giống hệt nhau** — kiểm bằng `diff -rq`):

* `references/perf-federated.md` §5 — mục *"Four traps in the validation harness itself"*
  (trước là "Two", vốn đã sai vì đang liệt 3): **trap 4** = tắt dropout ở **cả hai** phía khi
  so eager với compiled, kèm bảng 6,07e-01 vs 7,32e-04, cách nó phá ngưỡng decisive-margin của
  trap 2, đoạn code có `finally`, và ba bài học đi kèm — *tái hiện gate trên GPU local trước
  khi đổ lỗi cho kiến trúc đích*, *fallback im lặng cần một field ồn ào*, *đừng so dự đoán
  per-step với wall-clock cả round*.
* `references/perf-2xT4.md` §torch.compile — hai gạch đầu dòng ngắn + trỏ sang trap 4, vì một
  dự án **không** federated sẽ đọc file này trước.

Ghi chú: skill §3 của `perf-federated.md` **đã** khuyến nghị `torch._foreach_*` cho proximal
từ trước; `client_update` nay mới thật sự làm theo.

---

## 15. Phiên 2026-09-11 (phiên 2) — phiên 1 xong, tiếp nối trên tài khoản mới

### 15.1 Tài khoản `minhtriethihi` (24521851@gm.uit.edu.vn) — 30 h GPU, 20 h TPU

* Token `KGAT_` chủ dự án đưa đã lưu ở `~/.kaggle/accounts/minhtriethihi.mcp-token` (0600,
  thư mục 0700; **không** in ra ở đâu). `IntrospectToken` với chính token đó ⇒ `active=True,
  username='minhtriethihi'`, quota 30,00 h, refresh 2026-09-12T00:00Z — **danh tính do server
  xác nhận**, không phải từ tên file.
* **Phát hiện:** token `KGAT_` là **API token của CLI 2.2.4** (`kagglesdk` đọc
  `KAGGLE_API_TOKEN`, và chấp nhận **đường dẫn file**). Mọi thao tác CLI của phiên này —
  push probe, push 2 kernel GPU, tạo 2 dataset, `kernels status/files` — chạy qua
  **`kaggle_as.py <user> -- <lệnh>`** (mới, ở `.agents/skills/kaggle-training-notebook/scripts/`, mirror `.claude/`): OAuth snapshot ⇒ mint access token trong
  bộ nhớ + introspect; token-only ⇒ truyền đường dẫn file. **Không đổi login active**, không
  cần `/mcp` reconnect, không cần snapshot cho tài khoản chỉ có token.
* **Chưa có OAuth snapshot** ⇒ `kaggle_account.py use/ensure/health/quota/plan` **chưa quản lý
  được** tài khoản này (`list` sẽ báo half-installed). Chủ dự án đã đồng ý đăng nhập trình
  duyệt: `conda run -n nckh python .agents/skills/kaggle-training-notebook/scripts/kaggle_account.py add-account minhtriethihi`
  **ĐÃ LÀM 01:57Z:** chủ dự án đăng nhập trình duyệt (chạy `auth login --force` trực tiếp,
  không qua `add-account`, nên **không có snapshot**; `minhtran0601` an toàn nhờ snapshot hook
  08:22 local). Agent `save minhtriethihi` → `health: ok` → `use minhtriethihi --confirm`
  ⇒ **CLI active = `minhtriethihi`** (OAuth), `.mcp.json`/`.vscode/mcp.json`/`.codex` đã sync.
  Chủ dự án còn phải `/mcp` reconnect. Quota lúc 01:58Z: `minhtriethihi` 17,77 h (12,2 h đang
  reserve cho 2 session), `khanhmay0304` 2,78, `minhtran0601` 1,33, `odixe0502` 0,66,
  `minhtrit06` 0,42 — tới refresh 2026-09-12T00:00Z.
  Lỗi **#42** (skill helper): `use` so `"username: X"` với dòng `- username: X` của CLI 2.2.4
  ⇒ báo "revoked" giả và exit 1 dù đã đổi xong. Sửa ở cả hai cây + test (20/20).
* ⚠ **Đính chính §14.5:** cả 4 bearer cũ (`khanhmay0304`, `minhtrit06`, `minhtran0601`,
  `odixe0502`) đều `IntrospectToken ⇒ active=True`. "Chết" là kết luận từ
  `probe_kaggle_mcp.py`, thứ `kaggle-credentials.md` đã ghi rõ **không** được dùng để chấp
  nhận/loại token. Cancel session qua MCP native (sau `/mcp` với bearer đúng) do đó **có thể**
  làm được — chưa thử lại; hiện vẫn coi cancel là việc của chủ dự án cho đến khi đo.
* Quota lúc 01:27Z: `minhtriethihi` 30,00 h. Các tài khoản khác gần cạn tới refresh 00:00Z
  (không đọc lại; không cần cho phiên này).

### 15.2 Phiên 1 (v3) — kết quả cuối, đã kéo về và kiểm

| run | round | phiên | round steady | dừng | `verify_run` (full, `--y-true`) |
|---|---:|---:|---:|---|---|
| `fdids_20c_v2` | **42**/50 → **50/50 ✅ phiên 2** | 6,15 h + 1,20 h | 520–524 s / 509–510 s | budget 6,40 h sau r42 → đủ 50 | **pass 1..50 (merged)** |
| `fdids_50c_v2` | **50/50 ✅** | 7,44 h | 528–538 s | đủ 50 | **pass 1..50** |
| `fdids_100c_v2` | **25**/50 → **50/50 ✅ phiên 2** | 5,86 h + 4,84 h | 738–981 s / 671–753 s | budget 6,20 h sau r25 → đủ 50 | **pass 1..50 (merged)** |

Output ở `papers/fd-ids-2025/runs/pulls/fdids_<K>c_v2.s1/` (`.gitignore` đã loại `papers/**/runs/`):
`runs/<run>/` đủ 8 thư mục + `history.csv`, log kernel, `proj/`, `wandb/` (không chứa key —
đã grep). Executed notebook ở `notebook/<K>c/fdids_<K>c_v2.executed.s1.ipynb` (0600, **có key**).

**Đường cong đầy đủ — đính chính cách gọi "thoái lui" ở §14.6.** Với đủ 42/50/25 round:

| run | `f1_macro` đỉnh | trung bình 10 round cuối | lệch | `accuracy` đỉnh → 10 cuối | `f1_weighted` đỉnh → 10 cuối |
|---|---:|---:|---:|---|---|
| 20c | 0,7822 @r7 | **0,7673** (0,7658–0,7685) | −1,9 % | 0,8161 @r4 → 0,7367 | 0,8083 → 0,7361 |
| 50c | 0,7205 @r6 | **0,6822** (0,6697–0,6900) | −5,3 % | 0,7642 @r3 → 0,6724 | 0,7545 → 0,6446 |
| 100c | 0,6870 @r9 | **0,6687** (0,6641–0,6746) | −2,7 % | 0,7192 @r3 → 0,6672 | 0,7133 → 0,6473 |

Cách nói đúng: **`f1_macro` bão hoà sớm (round 6–9) rồi giữ một plateau nhiễu thấp hơn đỉnh
2–5 %**; cái **thật sự xói mòn** là `accuracy` và `f1_weighted` (−8 đến −11 điểm), tức lớp
đa số; `recall_macro` **tăng** nhẹ và `precision_macro` giảm. Loss train (`ce_client_mean`)
vẫn giảm đơn điệu. 50c hội tụ hẳn quanh round 20 (`ce` dao động 0,245–0,252, `f1` phẳng
0,67–0,69 suốt 30 round). Giả thuyết BN-under-FedAvg ở §14.6 **vẫn chưa kiểm** — không chép
vào report như kết luận. Report phải neo vào đường cong + round đỉnh, không trích round 50.

### 15.3 Continuation qua checkpoint dataset — đã làm, số đo ở `TEST_LOG.md` §3.7

Thủ tục chuẩn nay ở **§10.3**. Điểm mấu chốt: cây run phải nằm **dưới một cấp `runs/`** trong
thư mục dataset (Kaggle bỏ một cấp khi giải nén); `-t` để `history.csv` không bị chuyển đổi;
đối chiếu tên + kích thước từng file (297/297, 178/178 khớp); **probe CPU** chạy đúng
`resolve_resume` production ⇒ `RESUME PROBE OK`, 42 và 25; fingerprint CFG notebook
(`5f74ba70f2c0af90`, `43fb1276b4aa83b1`) **khớp** manifest; W&B `resume="allow"` trên run
`finished` đã probe (giữ step cũ, nhận step mới, config đổi không lỗi).

Code đổi trong phiên: `gen_notebook.py --dataset-source`; validator nới `dataset_sources[:2]`
+ quy tắc attach ↔ `require_resume` (2 mutation mới, **10/10**); `tests/test_validator.py` sửa
lỗi **#41** (hardcode tên notebook — suite cũ đã gãy từ lúc relaunch `_v2` mà không ai chạy
lại); `scripts/gen_resume_probe.py` (mới, riêng dự án); `kaggle_as.py` (mới, trong skill vì dùng chung). Probe notebook ở
`notebook/resume_probe_{20,100}c/` — validator không quét, regenerate được.
Skill (cả hai cây): `kaggle-credentials.md` (token `KGAT_` = API token CLI, `IntrospectToken`
là cách kiểm tin được, `kaggle_as.py`) và `multi-account.md` (layout mount
`datasets/<owner>/`, đối chiếu mọi trang `datasets files`, probe CPU phải assert đúng round).
Viết **chung**, không nêu yêu cầu riêng của bài này — chủ dự án nhắc 2026-09-11.

### 15.4 Đang chạy — push 2026-09-11 01:46:10Z, cả hai `RUNNING` trong 40 s, W&B `running`

| kernel (v1) | `run_name` | tiếp từ | `max_hours` | kết quả |
|---|---|---:|---:|---|
| `minhtriethihi/fd-ids-veremi-20-clients` | `fdids_20c_v2` | 43 | 3,0 | **COMPLETE 02:59Z**, 50/50, 1,20 h, 509–510 s/round; đã pull, verify, **merge** (§15.2b) |
| `minhtriethihi/fd-ids-veremi-100-clients` | `fdids_100c_v2` | 26 | 11,0 | **COMPLETE ≈06:37Z**, 50/50, 4,84 h, 671–753 s/round; đã pull, verify, **merge** (§15.7) |

Hai session đồng thời trên một tài khoản (trần 2). Tiêu ≈ 8,5 h / 30 h. Log live 20c (qua
`kaggle kernels logs -f`, **có** stream được khi kernel đang chạy — khác với giả định cũ ở
§14.5 rằng stdout của kernel đang chạy không đọc được): `42 verified, imported 1..42` →
`resume from round 42` → prepack 132 s → `compile OK` cả hai rank → `workers ready on
compiled (246 s)` → **`[driver] resumed at round 43`**. `--max-hours` chỉ là
trần dừng sạch; run kết thúc khi đủ round 50. Dataset `minhtriethihi/fdids-{20c,100c}-v2-ckpt`
(private) là input thứ 3 của mỗi kernel. Watcher `scratchpad/watch3.py` (seed step 42/25,
poll 60 s, Kaggle status + W&B state/`backend`/hàng mới, heartbeat 30 ph) đang chạy.

⚠ `summary["backend"]` = `compiled` đọc được **ngay** khi W&B mở lại run — đó là giá trị
**của phiên 1** còn lưu trong summary, không phải bằng chứng phiên 2. Phiên 2 dùng backend
nào chỉ đọc được từ **thời gian round** (≈521 s ⇒ compiled; ≈1074 s ⇒ eager) hoặc từ
executed notebook sau khi xong.

✅ **W&B `state=crashed` (02:01Z) cho cả hai run là GIẢ — đã đóng.** Hàng step 43 của 20c vào
history lúc 02:08:56Z (chỉ ~8 ph sau khi kernel log), state tự về `running` (20c 02:08:55Z,
100c 02:14:27Z). Cơ chế: `heartbeat_at` đứng đúng lúc `workers ready` (spawn 2 worker), trong
khi **system stream vẫn chảy** mỗi 15 s. **Skill đã ghi đúng bẫy này từ 2026-09-09**
(`wandb.md` § "W&B run state is not evidence…") — tôi chẩn đoán lại từ đầu thay vì đọc nó
trước. Bài học: **đọc `wandb.md` mục đó trước khi diễn giải bất kỳ `state` nào của run
resume.** Đã bổ sung chi tiết hôm nay vào mục đó (cả hai cây). Giám sát 0 trễ:
`kaggle kernels logs -f <ref>` (watcher `scratchpad/watch_log.py`, song song với `watch3.py`).

### 15.2b 20c đã đủ 50 round — số cuối và bằng chứng continuation

`runs/fdids_20c_v2/` = merge của pull `.s1` (minhtrit06, r1–42) và `.s2` (minhtriethihi, r43–50)
bằng `merge_sessions.py`: **weights và preds round 1..42 giống hệt từng byte** giữa hai pull,
history khớp 42/42 hàng, 50 round liên tục, `verify_run --require-rounds 50` **pass mode full**.
`logs/sessions.json` ghi ai sản xuất round nào. Hai pull đã xoá sau khi bản merge pass.

Đường cong đủ 50: đỉnh `f1_macro` **0,7822 @r7**, **r50 = 0,7655**, `accuracy` 0,8161 → 0,7327,
`f1_weighted` → 0,7297. **Round 44 tụt xuống 0,7294** (−3,75 điểm, gấp 2,6× cú tụt lớn nhất
của phiên 1) rồi hồi 0,7499 → 0,7606 → … → 0,7655. Đã soi `logs/round_044.json`: không client
nào bất thường (ce/gnorm/skip/nonfinite như r43, r45) ⇒ dao động một round của phép gộp, không
phải lỗi continuation. Số liệu và bảng 10 metric r50 ở `TEST_LOG.md` §3.8.

Skill `merge_sessions.py` đã sửa **chung** (cả hai cây) vì nó từ chối sai merge này: history ở
gốc run, `manifest.json` là provenance từng phiên (nhóm `PER_SESSION` cùng `config.json`),
`logs/round_*` là artifact từng round. Chi tiết `TEST_LOG.md` §3.8.

### 15.5 Lỗ hổng còn mở — cổng GPU không kiểm `last == expected`

`require_resume` chỉ đòi `last is not None`. Dataset hỏng **một** round ở giữa ⇒ import dừng
sớm, cổng vẫn qua, phiên **train lại** các round thiếu dưới cùng `run_name`; W&B vứt step không
đơn điệu **im lặng**; chỉ `merge_sessions.py` bắt được (bytes khác) — sau khi đã tốn GPU.
Phiên này chắn bằng probe CPU (§10.3 bước 3). Sửa tận gốc = thêm `resume_from` vào CFG
(ngoài fingerprint) và assert ở cổng; **chưa làm** vì đang giữa chuỗi nhiều phiên — đổi code
production giữa chừng là điều `pull-outputs.md` khuyên tránh. Làm sau khi 20c/100c đủ 50.

### 15.6 Việc còn lại

1. ~~Khi 100c `COMPLETE`: pull, verify, merge, xoá pull~~ **ĐÃ XONG** (§15.7, `TEST_LOG.md` §3.9).
2. ~~Kernel phiên 2 dừng sớm~~ không xảy ra: 100c chạy trọn 25 round còn lại trong một phiên.
3. **Report** — **HOÀN TẤT** (§15.7). Lịch sử: bản interim `report.md` (gốc repo, theo yêu cầu chủ dự án 04:00Z) từ
   `scripts/make_report.py`: đọc `runs/fdids_{20c,50c}_v2`, kiểm CSV↔JSON↔confusion↔per-class từng
   round trước khi ghi; 50 round × 10 metric (`METRIC_KEYS` đúng thứ tự) mỗi cấu hình, bảng vận hành,
   per-class r50, 7 hình trong `figures/`, bài báo ở mục riêng, caveat DATASET §6 trong phần kết
   quả. Cổng độc lập đã chạy: 100 hàng, 0 lệch với JSON ở 4 chữ số, mọi ảnh tồn tại. **Phát hiện
   từ bảng per-class:** thoái lui nằm ở **recall `benign`** (20c 0,577 → 0,222; 50c 0,355 → 0,082
   từ round đỉnh tới r50) = báo động giả tăng theo round; `trafficCongestionSybil` ổn định 0,97;
   lớp khó: `timeDelayAttack`, `positionMirroring`. Sau khi merge 100c: chạy lại generator, kiểm
   lại cổng, bỏ nhãn INTERIM tự động. Mỗi kịch bản là **một** run 1..50 duy nhất (đã merge), phiên
   chỉ xuất hiện ở dòng provenance — theo yêu cầu chủ dự án. **Sau merge 100c:** generator chạy lại,
   cổng độc lập 3 × 50 hàng / 0 lệch / 10 ảnh, nhãn INTERIM tự rơi; prose §0/§8 nay tính từ artifact
   cho mọi cấu hình (§15.7 ghi một câu sai đã đính chính: KD loss **tăng dần** sau r7–8, chỉ CE đi ngang).
4. ~~`add-account` → `use`~~ đã xong (§15.1); còn `/mcp` reconnect của chủ dự án nếu chưa làm.
5. Sửa lỗ hổng §15.5 sau khi chuỗi xong; cân nhắc đo teacher (§14.7 mục 3); thí nghiệm BN
   trên checkpoint đã lưu nếu muốn nói gì về nguyên nhân plateau.
6. Đã xoá: staging `runs/ckpt_ds/`, toàn bộ `runs/pulls/` (nội dung đã merge + verify). Còn
   giữ: probe notebook (nhỏ), hai checkpoint dataset trên Kaggle `minhtriethihi/fdids-{20,100}c-v2-ckpt`
   (private; có thể xoá khi không cần chạy lại continuation), executed notebook `.s1`/`.s2` (0600, có key).

### 15.7 Phiên 3 (2026-09-11 ~09:45–10:05Z) — 100c xong, merge, report hoàn tất

* Kernel `minhtriethihi/fd-ids-veremi-100-clients` `COMPLETE` ≈06:37Z: `imported 1..25` →
  `resumed at round 26`, `workers ready on compiled` 250 s, 25 round **671–753 s**, phiên 4,84 h,
  W&B synced. `f1_macro` r26–50 dao động 0,647–0,670, r50 **0,6606**; tụt lớn nhất −1,75 điểm @r44.
* Pull `.s2` (609 MB, 378 file) verify **pass 1..50 full**; `merge_sessions.py` s1+s2: overlap 25/25
  hàng khớp, new rounds [25, 25], `logs/sessions.json` (1–25 `khanhmay0304`, 26–50 `minhtriethihi`);
  `rmdir checkpoints client_log protos`; bản merge verify **pass 1..50 full**, cây file giống 20c.
  Executed notebook `notebook/100c/fdids_100c_v2.executed.s2.ipynb` (0600, **có W&B key** — không
  commit/chia sẻ; gitignore đã loại). Đã xoá `runs/pulls/`.
* `scripts/make_report.py`: điền giờ kết thúc 100c; **bỏ prose hardcode hai kịch bản** — caption §0
  (plateau từng cấu hình, đơn điệu theo số client), §8 (dải đỉnh/xói mòn tính từ dữ liệu, continuation
  liệt kê mọi cấu hình nhiều phiên), caption hình hội tụ + tiêu đề panel phải nêu **CE đi ngang, KD
  tăng dần** (số đo: `TEST_LOG.md` §3.9). `report.md` 649 dòng, HOÀN TẤT, cổng độc lập pass.
* 10 metric r50 ba cấu hình (accuracy / F1 macro / F1 weighted): 20c 0,7327 / 0,7655 / 0,7297;
  50c 0,6714 / 0,6815 / 0,6423; 100c 0,6552 / 0,6606 / 0,6223. Đỉnh F1 macro hậu kiểm 0,7822 @r7 /
  0,7205 @r6 / 0,6870 @r9. Chi tiết và per-class: `report.md` §4–§6, §8.
* Quota `minhtriethihi` đo lúc 10:05Z: **GPU còn 23,94 h / 30 h** (chuỗi 20c + 100c phiên 2 tiêu
  ≈6,1 h), TPU 20 h; refresh 2026-09-12T00:00Z.
