# CONTEXT — pFedES trên VeReMi NextGen / DAGSNet

Bối cảnh dùng lại qua nhiều phiên. Đọc file này **trước** khi làm gì khác.
Cập nhật lần cuối: **2026-09-16 ~07:00Z** (giờ máy local = UTC+7).

**Repo đã chuyển chỗ 14-09:** `~/nckh/pfedes` → **`~/nckh/veremi/pfedes`**; dataset gốc
`~/nckh/veremi/dataset`; dự án anh em `~/nckh/veremi/{afpha,edl_cmso,fd_ids,tinyproto}`. Mọi đường
dẫn trong skill/test/knowledge/`.codex` đã sửa (§14.5); memory Claude đã chép sang project mới.

**Trạng thái: CẢ BA KỊCH BẢN XONG 50/50 round, report chính thức đã viết** — production `_cos`
(C = 100 %, LR cosine 1e-3 → 1e-5, T = 50), §14–16.
- Thư mục run chuẩn, **mỗi file một bản duy nhất**: `runs/merged/pfedes_{20c,50c,100c}_cos/` (1,9 / 4,5 / 8,9 GB),
  mỗi thư mục là MỘT run 50 round liền mạch (ghép 2 / 3 / 4 phiên, byte-identical trên mọi round chồng,
  `verify_run --require-rounds 50` pass); provenance từng phiên trong `logs/sessions.json` + `logs/sessions/<n>/`
  (log kernel, `manifest.json`, `executed.ipynb`, `wandb/`). **Mọi pull gốc đã xoá 16-09** (25 GiB) — §16.3.
- f1_macro r50: **20c 0,4856** (acc 0,5276) · **50c 0,3652** (acc 0,4335) · **100c 0,2900** (acc 0,3692).
  Đường cong khớp §12.4: phẳng từ r15 (20c, 50c) / r25 (100c); giảm so đỉnh −3,1 / −6,2 / −10,5 %.
- Số liệu: `papers/pfedes-yi-2025/report_data/` (sinh lại 16-09 từ ba thư mục merged); báo cáo:
  `papers/pfedes-yi-2025/report.md` **bản chính thức 16-09** (phụ lục A–C = 50 × 3 hàng, 0 ô lệch so `summary.json`).
- Kaggle: 100c s4 (`minhtrit06/pfedes-veremi-100-clients-cos-s4`) COMPLETE 15-09 ~20:15Z, 11 round, steady 2907 s
  (chậm hơn s3 12 %); quota còn (16-09 03:30Z): minhtran0601 17,59 · minhtrit06 9,97 · khanhmay0304 3,57 ·
  odixe0502 1,50 · minhtriethihi 0,97 (ba tài khoản sau bị dự án khác dùng thêm từ 15-09). Refresh 19-09 00:00Z.

Lịch sử: `_full` (LR hằng, dừng tay 12-09) §11–12. Bản chạy C < 100 % (11-09) đã bị **xoá hẳn 16-09** theo yêu
cầu chủ dự án và không còn được nhắc trong report/rebuild.

---

## 0. Mục lục — ở đâu có gì

| đường dẫn | nội dung | ai sửa |
|---|---|---|
| [`00121-YiL.md`](00121-YiL.md) | bài báo pFedES (Yi et al.) — **không có phụ lục** (μ, E_fe, Algorithm 1 thiếu) | chỉ đọc |
| [`knowledge/`](knowledge/) | **sự thật không đổi** dùng chung nhiều phương pháp | sửa khi *đo lại* |
| ├ [`ARCHITECTURE.md`](knowledge/ARCHITECTURE.md) | DAGSNet 395.024 tham số, 66 cột, 16 lớp, mã nguồn đã kiểm | |
| ├ [`DATASET.md`](knowledge/DATASET.md) | 43.045.415 train / 10.761.343 test, 3 phân mảnh α=0,5, ngân sách bước | |
| ├ [`LOCAL_ENV.md`](knowledge/LOCAL_ENV.md) | máy local (8 GB WSL, RTX 3050 4 GB, sm_86) + **ranh giới** local↔Kaggle | |
| ├ [`KAGGLE_DATASETS.md`](knowledge/KAGGLE_DATASETS.md) | 4 dataset Kaggle, đều public | |
| └ [`runtime.json`](knowledge/runtime.json), `meta.json`, `scaler.json` | digest image đã kiểm, thứ tự cột, tên lớp, mean/std | |
| [`papers/pfedes-yi-2025/`](papers/pfedes-yi-2025/) | **phương pháp này** | |
| ├ [`paper.md`](papers/pfedes-yi-2025/paper.md) | trích xuất phương pháp + những chỗ bài báo để trống | |
| ├ [`rebuild.md`](papers/pfedes-yi-2025/rebuild.md) | **quyết định đã chốt, 10 deviation, hợp đồng artifact, ước lượng chi phí, bảng test** | |
| ├ [`proj/`](papers/pfedes-yi-2025/proj/) | **8 module** — nguồn duy nhất của code notebook | sửa ở đây |
| │ ├ `model.py` | DAGSNet; `build_model` (F_k, 16 lớp) và `build_proxy` (G, ra 66) | |
| │ ├ `pfedes.py` | Eq. (4)–(11): `client_update` (bước ①/②), `aggregate`, `select_clients`, flat layout | |
| │ ├ `evaluate.py` | fold BN (exact), template eval compiled, `eval_model` | |
| │ ├ `driver.py` | 2 worker/2 GPU, trọng số per-client thường trú, gate compile train+eval, eval cache, commit atomic | |
| │ ├ `ckpt.py` | weights (G + N F_k) / resume / marker / fingerprint / import qua staging | |
| │ ├ `data.py`, `metrics.py` | parquet → fp16 thường trú; 10 metric từ confusion (chép từ `fd_ids`) | |
| │ └ `verify.py` | dựng lại mọi con số từ artifact, per-client, kiểm cache carry-forward | |
| ├ [`notebook/`](papers/pfedes-yi-2025/notebook/) | `{20,50,100}c/` production phiên 1, run-tag `_cos` = C 100 % + LR cosine (owner 20c `minhtriethihi`, 50c `khanhmay0304`, 100c `odixe0502`; W&B key đã nhúng, mode 600; **đã push 13-09**); `100c_s2/` phiên 2 (đã push 22:52Z 13-09); `20c_s2/`, `50c_s2/` phiên 2 (đã push 00:51Z 14-09); `50c_s3/` (push 17:03Z 14-09), `100c_s3/` (owner **`minhtrit06`**, `--dataset-source`, push 17:47Z 14-09), `100c_s4/` (owner `minhtrit06`, `--kernel-source …-s3`, push 10:44Z 15-09); `100c_ckpt_probe/` (CPU probe của cổng resume, sinh bởi `scripts/gen_ckpt_probe.py`; **đã xoá 16-09**, sinh lại được); `20c_probe/` (owner `khanhmay0304`, sinh lại 13-09, chưa push lại). Thư mục phiên tiếp (`{K}c_sN/`) **sinh khi cần** bằng `--session N --require-resume --kernel-source …`. `.ipynb` **sinh tự động** | ❌ không sửa tay |
| ├ `runs/pulls/probe20/` | output probe 20c kéo về (calibration.json, history, weights 2 round, log kernel) — pull duy nhất còn giữ | |
| ├ `runs/merged/pfedes_{20c,50c,100c}_cos/` | **20c ghép 2 phiên, 50c 3, 100c 4 = một run duy nhất 50 round mỗi K** (`merge_sessions.py`; `logs/sessions.json` + `logs/sessions/{n}/` giữ log kernel, manifest, `executed.ipynb`, `wandb/` từng phiên); verify pass. **Thư mục chuẩn cho report và bản duy nhất của mọi artifact** — pull gốc đã xoá 16-09 | |
| ├ [`report.md`](papers/pfedes-yi-2025/report.md) | **báo cáo dựng lại** (tiếng Việt, **bản chính thức 16-09**, cả ba K đủ 50 round): tóm tắt, deviation, dữ liệu+caveat, cấu hình/hạ tầng/phiên, đường cong, 10 metric round cuối, phân tán client, per-class + confusion, chi phí, kết quả bài báo để riêng, bằng chứng/giới hạn, tái lập; **phụ lục A–C = mọi round × 10 metric** nối từ `rounds_Kc.md` | sửa tay thân bài; phụ lục nối lại bằng `tail -n +5 rounds_Kc.md` |
| └ [`report_data/`](papers/pfedes-yi-2025/report_data/) | **số liệu cho report**: `README.md` (bảng tổng hợp tiếng Việt), `rounds_Kc.md`/`history_Kc.csv` (đủ 10 metric mọi round), per-client, per-class, confusion gộp, `summary.json`, 9 hình PNG | sinh lại bằng `scripts/report_data.py` |
| [`scripts/`](scripts/) | | |
| ├ `gen_notebook.py` | sinh notebook từ `proj/`: `--owner --clients --probe --max-hours --run-tag --require-resume --kernel-source --dataset-source --eval-batch --session N` | |
| ├ `verify_run.py` | verifier offline cho run đã kéo về: `<run dir> --require-rounds 50` | |
| ├ `report_data.py` | sinh `report_data/` từ artifact đã verify: `--out <dir> 20c=<run> 50c=<run> 100c=<run>` (bảng đủ 10 metric mọi round, per-client/per-class/confusion, README, hình — palette skill dataviz) | |
| ├ `gen_ckpt_probe.py` | notebook CPU (0 GPU quota) kiểm checkpoint dataset qua đúng `proj/ckpt.py`: `--owner --dataset --run-name --expect-round --out` | |
| ├ `watch_prod.py` | poller W&B (+ `kaggle kernels status` với `--kernel owner/slug`): `python scripts/watch_prod.py pfedes_20c_cos pfedes_50c_cos pfedes_100c_cos --every 300` | |
| ├ `validate_notebooks.py` | cổng tĩnh: metadata, CFG qua AST, module ↔ nguồn từng byte, luồng cell, tên biến | |
| ├ `run_local_checked.py` | **watchdog RAM bắt buộc** cho mọi test local | |
| └ `kaggle_mcp_headers.py`, `sync_kaggle_mcp.py`, `probe_kaggle_mcp.py` | MCP (chép từ `fd_ids`) | |
| [`tests/`](tests/) | 7 file — xem `rebuild.md` §5; chạy: `python scripts/run_local_checked.py python tests/<file>` | |
| `.agents/skills/` + `.claude/skills/` | skill `kaggle-training-notebook` — `.agents` là bản gốc, `.claude` mirror; **sửa thì sửa cả hai**. `scripts/pull_kernel_output.py` (16-09): kéo output có resume/retry, thay `kaggle kernels output` | |
| `.mcp.json`, `.vscode/mcp.json`, `.codex/config.toml` | MCP Kaggle; bearer của tài khoản CLI đang active | `kaggle_account.py use` |

Dữ liệu local: train `~/nckh/veremi/dataset/fl_client/alpha05/{20,50,100}_client/train/client_id=NNN/`,
test `~/nckh/veremi/dataset/centralized/test/`. Kaggle: `odixe0502/veremi-fl-{20,50,100}client` +
`odixe0502/veremi-nextgen2026-centralized` (public).

Dự án anh em cùng dữ liệu/kiến trúc (đã khai thác xong, **không cần mở lại**; bài học đã
ghi vào skill): `~/nckh/veremi/fd_ids` (FD-IDS, global model), `~/nckh/veremi/tinyproto` (per-client model).

**Vòng đời chuẩn:** sửa `proj/*.py` → chạy test liên quan → `python scripts/gen_notebook.py
--owner <acct> [...]` → `python scripts/validate_notebooks.py` → nhúng W&B key
(`.agents/skills/kaggle-training-notebook/scripts/embed_wandb_key.py NB --metadata META`) →
**validate lại** → push (`kaggle_as.py <acct> -- kaggle kernels push -p <dir>`).

---

## 1. Quyết định khoa học đã chốt (chủ dự án, 2026-09-11)

| | |
|---|---|
| F_k | DAGSNet 395.024 tham số, mỗi client một bộ trọng số, cùng init seed 42 |
| G (proxy, thứ duy nhất server giữ) | **DAGSNet với head ra 66** (407.874 tham số) — chủ dự án chọn thay 2-conv CNN của bài báo |
| C tham gia | **100 % cho cả ba kịch bản** (K = 20 / 50 / 100) — chủ dự án chốt 12-09, thay cho 100 %/20 %/10 % của Table 1–2. Lý do: mean/std lấy trên mọi client chỉ có nghĩa khi mọi F_k đều đã được huấn luyện (§8) |
| Optimizer | **AdamW, wd 1e-4**, tạo mới mỗi client mỗi round (moment bị bỏ) — bài báo dùng SGD 0,01 |
| Learning rate | **cosine theo round 1e-3 → 1e-5, T = 50**, hằng trong round, chung η_ω = η_θ (`proj/pfedes.py::lr_at`, công thức `afpha`) — chủ dự án chốt **13-09** làm lịch **thống nhất mọi dự án anh em**; thay LR hằng 1e-3 của 11-09. `lr_schedule`, `lr_min`, `rounds` nay nằm trong fingerprint |
| μ | **0,5** (bài báo không cho giá trị) |
| E / E_fe | **1 / 1**; 50 round; batch 512/512/256 |
| Eval | **mỗi round, mọi N client, đủ 10.761.343 dòng test**, 10 metric/client + mean/std/min/max |
| Checkpoint | **chỉ trọng số** (state_dict G + N state_dict F_k), `weights_only=True`; không lưu module, không optimizer |

Deviation đầy đủ (10 mục, phải công bố kèm mọi con số): `rebuild.md` §2. Ba mục quan
trọng nhất: (1) bước ① chạy **một** forward trên batch ghép `[x̂; x]` (BN thấy cả hai nửa)
vì CUDA-graph không cho gọi module compiled hai lần trước backward; (2) Eq. (11) chuẩn hoá
theo tập **được chọn**; (3) **eval cache**: client không được chọn giữ nguyên F_k ⇒ ma trận
nhầm lẫn giữ nguyên, không chạy lại; mỗi 10 round và round cuối eval lại toàn bộ và so
(`cache_mismatch` phải = 0).

## 2. Số đo đã có

**Local (sm_86, không phải bằng chứng cho sm_75):** compile train certify (`max|dlogit|`
9,8e-4 / 7,9e-4), compiled ≈ 9,4 ms/phase-step ở B=256 vs eager ~28 ms; eval compiled
343k rows/s vs eager-folded 255k (B=4096), không re-record khi đổi trọng số client; 1 worker ≡
2 worker bit-identical; crash-replay bit-identical.

**T4 (kế thừa từ `fd_ids`/`tinyproto`, đo thật):** compiled+AMP 6,2 ms/step B=512, 5,7 ms
B=256; eval ~300k rows/s/GPU (eager-folded hoặc compiled-unfolded); prepack 43 M dòng ~2–3
phút; startup 12–17 phút; 100c batch 256 **nghẽn 4 vCPU** (util 58 %) → round trôi +20 %.

**T4 đo thật bằng probe 20c (`khanhmay0304`, 13:17Z–14:16Z, 0,99 h quota;
`runs/pulls/probe20/`):** train **compiled** 12,06 ms/phase-step (eager 42,7 → ×3,54), compile+gate
264 s; eval compiled-folded @16384 **414k rows/s**/GPU (eager-folded 296k; @8192 compiled chỉ
76k vì dính re-record → **giữ 16384**); startup+prepack+compile **15,7 phút** (prepack 128 s);
round 1 = 1351 s, round 2 = **1256 s** = train 1000 s + eval 256 s (20 client); VRAM 7,3 GiB/GPU;
skip AMP 12+27 / 84 083+84 083 bước (warm-up GradScaler); `cache_mismatch=0`; f1_macro
0,487 → 0,495 (std 0,10, min 0,28, max 0,68), accuracy 0,566 → 0,559; `verify_run` local pass.
Gate compile trên sm_75: max|dlogit| 9,8e-4 / 7,9e-4, 0 flip decisive.

**Ngân sách cho C = 100 % (chốt 12-09)** — ngoại suy từ chính các số đo trên, không phải
ước lượng suông. Ở C = 100 % mỗi round train **đúng một epoch trên toàn bộ 43 045 415 dòng**
(tổng các shard client), nên train/round gần như **không phụ thuộc N**, chỉ phụ thuộc batch;
eval/round là eval-all vì cache carry-forward **không bao giờ được dùng**:

| | train/round | eval/round | round | 50 round | phiên (11/11/10,5 h) |
|---|---:|---:|---:|---:|---:|
| 20c (B=512) | 1000 s *(đo thật, probe r2)* | 265 s *(đo thật)* | ~21 phút | **17,7 h** | 2 |
| 50c (B=512) | ~1060 s | 670 s *(đo thật, round eval-all)* | ~29 phút | **24,6 h** | 3 |
| 100c (B=256) | ~1600 s | 1442 s *(đo thật, round eval-all)* | ~51 phút | **42,8 h** | 5 |

Cách suy: 11,9 ms/phase-step ở B=512 và 8,74 ms ở B=256 (chia ngược từ `train_sec`/`steps_w`
của probe và các run trước), × 2 pha × số bước toàn tập ÷ 2 GPU; 100c cộng thêm ~10 % nghẽn 4 vCPU.
Tổng ≈ **85 h round-time / ~100 h session**. Dung lượng weights: 1,7 / 4,0 / 8,0 GB (không đổi).
Tail sau khi driver dừng (verify + W&B + nbconvert) đo ở 100c (run trước, 11-09): **18 phút** — vì vậy
`--max-hours` 11,0 / 11,0 / **10,5** (100c nhỏ hơn vì commit 8 GB).

## 3. Bẫy đã gặp trong phiên này (đừng lặp lại)

0. **Hai worker/2 GPU không cho logit bit-identical** (cuDNN benchmark chọn thuật toán riêng
   từng process): eval tất định *trong* một worker, lệch vài dòng fp16 *giữa* hai worker.
   Split eval theo vị trí trong danh sách ⇒ cache check so GPU 0 với GPU 1 ⇒ `cache_mismatch`
   28/90 ở 100c round 10 (đúng bằng số client đổi GPU). Sửa: client c luôn eval trên worker
   `c % W` (`driver.py::_rounds`); verifier chấp nhận ≤ `CACHE_TOL_ROWS = 100` dòng.

1. **Illegal memory access** khi 4 CUDA graph (G eval no-grad, F train, G train, F eval
   với grad đầu vào) xen kẽ trong một bước: sửa bằng
   `torch.compiler.cudagraph_mark_step_begin()` đầu mỗi step (`pfedes.py::_epoch`). Chỉ
   hiện khi **không** có `CUDA_LAUNCH_BLOCKING=1`.
2. **Dynamo recompile_limit = 8 theo code object**: F, G và template eval dùng chung
   `DAGSNet.forward` ⇒ 9 biến thể ⇒ biến thể thứ 9 (eval) **âm thầm chạy eager**. Sửa: đặt
   `recompile_limit`/`cache_size_limit` = 64 trong worker và mọi chỗ compile in-process.
3. Payload qua `mp.Queue` phải là **numpy** (tensor torch đi qua `/dev/shm`, container có
   thể cap 64 MB; bảng trọng số client 160 MB ở 100c).
4. 2 worker CPU + parent vượt watchdog 8 GB nếu import pyarrow ở module test (spawn
   re-import) — import lazy trong `head()`; test 2 worker dùng fixture 2k dòng, batch 128.
5. Inductor trong parent + worker CUDA cũng vượt watchdog ⇒ `test_notebook_sim.py` chạy
   CPU (mọi cell trừ calibration) và `--gpu` (tới hết calibration, eager).
6. **`rounds` nằm trong fingerprint từ 13-09** (lịch cosine trải theo T): phiên nối tiếp phải
   sinh với cùng `rounds=50`; test/smoke muốn "chạy ít round rồi tiếp" phải giữ nguyên T và
   mô phỏng ranh giới phiên bằng cách xoá round cuối (xem `test_smoke_real.py`).
7. Trần watchdog = min(3000, MemAvailable − 2048) MiB: khi VSCode server chiếm ~2,5 GB, trần
   còn ~2,6 GB và `test_two_workers`/`test_gpu_local`/driver 2-process không vừa. Thí nghiệm
   local dùng **một process** (không spawn worker) để tiết kiệm ~500 MB.

## 4. Tài khoản & quota (đọc lại bằng `kaggle_account.py quota` trước mỗi launch)

**Đọc 16-09 03:30Z (mọi run xong):** minhtran0601 17,59, minhtrit06 9,97 (s3 + s4 = 20,0 h), khanhmay0304 3,57,
odixe0502 1,50, minhtriethihi 0,97 — ba tài khoản sau giảm so 15-09 do dự án khác (`nilm-fl`) dùng. Refresh **2026-09-19T00:00Z**.
Đọc 15-09 11:24Z (s4 100c đang chạy): minhtrit06 18,82 (s3 tiêu 10,5 h; s4 sẽ tiêu ≈ 9 h ⇒ còn ≈ 10 h),
minhtran0601 28,19 (chưa dùng, dự phòng), odixe0502 11,08, minhtriethihi 9,63 (20c xong), khanhmay0304 3,57
(50c xong). Refresh **2026-09-19T00:00Z**.
Đọc lại 12-09 ~11:00Z: **cả 5 tài khoản đều 30,00 h GPU / 20,00 h TPU**
(`khanhmay0304`, `minhtran0601`, `minhtriethihi`, `minhtrit06`, `odixe0502`), refresh tiếp
**2026-09-19T00:00Z** ⇒ tổng **150 h** cho tuần này, đủ cho ngân sách ~100 h của §2 nhưng
100c (≈52 h, 5 phiên) **không vừa một tài khoản** — xem §7.4.
Kaggle giữ reserve khi session chạy (23,94 → 11,93 free khi 2 session bắt đầu) và trả lại khi xong. Chủ dự án 13-09: **dùng cả 5 tài khoản cho production** `_cos`
(20c minhtriethihi, 50c khanhmay0304, 100c odixe0502 → minhtrit06 → minhtran0601; §12.5);
trước đó production chỉ trên `minhtriethihi`, probe trên `khanhmay0304`/`minhtran0601`. Hook SessionStart có lúc báo
`minhtriethihi DEAD` thoáng qua; `kaggle_account.py health` ngay sau đó trả `ok` cho cả 5.

Chính sách 2026-09-10: agent tự đổi tài khoản (`kaggle_account.py use/ensure --confirm`),
phải **nói rõ** đã đổi; chủ dự án chỉ cần reconnect MCP (`/mcp`) khi thực sự cần. Push/kéo
dưới tài khoản khác **không cần đổi**: `kaggle_as.py <user> -- kaggle ...`. Tối đa **2 GPU
session đồng thời/tài khoản**, 12 h/session, `/kaggle/working` xoá khi session mới.

W&B: key **đã nhúng** vào notebook probe (uỷ quyền thường trực 2026-09-07; key vĩnh viễn
trong version history Kaggle; `.ipynb` không commit — `.gitignore` đã chặn). Project W&B:
`pfedes-veremi`, run id = `run_name`.

## 5. Theo dõi production — đọc gì

W&B project `21522798-uit/pfedes-veremi`, run id = `run_name`. Từ 13-09 là
`pfedes_20c_cos`, `pfedes_50c_cos`, `pfedes_100c_cos` (run-tag `_cos` = C 100 % + LR cosine);
`pfedes_*_full` (12-09, crashed sau 3/2/1 round) và `pfedes_50c`/`pfedes_100c` (run trước 11-09) là lịch sử,
**không được ghi đè** — đó chính là lý do phải đổi `run_name`: `resume="allow"` trên cùng id
sẽ nối round mới vào sau round cũ. Phiên 2, 3… của cùng kịch bản ghi tiếp cùng run id. Mỗi
round một hàng: `round, lr, f1_macro(_std/_min/_max), accuracy, train_sec, eval_sec, seconds,
evaluated, cache_mismatch, skipped_w/theta, steps_w, vram_train_gb`; `summary.backend`/
`backend_eval` phải `compiled`; `lr` phải đúng `lr_at` (r1 1e-3, r10 9,2e-4, r25 5,2e-4, r50 1e-5).
Kaggle stdout **không đọc được** khi đang chạy; chỉ `kernels status`.
Poller: `scripts/watch_prod.py` (trong repo từ 13-09). Local scan cùng project: group
`lrscan_local` (`lrscan_c1e-3` crashed = dừng tay, `lrscan_cos1e-3`).

**Ở C = 100 % `cache_mismatch` luôn = 0 theo cấu trúc** (không có client nào "không được
chọn" nên nhánh cache không bao giờ chạy) — chính là lỗi đã làm hai run trước (C < 100 %, 11-09) ERROR ở cell
cuối, nay biến mất. Hàng W&B đầu tiên xuất hiện sau ≈ startup 8 phút + 1 round: **~30 phút (20c)**,
**~37 phút (50c)**, **~1 h (100c)**; `evaluated` phải luôn bằng N và `selected` bằng N.

W&B history **không có khoá `round`** — round = `_step` của hàng (poller phải đọc `_step`). Dấu hiệu phải báo chủ dự án dừng (`cancel_notebook_session` qua
MCP hoặc nút Stop): backend `eager` (ngân sách ×3,5 — không đủ 12 h), `cache_mismatch > 0`,
skip tăng dần theo round (không phải hằng ~1–3/client), `f1_macro` giảm liên tục về 0 hoặc
NaN, `train_sec` tăng > 30 % so với round trước cùng loại, VRAM > 14 GiB, kernel ERROR.
Kết quả probe: skip hằng 12+27/round ở 20c là bình thường.

Khi COMPLETE: **không dùng `kaggle kernels output`** (treo vô hạn trên socket CDN chết, để lại file 0 byte rồi coi là
"đã cập nhật" — §16.1). Dùng `kaggle_as.py <acct> -- python $H/pull_kernel_output.py <acct>/<slug> -p runs/pulls/<name>
[--exclude 'weights/round_0(0\d|[1-3]\d)\.pt$']` (resume/retry/stall-cut; loại weights đã có từ phiên trước) +
`download_executed_notebook.py … --output runs/pulls/<name>/executed.ipynb`; file 0 byte hợp lệ chỉ là marker
`complete/*.done` và `proj/__init__.py`. Rồi `merge_sessions.py` (kiểm byte-identical), `verify_run.py <merged>
--require-rounds 50` (`CUDA_VISIBLE_DEVICES=""` + watchdog), chép `executed.ipynb` + `wandb/` của pull vào
`logs/sessions/<n>/`, xoá pull (mỗi file một bản — §16.3).

## 6. Cách chạy test local

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate nckh
CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/test_units.py
CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/test_smoke_real.py      # ~4 phút
CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/test_two_workers.py
CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/test_compile_gate.py
CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/test_ckpt_verify.py
python scripts/run_local_checked.py python tests/test_gpu_local.py                             # GPU local, Inductor
CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/test_notebook_sim.py   # thêm --gpu cho cell calibration
python scripts/validate_notebooks.py
```
Một cây test một lúc; watchdog dừng thì giảm bài test, **không nới ngưỡng**.

## 7. Việc tiếp theo — training và report đã xong

1. **Dọn Kaggle (tuỳ chọn, chưa làm):** dataset `minhtrit06/pfedes-100c-cos-ckpt-s2` (4,2 GB, private) và kernel
   `minhtrit06/pfedes-100c-cos-ckpt-probe` không còn cần; output các kernel `_cos` trên Kaggle vẫn là bản sao dự phòng
   của `runs/merged/` (kéo lại bằng `pull_kernel_output.py` nếu cần).
2. **Dự án anh em** (`afpha`, `edl_cmso`, `fd_ids`, `tinyproto`, nay ở `~/nckh/veremi/`): chỉ ghi note 13-09 về lịch LR
   thống nhất; **chưa sinh notebook** — chỉ làm khi chủ dự án yêu cầu.
3. **Test local còn nợ:** `test_two_workers`, `test_gpu_local` (watchdog, §3.7); `test_smoke_real.py` đã đổi sang đường
   dẫn dataset mới nhưng **chưa chạy lại** sau khi đổi.
4. **RAM local:** watchdog đòi MemAvailable ≥ ~3,6 GB; khi VSCode server + Pylance + Copilot chiếm ~3,7 GB thì
   `verify_run` bị từ chối (gặp 15-09, tự hết sau vài phút). `merge_sessions.py` không cần watchdog.

## 8. Caveat bắt buộc kèm mọi con số công bố

Kế thừa `knowledge/DATASET.md` §6, `ARCHITECTURE.md` §8: split theo thời gian mô phỏng;
41:1 mất cân bằng → đọc `f1_macro`; rò rỉ Sybil; scaler fit trên toàn bộ train (rò rỉ
thống kê toàn cục trong FL); fp16 lượng tử hoá đặc trưng; test không chia theo client
(điểm đo tổng quát hoá toàn cục của mô hình cá thể hoá, khác bài báo — test riêng từng
client); một seed; không đặt số cạnh số của bài báo (CIFAR/MNIST, accuracy trên test riêng);
mean lấy trên mọi N client — **ở C = 100 % caveat này không còn**: mọi client được huấn luyện mỗi
round, không còn F_k đứng ở init kéo mean xuống. Ngược lại phải công bố rằng **C = 100 % không phải
cấu hình của bài báo** (Table 1–2 dùng 100 %/20 %/10 %), nên số của 50c/100c không so trực tiếp
được với Table 1–2.

---

## 9. Đối chiếu notebook đã build với quyết định đã chốt — 2026-09-11

**Phạm vi theo yêu cầu mới nhất:** chỉ kiểm tra notebook hiện tại có thực hiện đúng các
quyết định chủ dự án đã ghi tại §1 và `rebuild.md` §1 hay không. Không đánh giá lại bài báo,
không đề xuất thay mô hình/dataset/siêu tham số. Những lựa chọn triển khai tại `rebuild.md`
§2 được đối chiếu riêng, không mặc nhiên coi tất cả là quyết định trực tiếp của chủ dự án.

**Kết luận:** ba notebook production **20c/50c/100c khớp các quyết định khoa học đã chốt**
về mô hình, dataset, optimizer, μ, epoch, round, batch, tỉ lệ tham gia, đối tượng đánh giá
và nội dung checkpoint. Có điểm chưa cưỡng chế đúng yêu cầu kiểm chứng eval cache (§9.3).
Đây là xác nhận về **cấu hình và mã đã build**, chưa phải xác nhận run Kaggle hoàn tất.

### 9.1 Cấu hình đọc trực tiếp từ notebook *(bảng đã cập nhật cho bản `_full` 12-09)*

Đã phân tích cell `CFG` bằng AST trong từng `.ipynb`, kiểm tra code nhúng với nguồn
`proj/`, rồi đọc luồng train/eval/checkpoint thực tế.

| notebook | N client | tham gia C → K | round | E / E_fe | batch train | kết luận |
|---|---:|---|---:|---|---:|---|
| `notebook/20c/pfedes_20c_full.ipynb` | 20 | 100% → 20 | 50 | 1 / 1 | 512 | **Đã ổn — khớp** |
| `notebook/50c/pfedes_50c_full.ipynb` | 50 | **100% → 50** *(đổi 12-09)* | 50 | 1 / 1 | 512 | **Đã ổn — khớp** |
| `notebook/100c/pfedes_100c_full.ipynb` | 100 | **100% → 100** *(đổi 12-09)* | 50 | 1 / 1 | 256 | **Đã ổn — khớp** |
| `notebook/20c_probe/pfedes_20c_probe.ipynb` | 20 | 100% → 20 | **2** | 1 / 1 | 512 | **Đúng vai trò probe**, không phải production 50 round |

Các đường dẫn notebook ở bảng tương đối với `papers/pfedes-yi-2025/`.

| quyết định đã ghi nhận | notebook/code thực tế | kết luận |
|---|---|---|
| F_k = DAGSNet, 395.024 tham số, trọng số riêng từng client, cùng init seed 42 | `build_model` đúng cấu hình `knowledge/ARCHITECTURE.md`; driver tạo F0 sau seed 42, sao chép init cho N client, giữ và cập nhật từng F_k | **Đã ổn — khớp** |
| G = DAGSNet với head ra 66, 407.874 tham số | `build_proxy` dùng backbone DAGSNet, `out_dim=n_features=66` | **Đã ổn — khớp** |
| thuật toán server chỉ tổng hợp G | `aggregate` nhận vector G của các client được chọn; F_k không bị FedAvg | **Đã ổn — khớp**; driver trung tâm giữ các F_k để mô phỏng và lưu checkpoint theo yêu cầu |
| VeReMi NextGen, 66 feature, 16 lớp, phân hoạch α=0,5 tương ứng 20/50/100c | metadata gắn dataset tương ứng; feature/class order lấy từ `knowledge/meta.json`; kiểm tổng 43.045.415 train và 10.761.343 test | **Đã ổn — khớp trong mã** |
| preprocessing theo `knowledge/` | train đã chuẩn hoá được đọc nguyên; test áp mean/std của scaler train đúng một lần, không fit test | **Đã ổn — khớp** |
| AdamW lr=1e-3, weight_decay=1e-4, reset mỗi client mỗi round, LR hằng | worker tạo mới cả optF/optG trong tác vụ train client; cùng lr/wd; không có scheduler thay LR | **Đã ổn — khớp** |
| μ=0,5 | CFG đặt 0,5; pha ① tính `μ·CE(F(G(x)),y)+(1−μ)·CE(F(x),y)` | **Đã ổn — khớp** |
| E=1, E_fe=1 | vòng epoch của hai pha đọc `local_epochs=1`, `proxy_epochs=1`; giữ batch cuối | **Đã ổn — khớp** |
| đánh giá mỗi round trên toàn bộ test, mọi N client; 10 metric/client và mean/std/min/max | eval dùng **F_k(x)**; mỗi confusion có đủ n_test; driver tính metric cho mọi N client, rồi tổng hợp không trọng số giữa client | **Đã ổn theo cơ chế cache đã ghi nhận**, xem §9.2–9.3 |
| checkpoint mỗi round chỉ lưu trọng số G + mọi F_k, không module/optimizer | `save_round_weights` ghi state_dict gồm BN buffers, cfg/metadata dựng lại; RNG ở file resume riêng; đọc bằng `weights_only=True` | **Đã ổn — khớp** |

Seed, init chung và các chi tiết triển khai nêu trên được đối chiếu với hồ sơ hiện có;
`rebuild.md` ghi init chung là lựa chọn triển khai của agent, không phải một yêu cầu mới.

### 9.2 Những chi tiết dễ hiểu nhầm nhưng không phải lệch quyết định

- **Eval cache:** ở 50c/100c, round thường chỉ chạy forward lại F_k của client vừa train;
  các client không đổi trọng số dùng confusion đã tính trên **đủ test**. Vẫn có kết quả cho
  mọi N client mỗi round. Code eval toàn bộ ở round đầu, mỗi 10 round và round cuối,
  đúng cách tối ưu đã ghi tại §1/`rebuild.md` §2.
- **Batch ghép:** pha ① dùng một forward trên `[G(x); x]`, nên F thấy 2B dòng; hai loss
  lấy mean riêng. CFG batch train vẫn là 512/512/256 như đã chốt. Đây là lựa chọn đã
  công bố, không phải notebook tự tăng batch cấu hình.
- **Đóng băng và tổng hợp:** module bị đóng băng dùng `eval()` và tắt gradient tham số;
  pha ② vẫn truyền gradient qua F tới G. G được gộp theo số dòng train của tập được chọn;
  BN mean/var lấy trung bình có trọng số, counter lấy max. Khớp `rebuild.md` §2.
- **Prediction:** production lưu `y_pred` tại round 50; confusion/metric/weights lưu từng
  round. Đây đúng ngoại lệ dung lượng đã ghi tại `rebuild.md` §2, không phải thiếu metric
  các round trước.
- **Probe:** chỉ 2 round, không lưu `y_pred`, `eval_all_every=1000`; vì C=100%, mọi client
  vẫn được train và eval ở cả hai round. Không dùng probe để xác nhận đã chạy 50 round.
- **Eval batch:** cả bốn notebook hiện là **16.384**. §7 còn để việc chốt 8.192 hay 16.384
  sau probe, nên đây là giá trị đang build, **chưa coi là lựa chọn hiệu năng đã nghiệm thu**.
- **50 round là mục tiêu toàn run:** driver có thể dừng sớm theo ngân sách phiên rồi resume.
  `require_resume=False` của các production hiện tại đúng với notebook khởi chạy mới;
  notebook tiếp tục phiên phải được sinh với cấu hình resume như §7.

### 9.3 Điểm chưa khớp đầy đủ với yêu cầu kiểm chứng đã ghi — **ĐÃ SỬA, rồi điều chỉnh 12-09**

**1. Cache check nay được cưỡng chế ở verifier, với ngưỡng.** `verify.py` tự tính lại từ
artifact: client **không được chọn** mà được eval lại phải cho confusion **y hệt** round trước
hoặc lệch ≤ `CACHE_TOL_ROWS = 100` dòng (nhiễu fp16 chéo GPU, §3.0); số client lệch tính lại
phải bằng `cache_mismatch` trong json; vượt 100 dòng ⇒ `ok=False`. Verifier in dòng `cache :`
nêu số client-round lệch và mức lệch lớn nhất. Test: 15 ca tamper + 1 ca "lệch 5 dòng được
chấp nhận" → 21/21. `driver.py` vẫn chỉ cảnh báo và ghi số. Bản cưỡng chế "= 0 tuyệt đối"
(11-09 15:45Z) chính là thứ làm hai run trước (C < 100 %) ERROR ở cell cuối.

**2. Cell cuối đã có `assert ok, "artifact verification FAILED …"`** sau `run.finish()` — kernel
sẽ ERROR khi verifier fail (điểm này thực ra đã có trước phiên rà soát). Output vẫn được Kaggle
lưu khi ERROR.

### 9.4 Bằng chứng xác nhận

- Chạy lại `scripts/validate_notebooks.py` qua watchdog: **pass cả 4 notebook**,
  bao gồm kiểm metadata, CFG, luồng cell và mã nhúng khớp nguồn.
- Chạy lại `tests/test_units.py` trên CPU qua watchdog: **24/24 pass**, gồm tham số F/G,
  shape, hai pha/gradient, aggregation, giữ batch cuối, metric khớp sklearn.
- Fixture artifact sạch qua verifier; phép thay `cache_mismatch` ở §9.3 tái hiện được
  điểm chưa cưỡng chế. Không phải kết quả huấn luyện trên dataset đầy đủ.
- Bốn file notebook local có **0 cell đã thực thi và 0 cell có output**. Phiên này xác nhận
  **bản build khớp quyết định**, chưa xác nhận runtime 2×T4 hoặc production hoàn tất.

---

## 11. Rebuild C = 100 % (2026-09-12) — đã đổi những gì, và tại sao
*(run `_full` sinh ở đây đã được push và dừng tay 12-09; bị thay bằng `_cos` ở §12 — giữ lại vì
mọi thay đổi C = 100 % vẫn đúng và là nền của `_cos`)*

Yêu cầu chủ dự án: chạy lại cả 20c/50c/100c, **mỗi round lấy toàn bộ 100 % client** thay vì
bốc ngẫu nhiên một phần; giữ mọi quyết định khác của §1; nhúng W&B key để theo dõi vài round
đầu. Đã sửa **4 file nguồn**, không đụng `proj/*.py`:

| file | thay đổi | lý do |
|---|---|---|
| `scripts/gen_notebook.py` | `SCENARIOS[50].participation` 0,2 → **1,0**; `SCENARIOS[100]` 0,1 → **1,0** | yêu cầu chính |
| `scripts/gen_notebook.py` | `run_tag` nay đi vào **title** ⇒ vào slug kernel | tách kernel `_full` khỏi kernel cũ |
| `scripts/gen_notebook.py` | markdown mở đầu nói đúng C = 100 %: bỏ câu "client không được chọn giữ confusion cũ" | prose cũ mô tả cơ chế không còn chạy |
| `scripts/validate_notebooks.py` | kỳ vọng `participation` = 1,0 cho cả 3 K | cổng tĩnh phải chặn nếu ai đó sinh lại bằng giá trị cũ |
| `tests/test_units.py` | đổi nhãn 1 dòng assert (`n_selected`) | nhãn cũ ghi "cho ba kịch bản", nay sai |

**`proj/` không đổi một byte** — `select_clients(N, 1.0, …)` đã trả `list(range(N))` sẵn từ
đầu, `participation` đã nằm trong `FINGERPRINT_KEYS` (⇒ checkpoint C = 20 % **không thể** bị
resume nhầm vào run C = 100 %), và `aggregate` đã chuẩn hoá theo tập tham gia.

**Lệnh đã chạy để sinh (ghi lại để tái lập):**
```bash
python scripts/gen_notebook.py --owner minhtriethihi --clients 20 50 --run-tag _full --max-hours 11.0
python scripts/gen_notebook.py --owner minhtriethihi --clients 100  --run-tag _full --max-hours 10.5
.agents/skills/kaggle-training-notebook/scripts/embed_wandb_key.py <nb> --metadata <meta>   # ×3
python scripts/validate_notebooks.py
```

| | run_name / W&B id | kernel slug | max_hours |
|---|---|---|---:|
| 20c | `pfedes_20c_full` | `minhtriethihi/pfedes-veremi-20-clients-full` | 11,0 |
| 50c | `pfedes_50c_full` | `minhtriethihi/pfedes-veremi-50-clients-full` | 11,0 |
| 100c | `pfedes_100c_full` | `minhtriethihi/pfedes-veremi-100-clients-full` | 10,5 |

### 11.1 Các lỗi cũ, và trạng thái ở bản này

| lỗi (§3) | trạng thái |
|---|---|
| `cache_mismatch ≠ 0` chéo GPU ⇒ `assert ok` ⇒ kernel ERROR | **không còn khả năng xảy ra**: ở C = 100 % không có client "không được chọn", nhánh cache không chạy, `cache_mismatch` ≡ 0. Vẫn giữ cả hai lớp phòng thủ đã sửa 11-09 (pin client c → worker `c % W`; `CACHE_TOL_ROWS = 100` ở verifier) |
| illegal memory access do 4 CUDA graph xen kẽ | đã sửa từ trước (`cudagraph_mark_step_begin()`), code không đổi |
| `recompile_limit = 8` ⇒ eval âm thầm chạy eager | đã sửa từ trước (đặt 64), code không đổi |
| payload `mp.Queue` phải là numpy | không đổi. Ở C = 100 % broadcast `set_clients` mỗi round mang **cả N client** (100c ≈ 160 MB) thay vì 10 — **đã được chứng minh chạy được**: broadcast `init` của run v1 100c mang đúng kích thước đó |
| W&B `resume="allow"` cùng id nối tiếp run cũ | tránh bằng run-tag `_full` (§5) |
| quota/12 h/session | ngân sách mới ở §2; `max_hours` giảm còn 10,5 cho 100c vì tail verify+commit đo được 18 phút |

### 11.2 Bằng chứng đã chạy trong phiên 12-09

`validate_notebooks.py` pass 20c, 20c-probe, 50c, 100c (và pass cả một bản `20c_s2` sinh thử
để xác nhận `--run-tag` + `--session` cho slug `…-full-s2` đúng, đã xoá sau khi kiểm).
Test local qua watchdog, **tất cả pass**: `test_units` 24/24, `test_ckpt_verify` 21/21,
`test_two_workers` (1 vs 2 worker bit-identical), `test_compile_gate` 7/7,
`test_smoke_real` (dữ liệu VeReMi thật, gồm phase "nothing-to-do" và gate `require_resume`),
`test_notebook_sim` CPU (chạy hết luồng cell của notebook probe + verifier).
Ba `.ipynb` production: mode 600, có `wandb.login(key=…)` nội tuyến, **không còn** nhánh
`UserSecretsClient`; `.gitignore` đã chặn `papers/**/notebook/**/*.ipynb`.

**Chưa có bằng chứng nào từ T4**: mọi con số ở §2 cho C = 100 % là ngoại suy. Phiên 1 chính
là phép đo (§7.4).

---

## 12. Run `_full` bị dừng (2026-09-12) — chẩn đoán, LR scan local, lịch LR cosine (2026-09-13)

### 12.1 Hiện tượng (W&B, 3 run `_full`, chủ dự án dừng tay ⇒ state `crashed`)

| run | round | f1_macro | accuracy | `ce_orig` (CE train local) |
|---|---|---|---|---|
| `pfedes_20c_full` (minhtriethihi) | 1→2→3 | 0,484 → 0,497 → 0,4995 | 0,5597 → 0,5610 → **0,5560** | 0,244 → 0,130 → 0,098 |
| `pfedes_50c_full` (khanhmay0304) | 1→2 | 0,387 → 0,390 | 0,483 → **0,476** | 0,275 → 0,120 |
| `pfedes_100c_full` (minhtran0601) | 1 | 0,324 | 0,424 | 0,239 |

Backend compiled cả train lẫn eval, `evaluated = selected = N`, `cache_mismatch = 0`, skip
AMP hằng — **pipeline chạy đúng**. Loss train local rơi rất nhanh trong khi test toàn cục đứng
yên/đi xuống. Ba kernel đã CANCEL; quota còn 30/30/28,2/28,2/28,2 h (13-09).

### 12.2 Nguyên nhân — không phải bug

1. **Round 1 đã là trần của F_k(x) trên test toàn cục.** `tinyproto` (per-client DAGSNet, cùng
   test) cho `clf` f1 0,51/0,40/0,33 và **phẳng suốt 50 round**; pFedES round 1 = 0,48/0,39/0,32,
   tức đúng "train local 1 epoch".
2. **F_k chỉ được giám sát bằng nhãn local.** Probe 20c r2: mọi lớp có f1 < 0,05 của một client
   đều là lớp client đó có < 1,2 % dữ liệu (client 9: 0,00 % `trafficCongestionSybil` = 22 % test
   ⇒ f1 lớp đó 0, accuracy 0,39; client 12 thiếu 4 lớp ⇒ f1 0,28). Corr(log prior local, f1
   per-class) = 0,50. G(θ) có mang tri thức toàn cục nhưng chỉ đi vào **nhánh x̂**; lúc test dùng
   F_k(x) thuần theo bài báo. Bài báo test trên **phân bố riêng client** (8:2) nên không gặp điều
   này; dự án test trên test toàn cục (caveat §8).
3. **Overfit local theo epoch — đo được từ `clients.csv` của run C < 100 % 50c (11-09, đã xoá)** (f1 của client theo số lần đã
   được train, mỗi lần = 1 epoch): 0 → 0,015; 1 → 0,376; **2 → 0,404**; 3 → 0,380; 5 → 0,385;
   8 → 0,366; 10 → 0,367; 14 → 0,313 (acc 0,469 → 0,496 → … → 0,389). Ở C = 100 % mỗi client
   đi qua 50 epoch ⇒ với LR hằng 1e-3, đường cong **chắc chắn** đi xuống sau round ~2.
4. Đã **loại trừ** bằng thí nghiệm local (weights probe r2, 4 client 9/12/8/0, test subsample
   1/10 = 1.076.135 dòng, khớp số đã lưu tới 1e-3):

   | biến thể suy luận | f1_macro | accuracy | kết luận |
   |---|---|---|---|
   | (a) F_k(x) như production | 0,481 | 0,556 | — |
   | (b) F_k(G(x)) "combined model" | 0,487 | **0,613** | G có tri thức toàn cục; client yếu +0,04/+0,07 f1; **không dùng** (lệch quy tắc suy luận bài báo, chủ dự án giữ eval nguyên) |
   | (c) BN tính lại trên x-only | 0,393 | 0,483 | **tệ hơn** ⇒ BN gộp [x̂;x] (deviation #1) tự nhất quán, không phải lỗi |
   | (d) logit − log prior local | 0,485 | 0,539 | vô hiệu: lớp thiếu không có đặc trưng, không chỉ lệch bias |
   | (e) metric theo prior local (tái trọng số confusion, kiểu "individual accuracy" bài báo) | 0,535 | 0,754 | zero compute; **không thêm** theo quyết định chủ dự án 13-09 (giữ 10 metric toàn cục) |

   Script: scratchpad phiên 13-09 `exp/diag_probe20.py` (không có trong repo).

### 12.3 Quyết định chủ dự án 2026-09-13

- **Một lịch LR thống nhất cho mọi dự án anh em** (`afpha`, `edl_cmso`, `fd_ids`, `tinyproto`,
  `pfedes`): cosine theo round, công thức của `afpha` (`src/afpha.py::lr_for_round`). Trước đó:
  afpha cosine 1e-3→1e-5; edl_cmso warmup+cosine 1e-3; fd_ids/tinyproto/pfedes hằng 1e-3.
- **pFedES chạy lại trước** (3 kịch bản, 5 tài khoản); các dự án khác **chưa sinh notebook** —
  chỉ ghi note vào CONTEXT.md của từng dự án (đã ghi 13-09), chạy lại khi chủ dự án yêu cầu.
- Giữ nguyên batch 512/512/256, 50 round, E = E_fe = 1, μ = 0,5, G = DAGSNet-66, C = 100 %,
  eval **10 metric toàn cục mỗi client mỗi round** (không thêm metric phụ).
- Chốt giá trị đỉnh/đáy LR **sau LR scan local** (12.4), Kaggle probe nếu còn nghi ngờ.

### 12.4 LR scan local (13-09, RTX 3050, 1 process, eager, B = 512) — bằng chứng cho lịch cosine

Thiết lập: 6 client thật của phân mảnh 100c (id 3/5/7/11/12/14, 2,10 M dòng, 14–16 lớp),
C = 100 %, eval trên subsample 1/10 của test (stride 10, 1.076.135 dòng), cùng hàm
`client_update`/`aggregate`/`fold_bn`/`eval_model` của `proj/` (script scratchpad
`exp/lrscan_single.py`, không có trong repo; 4,6 phút/round). Nhiễu cuDNN benchmark giữa hai
run cùng seed ≈ ±0,005 f1. Chủ dự án **dừng** loạt LR hằng sau 5 round của 1e-3 (3e-4/1e-4 chưa
chạy) và chỉ giữ 1e-3 cosine.

| round | LR hằng 1e-3: f1 / acc | cosine 1e-3→1e-5 (T = 10): lr → f1 / acc |
|---|---|---|
| 1 | 0,298 / 0,356 | 1,0e-3 → **0,306** / 0,360 |
| 2 | **0,299** / 0,335 | 9,7e-4 → 0,304 / 0,346 |
| 3 | 0,294 / 0,326 | 8,8e-4 → 0,291 / 0,328 |
| 4 | 0,290 / 0,330 | 7,5e-4 → 0,289 / 0,334 |
| 5 | 0,284 / 0,324 | 5,9e-4 → 0,295 / 0,329 |
| 6–10 | (dừng) | 4,2e-4 … 1,0e-5 → 0,289 / 0,291 / 0,288 / 0,289 / **0,289** ; acc 0,324 → 0,323 |

Đọc: (1) cosine **chặn đà trôi** — phẳng ở 0,289 từ r3 (−5,6 % so với đỉnh) trong khi LR hằng
tiếp tục đi xuống; (2) cosine **không kéo lại đỉnh**: mức plateau ≈ mức LR hằng đạt ở cùng **tổng
LR tích luỹ** (≈ 5 epoch-tương-đương cho cả hai: 0,284 vs 0,289), `ce_orig` vẫn rơi 0,32 → 0,007;
(3) suy ra cho T = 50 đỉnh 1e-3: 25 round đầu LR > 5e-4, tích luỹ ≈ 25 epoch-tương-đương ⇒
**dự đoán đường cong giảm tới ~round 25 rồi phẳng**, mức cuối thấp hơn đỉnh đáng kể (−22 % ở 14 epoch trong
run LR hằng C < 100 %). Muốn đường cong đi lên rồi phẳng gần đỉnh thì phải hạ **đỉnh** LR (1e-4 → tích
luỹ 2,5; 3e-4 → 7,5) — chưa đo; chủ dự án chọn **giữ 1e-3 → 1e-5** để đồng bộ afpha và push ngay.

### 12.5 Đã đổi trong repo (13-09) và đã push

`proj/pfedes.py::lr_at` (+ `LR_SCHEDULES`), `driver.py` (AdamW theo `lr_at`, `lr` vào stats/log/
W&B/print), `ckpt.py` (`FINGERPRINT_KEYS` + `lr_schedule`, `lr_min`, `rounds`), `verify.py`
(so `lr` từng client với lịch), `scripts/gen_notebook.py` (`PAPER` + markdown + CFG),
`scripts/validate_notebooks.py` (kỳ vọng `lr_schedule="cosine"`, `lr_min=1e-5`),
`scripts/watch_prod.py` (mới), `rebuild.md` (§1 hàng LR, §2 deviation #11, §3 fingerprint 24 khoá),
tests: `test_units` (+8 ca `lr_at`, 32/32), `test_ckpt_verify` (fixture có `lr`, +1 ca tamper
"LR lệch lịch", 22/22), `test_smoke_real` (kế hoạch 4 round cố định; phiên nối tiếp mô phỏng
bằng xoá round 4 → resume phải **bit-identical**; `rounds` khác bị chặn ở cổng; pass),
`test_compile_gate` 7/7, `test_notebook_sim` pass, `validate_notebooks` pass 4/4.
**Nợ:** `test_two_workers`, `test_gpu_local` (watchdog, §7.6).

Sinh + nhúng key + validate + push 13:19–13:20Z (mỗi kernel version 1, RUNNING ngay):

| | owner / kernel | run_name | max_hours | phiên dự kiến |
|---|---|---|---:|---|
| 20c | `minhtriethihi/pfedes-veremi-20-clients-cos` | `pfedes_20c_cos` | 11,0 | 2 (≈18,5 h) |
| 50c | `khanhmay0304/pfedes-veremi-50-clients-cos` | `pfedes_50c_cos` | 11,0 | 3 (≈25 h) |
| 100c | `odixe0502/pfedes-veremi-100-clients-cos` | `pfedes_100c_cos` | 10,5 | 5 (≈48,5 h): s1–s2 odixe0502 → s3–s4 minhtrit06 → s5 minhtran0601 |

`notebook/20c_probe/` cũng được sinh lại (validator đòi module khớp nguồn) nhưng **không push**.

**Tiến trình phiên 1 (đo thật, 13-09):** round-time ổn định 20c **22 phút** (train 1050 + eval
274 s), 50c **29 phút** (1012 + 720), 100c **42 phút** (1280 + 1248; round 1 = 51 phút vì compile).
Round 1 của cả ba **trùng `_full`** (0,485/0,389/0,324) — đúng vì lr(1) = 1e-3 ở cả hai lịch.
Đường cong tới giờ: 20c f1 0,485 → **0,501 @r5** → 0,490 @r10 → 0,486 @r15 → 0,4875 @r20 (đã
phẳng), acc 0,562 → 0,534; 50c 0,389 → 0,381 @r5 → 0,371 @r10 → 0,364 @r15; 100c 0,324 →
0,308 @r5 → 0,296 @r10 → 0,293 @r13. Mọi hàng: `lr` đúng lịch, compiled, evaluated = N,
cache_mismatch 0, skip hằng, VRAM ≈ 7 GiB.

**100c phiên 1 dừng theo ngân sách sau round 13** (22:4xZ 13-09; 9,45 h quota = elapsed; kernel
COMPLETE, W&B `finished`, verifier pass). Vì `worst` round = round 1 (có compile, 3085 s) nên
gate dừng sớm hơn 1 round so với ước lượng — chấp nhận. **Phiên 2 đã push 22:52Z**:
`odixe0502/pfedes-veremi-100-clients-cos-s2` (dir `notebook/100c_s2/`, `--kernel-source
odixe0502/pfedes-veremi-100-clients-cos`, max_hours 10,5), RUNNING. Với ~13 round/phiên, 100c cần
**4 phiên**: s1–s2 odixe0502 (còn 20,55 h trước s2), s3–s4 **minhtrit06** (30 h) qua checkpoint
dataset; `minhtran0601` (28,2 h) thành dự phòng. 20c/50c phiên 1 dự kiến dừng ~00:20Z 14-09
(20c ≈ round 29, 50c ≈ round 22) → sinh `--session 2` cùng tài khoản (còn 18,7 h mỗi tài khoản).
Các file `.ipynb` `_full` cũ đã xoá (sinh lại được bằng `--run-tag _full` nếu cần).

---

## 13. Phiên 1 `_cos` kết thúc (20c/50c), phiên 2 push (2026-09-14 ~00:51Z)

### 13.1 Kết quả phiên 1 (kéo về, verify local pass)

| | 20c (`minhtriethihi`) | 50c (`khanhmay0304`) | 100c (`odixe0502`, §12.5) |
|---|---|---|---|
| round đạt / lý do dừng | **28** / gate ngân sách 11,0 h (round kế + 15 phút > budget) | **21** / gate ngân sách | 13 / gate (9,45 h) |
| round-time ổn định | **1321 s** (train 1047 + eval 274) | **1730 s** (1011 + 715) | 2531 s (1280 + 1248) |
| round 1 (có compile) | 1448 s | 1955 s | 3085 s |
| startup+prepack+compile | 9,4 phút | 9,2 phút | — |
| f1_macro: r1 → đỉnh → cuối | 0,485 → **0,501 @r5** → 0,484 @r28 | 0,389 → **0,389 @r1** → 0,362 @r21 | 0,324 → 0,324 @r1 → 0,292 @r14 |
| accuracy r1 → cuối | 0,562 → 0,528 | 0,483 → 0,434 | 0,424 → 0,376 |
| f1 std / min / max (cuối) | 0,096 / 0,280 / 0,641 | 0,063 / 0,226 / 0,493 | — / 0,137 / 0,379 |
| verifier Kaggle / local | 28/50 pass / pass (`--require-rounds 28`) | 21/50 pass / pass (`--require-rounds 21`) | 13/50 pass |
| kéo về | 950 MB, `runs/pulls/20c_cos_s1/` | 1,7 GB, `runs/pulls/50c_cos_s1/` | (chưa kéo) |

Mọi hàng: `lr` đúng `lr_at`, backend compiled cả train/eval, `evaluated` = `selected` = N,
`cache_mismatch` = 0, skip AMP hằng (20c 3–14 / 23–34; 50c 0–4 / 9–45), VRAM 7,0–7,4 GiB.
Đường cong khớp dự đoán §12.4: 20c phẳng ~0,484–0,487 từ r15 (−3,4 % so đỉnh); 50c giảm chậm
rồi phẳng ~0,362–0,364 từ r15 (−7 %); 100c 0,292 @r14 (−10 %). LR ở r28 = 4,26e-4, r21 = 6,46e-4.
File 0 byte trong output = marker `complete/round_NNN.done` + `proj/__init__.py` (hợp lệ, không
phải pull hỏng). `executed.ipynb` (19 output render) nằm cạnh trong cùng thư mục pull.

### 13.2 Phiên 2 (push 00:51Z, RUNNING ngay, chủ dự án chốt: cùng tài khoản phiên 1)

```bash
python scripts/gen_notebook.py --owner minhtriethihi --clients 20 --run-tag _cos --session 2 \
  --require-resume --kernel-source minhtriethihi/pfedes-veremi-20-clients-cos --max-hours 11.0
python scripts/gen_notebook.py --owner khanhmay0304 --clients 50 --run-tag _cos --session 2 \
  --require-resume --kernel-source khanhmay0304/pfedes-veremi-50-clients-cos --max-hours 11.0
embed_wandb_key.py ×2 → validate_notebooks.py (pass 7/7) → kaggle_as.py <acct> -- kaggle kernels push
```

| | kernel | nguồn resume | quota trước push | dự kiến |
|---|---|---|---:|---|
| 20c s2 | `minhtriethihi/pfedes-veremi-20-clients-cos-s2` | output s1 (round 28) | 17,70 h | 22 round ≈ 8,1 h + startup ⇒ **đủ 50 round**, COMPLETE ≈ 09:30Z 14-09 |
| 50c s2 | `khanhmay0304/pfedes-veremi-50-clients-cos-s2` | output s1 (round 21) | 17,89 h | ~22 round (→ r43), dừng ≈ 11:40Z; **cần s3** ≈ 3,9 h (quota còn ≈ 6,8 h) |

W&B: cùng run id (`pfedes_20c_cos`, `pfedes_50c_cos`), state `running` lại ngay sau push.

### 13.3 100c s2 — W&B mất sync 47 phút rồi tự hồi (không phải kernel treo)

s2 resume đúng: round 14 ghi 23:53Z (train 1643 + eval 1394 = 3041 s, gồm compile), f1 0,2917.
Round 15 hoàn tất 00:40:04Z (`_timestamp`) nhưng **chỉ xuất hiện trên W&B lúc 01:27Z**; trong
khoảng đó `heartbeatAt` đứng ở 00:40:17Z, state `crashed`, kernel Kaggle vẫn RUNNING. Khi sync trở
lại: state `running`, hàng 15 giữ đúng timestamp gốc ⇒ container mất mạng/W&B sync stall, training
không gián đoạn. Round 15: train 1461 + eval 1346 = **2811 s** (s1: 1280 + 1248 = 2531 s, +11 %,
dưới ngưỡng 30 % của §5), f1 0,2914, acc 0,3754, `evaluated` 100, mismatch 0, VRAM 7,3 GiB.
Dự kiến s2 (10,5 h) đạt ~12–13 round ⇒ dừng ≈ r26 lúc ~09:20Z 14-09; 100c vẫn cần **4 phiên**
(s3–s4 `minhtrit06` qua checkpoint dataset, §7.4).

---

## 14. Phiên 2 kết thúc cả ba, 20c hoàn tất, s3 push (2026-09-14 17:00–18:00Z)

### 14.1 Kết quả phiên 2 (kéo về `runs/pulls/{K}c_cos_s2/`, verify local pass, executed.ipynb 14/15 cell có output)

| | 20c (`minhtriethihi`) | 50c (`khanhmay0304`) | 100c (`odixe0502`) |
|---|---|---|---|
| round đạt (s1 + s2) | **50** (28 + 22) — **XONG** | 42 (21 + 21) | 25 (13 + 12) |
| lý do dừng s2 | hết 50 round | gate ngân sách 11,0 h | gate 10,5 h sau r25 @ 9,39 h (worst = r14 3041 s ×1,15 + 15 phút > 10,5) |
| round-time trung vị (cả run) | 1320 s (1046 + 273) | 1723 s (1011 + 709) | 2649 s (1309 + 1280); s2 2676–2845 s |
| f1_macro r1 → đỉnh → cuối | 0,4851 → **0,5010 @r5** → **0,4856 @r50** (−3,1 %) | 0,3893 → 0,3893 @r1 → 0,3643 @r42 (−6,4 %) | 0,3238 → 0,3238 @r1 → 0,2883 @r25 (−11,0 %) |
| accuracy r1 → cuối | 0,5615 → 0,5276 | 0,4826 → 0,4334 | 0,4248 → 0,3701 |
| f1 std / min / max cuối | 0,095 / 0,282 (#12) / 0,645 (#6) | 0,064 / 0,230 (#15) / 0,508 (#11) | 0,048 / 0,136 (#14) / 0,385 (#57) |
| pull | 1,9 GB | 3,4 GB | 4,0 GB |

Đủ 10 metric round cuối + per-class + confusion: `report_data/README.md` §2 và các CSV. 20c ở r50:
precision_macro 0,5921, recall_macro 0,5656, f1_weighted 0,5054, precision_weighted 0,6805. Lớp yếu
nhất (20c, mean trên client): `timeDelayAttack` 0,11, `positionMirroring` 0,10, `suddenConstantSpeed` 0,25,
`dataReplay` 0,27, `benign` 0,30 (recall gộp 0,26 — benign bị đoán thành dataReplay/timeDelay/
positionMirroring); mạnh nhất `dosAttack` 0,93. Đường cong khớp dự đoán §12.4: 20c phẳng 0,484–0,487
từ r15 tới r50 (LR < 1e-4 từ r41 không đổi gì); 50c phẳng 0,362–0,365 từ r15; 100c vẫn giảm chậm
0,292 (r14) → 0,287 (r23) → 0,288 (r25).

Mọi hàng: `lr` đúng `lr_at`, compiled/compiled, `evaluated` = N, `cache_mismatch` = 0, VRAM 7,0–7,3 GiB.

### 14.2 Ghép 20c thành một run duy nhất (yêu cầu chủ dự án 14-09)

`merge_sessions.py --out runs/merged/pfedes_20c_cos pulls/20c_cos_s1 pulls/20c_cos_s2`: 28 round chồng
**byte-identical** (weights/resume/metrics/confusion/logs), history.csv khớp từng hàng, 1..50 liền
mạch; `logs/sessions.json` ghi phiên 1 sinh r1–28, phiên 2 sinh r29–50; `logs/sessions/{1,2}/` giữ
log kernel, `manifest.json`, `executed.ipynb` mỗi phiên. Script tạo thêm thư mục rỗng
`checkpoints/ protos/ client_log/` (tên của dự án khác) — đã xoá. `verify_run --require-rounds 50` pass.
Đây là thư mục **chuẩn** cho report 20c; hai pull gốc vẫn giữ.

### 14.3 50c s3 (push 17:03Z 14-09, RUNNING ngay)

`gen_notebook.py --owner khanhmay0304 --clients 50 --run-tag _cos --session 3 --require-resume
--kernel-source khanhmay0304/pfedes-veremi-50-clients-cos-s2 --max-hours 7.0` (quota 7,69 − 0,5;
cần ≈ 4,2 h cho 8 round: startup 9 phút + r43 có compile ~33 phút + 7 × 28,7 phút + tail). Cùng W&B id
`pfedes_50c_cos`. Dự kiến COMPLETE ≈ 21:20Z 14-09, **đủ 50 round** — sau đó 50c không cần phiên nữa.

### 14.4 100c s3 trên `minhtrit06` qua checkpoint dataset, max-hours 11,5 (chủ dự án chốt 14-09)

Chủ dự án: "kéo dài mỗi phiên tới gần 12 h" ⇒ `--max-hours` 10,5 → **11,5** (T = 50 giữ nguyên, nằm
trong fingerprint). Tính từ cổng `driver._rounds` (elapsed + 1,15 × worst + 15 phút ≤ max_seconds,
đồng hồ T0 tính cả startup): 11,5 h ⇒ **13–14 round/phiên** (10,5 h ⇒ 11–12); phiên kết thúc ≤ ~11,4 h
(tail verify + commit 8 GB đo 18 phút, gate đã chừa 15 phút + 0,15 × worst) — còn ~35 phút dưới trần
12 h. Do đó 100c chỉ cần **s3 + s4**, cả hai trên `minhtrit06` (30 h): s3 → r38–39, s4 → r50.
`odixe0502` (11,08 h) bị bỏ vì với 10,5 h sẽ cần 3 phiên.

Các bước đã làm (thứ tự, tất cả bằng `kaggle_as.py minhtrit06 -- …`):
1. `kaggle datasets create -p <stage> -r zip -t` với `<stage>/pfedes_100c_cos/…` → **SAI LAYOUT**:
   `-r zip` nén mỗi thư mục cấp 1 thành `<tên>.zip`, Kaggle giải nén *nội dung* zip vào gốc ⇒ mất tên
   `pfedes_100c_cos` ⇒ `ckpt._attached_source` (lọc `run_name in p.parts`) sẽ không thấy gì.
2. Sửa bằng `kaggle datasets version -p <stage> -r zip -t -m …` với `<stage>/runs/pfedes_100c_cos/…`
   → v2 `ready` sau ~2 phút; **154/154 file khớp tên và kích thước** với local (4 199 570 997 byte).
   Dataset: **`minhtrit06/pfedes-100c-cos-ckpt-s2`** (private). Bài học đã ghi vào skill
   `references/multi-account.md`.
3. CPU probe `scripts/gen_ckpt_probe.py` (nhúng `proj/ckpt.py`, chạy `resolve_resume` thật, assert
   `last == 25`): kernel `minhtrit06/pfedes-100c-cos-ckpt-probe` COMPLETE sau 6 phút, log:
   mount `/kaggle/input/datasets/minhtrit06/pfedes-100c-cos-ckpt-s2/pfedes_100c_cos`, fingerprint
   `f7c3d21a510cf7dc` khớp, `25 marker(s), 25 verified, imported 1..25`, `PROBE_OK`. 0 GPU quota.
4. `gen_notebook.py --owner minhtrit06 --clients 100 --run-tag _cos --session 3 --require-resume
   --dataset-source minhtrit06/pfedes-100c-cos-ckpt-s2 --max-hours 11.5` (metadata: 2 dataset dữ liệu +
   dataset checkpoint, `kernel_sources` rỗng) → nhúng key → `validate_notebooks.py` pass 9/9 → push
   17:47Z, RUNNING. Cùng W&B id `pfedes_100c_cos`.

Dự kiến s3: startup ~16 phút (import 4 GB từ dataset + prepack + compile), r26 ≈ 3050 s, các round sau
≈ 2650–2850 s ⇒ dừng sau r38 hoặc r39 lúc ≈ 05:00Z 15-09. **s4** (§7.3) cùng tài khoản qua
`--kernel-source minhtrit06/pfedes-veremi-100-clients-cos-s3`, không cần dataset.

### 14.5 Đường dẫn sau khi chuyển repo vào `~/nckh/veremi/`

Đã sửa (`grep -rn 'nckh/(pfedes|dataset|fd_ids|tinyproto|afpha|edl_cmso)'` không còn kết quả ngoài
`nckh/veremi/`, trừ `wandb/` log cũ và `knowledge/dataset_audit.json` là bằng chứng lịch sử):
`.agents/skills/kaggle-training-notebook/SKILL.md`, `references/{federated-afpha,per-client-fl-veremi,
multi-account}.md` (+ mirror `.claude/skills`), `tests/test_smoke_real.py`, `knowledge/{audit_dataset,
fl_class_stats}.py` (đường dẫn dataset/tinyproto + output nay ghi cạnh script thay vì scratchpad cũ),
`knowledge/DATASET.md`, `.codex/config.toml` (header helper trỏ về `scripts/kaggle_mcp_headers.py`
của chính repo), CONTEXT.md. Memory Claude Code: chép 3 file từ
`~/.claude/projects/-home-odixe-nckh-pfedes/memory/` sang `-home-odixe-nckh-veremi-pfedes/memory/`.
Hook SessionStart dùng `$CLAUDE_PROJECT_DIR` nên không đổi. `.mcp.json`/`.vscode/mcp.json` không chứa
đường dẫn.

### 14.6 Số liệu report — `papers/pfedes-yi-2025/report_data/`

`scripts/report_data.py --out papers/pfedes-yi-2025/report_data 20c=runs/merged/pfedes_20c_cos
50c=runs/pulls/50c_cos_s2/runs/pfedes_50c_cos 100c=runs/pulls/100c_cos_s2/runs/pfedes_100c_cos`
(chạy dưới watchdog, 136 MiB). Đọc từ `metrics/round_NNN.json`, `confusion/*.npy`, `history.csv`,
`reports/manifest.json`, `logs/sessions.json`; **không lấy gì từ W&B**. README.md là bản tóm tắt
tiếng Việt để chép vào report; `rounds_Kc.md` là bảng bắt buộc "mọi round × 10 metric". Hình dùng
palette skill `dataviz` (20c xanh `#2a78d6`, 50c cam `#eb6834`, 100c lục `#1baf7a`; confusion ramp
xanh một sắc; không dual-axis). Chạy lại sau khi 50c/100c đủ 50 round với thư mục `runs/merged/`.

---

## 15. Phiên 3 kết thúc (50c XONG, 100c r39), 50c ghép, 100c s4 push (2026-09-15 10:40–11:30Z)

### 15.1 Kết quả phiên 3 (kéo về `runs/pulls/{50c,100c}_cos_s3/`, verify local pass, executed.ipynb 14/15 cell có output)

| | 50c (`khanhmay0304`, s3 max-hours 7,0) | 100c (`minhtrit06`, s3 max-hours 11,5) |
|---|---|---|
| round đạt (s1 + s2 + s3) | **50** (21 + 21 + 8) — **XONG** | 39 (13 + 12 + 14) |
| lý do dừng s3 | hết 50 round | gate ngân sách 11,5 h sau r39 |
| startup+prepack+compile s3 | 18,2 phút | 22,3 phút (import 4 GB từ dataset) |
| round-time steady s3 | 1688 s (train 1018 + eval 668) | 2586 s (train 1299 + eval 1285) |
| f1_macro r1 → đỉnh → cuối | 0,3893 → 0,3893 @r1 → **0,3652 @r50** (−6,2 %) | 0,3238 → 0,3238 @r1 → 0,2900 @r39 (−10,4 %) |
| accuracy r1 → cuối | 0,4826 → 0,4335 | 0,4248 → 0,3692 |
| f1 std / min / max cuối | 0,063 / 0,230 (#15) / 0,502 (#11) | 0,049 / 0,140 (#14) / 0,389 (#57) |
| verifier Kaggle / local | 50/50 / pass `--require-rounds 50` | 39/50 / pass `--require-rounds 39` |
| pull | 4,5 GB (~30 phút) | 6,2 GB (~45 phút) |

Mọi hàng: `lr` đúng `lr_at` (50c r50 = 1e-5; 100c r39 = 1,28e-4), compiled/compiled, `evaluated` = N,
`cache_mismatch` = 0, skip AMP hằng, VRAM 7,0–7,1 GiB. 50c ở r50 đủ 10 metric: accuracy 0,4335,
precision_macro 0,4882, precision_weighted 0,6172, recall_macro 0,4481, f1_weighted 0,4171. Lớp yếu nhất
50c (f1 mean trên client): `timeDelayAttack` 0,085, `positionMirroring` 0,090, `suddenConstantSpeed` 0,127,
`constantSpeedOffset` 0,171, `dataReplay` 0,188, `benign` 0,225 (recall gộp 0,19); mạnh nhất `dosAttack` 0,806,
`randomPositionOffset` 0,635. 100c: đường cong phẳng 0,288–0,290 từ r25 tới r39 — đúng dự đoán §12.4.

### 15.2 Ghép 50c thành một run duy nhất

`merge_sessions.py --out runs/merged/pfedes_50c_cos pulls/50c_cos_s1 pulls/50c_cos_s2 pulls/50c_cos_s3`:
21 round chồng s1/s2 và 42 round chồng s2/s3 **byte-identical** (weights/resume/metrics/confusion/logs),
history.csv khớp từng hàng, 1..50 liền mạch, round mới mỗi phiên [21, 21, 8]; `logs/sessions.json` +
`logs/sessions/{1,2,3}/` (log kernel, `manifest.json`, `executed.ipynb`). Đã xoá thư mục rỗng
`checkpoints/ protos/ client_log/`; layout top-level giống hệt `pfedes_20c_cos`. `verify_run --require-rounds 50`
pass (RSS đỉnh 1,1 GB). Thư mục **chuẩn** cho report 50c.

### 15.3 100c s4 (push 10:44Z 15-09, RUNNING ngay)

`gen_notebook.py --owner minhtrit06 --clients 100 --run-tag _cos --session 4 --require-resume
--kernel-source minhtrit06/pfedes-veremi-100-clients-cos-s3 --max-hours 11.5` → `embed_wandb_key.py` →
`validate_notebooks.py` pass 10/10 → push. Metadata: 2 dataset dữ liệu + `kernel_sources` = s3 (cùng tài khoản,
không cần dataset checkpoint). Cần 11 round: r40 có compile ≈ 3050 s + 10 × 2586 s + startup ~20 phút + tail
≈ **8,6 h** ⇒ COMPLETE ≈ 19:30Z 15-09, quota `minhtrit06` sau đó ≈ 10 h. W&B: `Resuming run` xảy ra ở giây
18 của kernel (log s3), heartbeat s4 đứng ở 10:58:59Z ⇒ state `crashed` trong khi kernel RUNNING — sync stall
như §13.3, không phải treo.

### 15.4 Số liệu report — sinh lại 11:23Z 15-09

`scripts/report_data.py --out papers/pfedes-yi-2025/report_data 20c=runs/merged/pfedes_20c_cos
50c=runs/merged/pfedes_50c_cos 100c=runs/pulls/100c_cos_s3/runs/pfedes_100c_cos` (watchdog, RSS 147 MiB).
README §1 nay ghi 50c **HOÀN TẤT 3 phiên (21/21/8)**, round-time trung vị 28,6 phút, tổng 23,9 h round-time;
100c 39/50 "số liệu tạm". Chạy lại lần cuối sau khi 100c ghép (§7.2).

### 15.5 `report.md` bản tạm (viết 11:40–12:00Z 15-09, yêu cầu chủ dự án "viết trước một phần")

`papers/pfedes-yi-2025/report.md`: thân bài viết tay từ `report_data/` + `rebuild.md` + `DATASET.md` + CONTEXT §12;
phụ lục A/B/C nối nguyên bảng `rounds_{20,50,100}c.md` (bỏ 4 dòng đầu). Cổng kiểm đã chạy: 10 metric round cuối
khớp `summary.json` (0 lệch), phụ lục 50/50/39 hàng, mọi hình tồn tại. Phân tích thêm ngoài README (tính từ CSV,
script scratchpad không lưu): Spearman(số dòng client, f1_macro) = 0,69 / 0,53 / 0,44; số client f1 < 0,30 =
1/20, 9/50, 52/100; cụm 4 lớp không tách được `benign`/`timeDelayAttack`/`positionMirroring`/`dataReplay` (hàng
`benign` ở 20c: 26/20/19/16 %; ở 50c/100c `dataReplay` là dự đoán lớn nhất ≈ 21 % cho cả ba lớp kia); lớp nhỏ mất
nhiều nhất khi N tăng (`constantSpeedOffset` 0,36→0,17→0,09). **Khi 100c xong:** thay mọi ô "(tạm, r39)" ở §1, 5.1,
5.2, 5.3, 5.4, 5.5, 8 và bảng phiên §4 (quota s4), nối lại phụ lục C từ `rounds_100c.md` mới, bỏ dòng trạng thái
TẠM ở đầu. **Chủ dự án 15-09 ~12:00Z:** xoá khỏi `rebuild.md` và `report.md` mọi chi tiết về bản C < 100 %
và bản `_full` (LR hằng); mỗi file chỉ giữ một "lưu ý lịch sử" rằng hai bản đó đã bị thay thế và không đóng góp
số nào. Đã làm: `rebuild.md` (lưu ý đầu file; §1 hàng C = 100 %; §2 #3/#5/#8/#11 viết lại cho C = 100 %; §4
thay bảng ước lượng cũ bằng số đo thật `_cos`), `report.md` (lưu ý cuối §2; bỏ tham chiếu −22 % ở §2, 5.1, 7).
CONTEXT.md §10–12 giữ làm nhật ký khi đó; **16-09 §10 đã cắt** cùng với `runs/pulls/v1_record/` (§16.4).

---

## 16. 100c XONG, ghép 4 phiên, report chính thức, dọn pull (2026-09-16 03:30–07:00Z)

### 16.1 Kết quả s4 và bài học kéo output

s4 (`minhtrit06/pfedes-veremi-100-clients-cos-s4`) COMPLETE ~20:15Z 15-09: resume `39 marker(s), 39 verified`,
11 round r40–50, verifier trên Kaggle 50/50, cache re-check 0, compiled/compiled, `lr` r50 = 1e-5, VRAM 7,1–7,3 GiB.
Startup 34,3 phút (import 6 GB từ kernel s3) — dài hơn s3 (22 phút); round steady **2907 s** (train 1451 + eval 1452)
so s3 2586 s (+12 %, eval +13 % trên cùng cấu hình ⇒ node T4 chậm hơn, không phải code). Round lớn nhất của cả run
100c là **r48 = 3114 s** (phiên 4), không phải round compile (r40 = 3109 s). Phiên: 9,36 h round-time + tail.
W&B "crashed" từ 10:59Z 15-09 chỉ là sync stall như §13.3: khi kết thúc state `finished`, đủ 50 hàng.

**Kéo output — `kaggle kernels output` không dùng được cho pull lớn** (đã ghi vào skill `pull-outputs.md`):
1. Lần 1 (toàn bộ 8,4 GB) tải file nhỏ + `preds/` 1,08 GB ổn, rồi `IncompleteRead` giữa `weights/round_040.pt`
   → file **0 byte**; lần 2 (`--file-pattern` loại weights r1–39) coi file 0 byte đó là "found more recently modified
   local copy" và bỏ qua; lần 3 **treo** ~1 h trên socket CDN `kaggleusercontent` chết (Send-Q kẹt, hệ thống 9 KB/s trong
   khi speed test 14,7 MB/s) — `requests.get(...).content` trong CLI không có timeout.
2. Cách làm được: liệt kê output qua API (`list_kernel_session_output`, có `url` ký sẵn và `log`) rồi `curl -C -
   --retry --speed-limit 20480 --speed-time 60` từng file: 11 × 167 MB trong **3,5 phút** (~8 MB/s). Gói thành
   `$H/pull_kernel_output.py` (`--exclude/--include` regex, so kích thước local↔remote qua Range probe, làm mới URL
   mỗi pass, ghi `<slug>.log`); đã test: tải, cắt file giữa chừng, chạy lại → resume từ byte cụt, byte-identical với merged.
3. Chỉ kéo phần **chưa có**: weights r1–39 (6,1 GB) đã có trong pull s3 byte-identical (kernel s4 import và verify chúng)
   nên s4 chỉ kéo weights r40–50 + phần nhỏ. `merge_sessions.py` chỉ so file có ở cả hai pull ⇒ pull thiếu vẫn ghép được.
4. Kéo thêm phần nhỏ của **s1** (`odixe0502/pfedes-veremi-100-clients-cos`, 26 MB, không weights) để `sessions.json`
   ghi đúng 4 phiên [13, 12, 14, 11] thay vì phiên "1" gộp s1+s2.

### 16.2 Ghép và verify 100c

`merge_sessions.py --out runs/merged/pfedes_100c_cos pulls/100c_cos_s1 pulls/100c_cos_s2 pulls/100c_cos_s3 pulls/100c_cos_s4`:
13 / 25 / 39 round chồng **byte-identical**, history khớp từng hàng, 1..50 liền mạch, 8,9 GB; xoá thư mục rỗng
`checkpoints/ protos/ client_log/`; chép `executed.ipynb` 4 phiên (mỗi cái 14/14 cell có output) vào `logs/sessions/{1..4}/`.
`verify_run --require-rounds 50` pass (RSS 1,6 GB). `report_data.py` chạy lại với ba thư mục merged (RSS 159 MiB):
100c r50 f1_macro **0,2900** (acc 0,3692, precision_macro 0,3987, precision_weighted 0,5576, recall_macro 0,3709,
f1_weighted 0,3556; std/min/max 0,0489 / 0,1424 (#14) / 0,3855 (#57)); đỉnh r1 0,3238 (−10,5 %); phẳng 0,287–0,291 từ r25;
54/100 client f1 < 0,30; Spearman(rows, f1) 0,44; round-time trung vị 2649 s (train 1318 + eval 1290), tổng 37,6 h.
Per-class 100c r50 lệch ≤ 0,004 so r39 (đường cong phẳng).

### 16.3 Dọn: mỗi file một bản (chủ dự án 16-09: "file gì đã có thì chỉ để 1 bản duy nhất")

Script scratchpad kiểm từng file của 9 pull `_cos` có bản byte-identical trong `merged/` (round artifact ở gốc; log kernel /
`manifest.json` / `executed.ipynb` ở `logs/sessions/<n>/`; `history.csv`/`clients.csv` đã được merge kiểm hàng): **0 file
không có bản**. Phần không có bản: `wandb/` của pull (8 file, vài MB) → chép vào `logs/sessions/<n>/wandb/`; `proj/` của
mọi pull **giống từng byte** `proj/` repo (chỉ `__init__.py` rỗng + `__pycache__` khác) → bỏ. Xoá 9 pull = **25,0 GiB**;
`runs/` 30 GB → **16 GB** (merged 1,9 + 4,5 + 8,9 + `probe20` 78 MB). Bản sao dự phòng duy nhất còn lại là output trên Kaggle.

### 16.4 Report chính thức và xoá bản C < 100 %

`report.md`: bỏ dòng trạng thái TẠM; §1, 4 (bảng phiên: 100c 4 phiên 13+12+14+11, 13-09 13:20Z → 15-09 20:15Z, quota
38,9 h = odixe0502 18,9 + minhtrit06 20,0), 5.1–5.5 (thêm hình `per_class_f1_100c.png`, `confusion_100c.png`; ghi chú
round lớn nhất 100c là r48 phiên 4), 7, 8 cập nhật số r50; phụ lục C nối từ `rounds_100c.md` mới (50 hàng). Cổng kiểm:
10 metric × 3 K khớp `summary.json` (0 lệch), phụ lục 50/50/50 hàng, 9 hình tồn tại.
**Chủ dự án 16-09:** xoá `runs/pulls/v1_record/` (bản C = 20 %/10 % của 11-09, chỉ còn text 1,8 MB) và **không nhắc bản đó
nữa**: `report.md` §2 và `rebuild.md` lưu ý lịch sử nay chỉ nói về bản LR hằng; CONTEXT.md §10 đã cắt, các chỗ còn lại
chỉ ghi "run trước (C < 100 %)" khi cần giải thích lỗi `cache_mismatch` (§3.0, §5, §9.3) và nguồn của số đo ngân sách §2.

### 16.5 Skill

`.agents/skills/kaggle-training-notebook/` (+ mirror `.claude`): thêm `scripts/pull_kernel_output.py`; `references/pull-outputs.md`
thêm mục "When `kaggle kernels output` hangs or truncates" và quy tắc "sau khi merged verify, pull là bản trùng — giữ một bản"
(kèm cách xử lý `wandb/`, `proj/`, và kéo phần nhỏ của phiên đầu để `sessions.json` đúng); `references/per-client-fl-veremi.md`
đổi lệnh pull trong chuỗi push/pull.
