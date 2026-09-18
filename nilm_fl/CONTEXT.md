# CONTEXT — Lightweight-FL NILM (học tương hỗ liên bang) trên VeReMi NextGen / DAGSNet

Bối cảnh dùng lại qua nhiều phiên. Đọc file này **trước** khi làm gì khác.
Cập nhật lần cuối: **2026-09-18 ~06:00Z** (giờ máy local = UTC+7).

**Trạng thái: CẢ BA KỊCH BẢN XONG 50/50** — `runs/merged/nilm_{20,50,100}c` verify `--require-rounds 50` pass (20c/50c 17-09;
100c 18-09 sau khi s5 `odixeuit` COMPLETE r37–r50, pull `pulls/100c_s5` PULL_OK, merge s1..s5). `report_data/` sinh lại 18-09;
**REPORT.md là bản cuối** (thân bài + phụ lục đủ 50 round, không còn nhãn interim). Việc còn lại: §8 bước 2 (xoá probe
kernels/datasets trên Kaggle — chưa làm). Tài khoản CLI/MCP active: **odixeuit**.

---

## 0. Mục lục — ở đâu có gì

| đường dẫn | nội dung | ai sửa |
|---|---|---|
| [`lightweight_fl_nilm.md`](lightweight_fl_nilm.md) | bài báo Li, Yao, Qin, Wang — *Lightweight Federated Learning for On-Device NILM* | chỉ đọc |
| [`knowledge/`](knowledge/) | **sự thật không đổi** dùng chung nhiều phương pháp (DAGSNet, dataset, máy local, Kaggle) | sửa khi *đo lại* |
| [`papers/nilm-li-2024/`](papers/nilm-li-2024/) | **phương pháp này** | |
| ├ [`paper.md`](papers/nilm-li-2024/paper.md) | trích xuất phần học tương hỗ (Eq. 17–19, Algorithm 1) + 8 chỗ bài báo để trống | |
| ├ [`rebuild.md`](papers/nilm-li-2024/rebuild.md) | **quyết định đã chốt, 13 deviation, hợp đồng artifact, ước lượng chi phí, bảng test** | |
| ├ [`proj/`](papers/nilm-li-2024/proj/) | **8 module** — nguồn duy nhất của code notebook | sửa ở đây |
| │ ├ `model.py` | DAGSNet 395.024 tham số; `build_model` dùng cho **mọi** model (w_s, w_r, w̄_r) | |
| │ ├ `nilm.py` | Eq. (17)–(19): `mutual_loss`, `client_update` (2 forward + 1 backward), `make_optimizer`, `aggregate` (1/K), `lr_at`, flat layout | |
| │ ├ `driver.py` | 2 worker/2 GPU, bảng w_s per-client + w̄_r thường trú, gate compile train+eval, eval N+1 model, commit atomic, gate ngân sách phiên | |
| │ ├ `ckpt.py` | weights (w̄_r + N w_s) / resume / marker / fingerprint 21 khoá / import qua staging / **handoff bundle** (`HANDOFF`, chuỗi sha) | |
| │ ├ `evaluate.py` | fold BN (exact), template eval compiled, `eval_model` (+ `cudagraph_mark_step_begin` mỗi batch) | |
| │ ├ `data.py`, `metrics.py` | chép nguyên từ `pfedes` | |
| │ └ `verify.py` | dựng lại mọi con số từ artifact: per-client, proxy, csv, preds, log; cây bắt đầu ở r₀ nếu là bundle | |
| ├ [`notebook/`](papers/nilm-li-2024/notebook/) | `{K}c/`, `{K}c_sN/` (phiên N), `{K}c_ckpt_probe*/` (probe CPU); mỗi thư mục đã push có file `PUSHED`; `.ipynb` **sinh tự động**, nhúng key W&B (mode 600, không commit) | ❌ không sửa tay |
| ├ `runs/pulls/{K}c_sN/` | output từng phiên đã kéo (`PULL_OK`); `runs/pulls/probe20/` = probe calibration | |
| ├ `runs/merged/nilm_{K}c/` | **cây chuẩn** ghép phiên bằng `merge_sessions.py`, verify local (20c 50 ✔, 50c 50 ✔, 100c 50 ✔) | sinh lại sau mỗi phiên |
| ├ `runs/ckpt_ds/100c_sN/` | staging handoff bundle round cuối phiên N → dataset của tài khoản chạy phiên N+1 | |
| ├ [`REPORT.md`](papers/nilm-li-2024/REPORT.md) | **báo cáo bản cuối (18-09)**: mục 1–8 + phụ lục mọi round; phụ lục sinh tự động giữa marker `APPENDIX` | thân bài sửa tay; phụ lục ❌ |
| └ [`report_data/`](papers/nilm-li-2024/report_data/) | số liệu + hình cho REPORT, sinh bởi `scripts/report_data.py` (README.md = bảng headline) | ❌ sinh lại |
| [`scripts/`](scripts/) | `gen_notebook.py`, `validate_notebooks.py`, `verify_run.py`, `run_local_checked.py` (watchdog RAM bắt buộc), `pull_output.sh` (kéo output có retry, in `PULL_OK`, log `<dest>.cli.log`), `stage_ckpt_dataset.py` (`--last-only` = handoff bundle), `gen_ckpt_probe.py` (probe CPU, in `PROBE_OK`), `report_data.py` (số liệu/hình/phụ lục REPORT), `watch_prod.py`/`poll_prod.sh` (trạng thái kernel + W&B) | |
| [`tests/`](tests/) | 7 file — bảng ở `rebuild.md` §5; `test_ckpt_verify.py` 31 check (case 7–10 = handoff bundle/merge) | |
| `.agents/skills/`, `.claude/skills/` | skill `kaggle-training-notebook` (`.agents` gốc, `.claude` mirror); `scripts/kaggle_account.py`, `kaggle_as.py`, `merge_sessions.py`, `embed_wandb_key.py` | |
| `.mcp.json`, `.vscode/mcp.json`, `.codex/config.toml` | MCP Kaggle; bearer của tài khoản CLI đang active | `kaggle_account.py use` |

Dữ liệu local: train `~/nckh/veremi/dataset/fl_client/alpha05/{20,50,100}_client/train/client_id=NNN/`,
test `~/nckh/veremi/dataset/centralized/test/`. Kaggle: `odixe0502/veremi-fl-{20,50,100}client` +
`odixe0502/veremi-nextgen2026-centralized` (public). Khuôn mẫu code: `~/nckh/veremi/pfedes`
(đã khai thác xong, không cần mở lại; bài học nằm trong skill `references/per-client-fl-veremi.md`).

**Vòng đời chuẩn:** sửa `proj/*.py` → chạy test liên quan → `python scripts/gen_notebook.py
--owner <acct> [...]` → `python scripts/validate_notebooks.py` → nhúng W&B key
(`.agents/skills/kaggle-training-notebook/scripts/embed_wandb_key.py NB --metadata META`) →
**validate lại** → push (`kaggle_as.py <acct> -- kaggle kernels push -p <dir>`) → ghi file `PUSHED`.

---

## 1. Quyết định khoa học đã chốt (chủ dự án, 2026-09-15)

| | |
|---|---|
| Mọi model | DAGSNet 395.024 tham số; w_s (cá nhân hoá, ở lại client), w_r (proxy, upload), w̄_r (server) — **cùng một init seed 42** |
| Bỏ | MNAS (§III.B) và backbone trích xuất đặc trưng; fine-tune cuối (§III.A bước 4) |
| Loss | ℓ_s, ℓ_r = CE; **ℓ(y_s,y_r) = KL trên softmax 2 chiều** (w_s ← KL(p_r‖p_s), w_r ← KL(p_s‖p_r), phía kia detach); **mẫu số 1/(ℓ_s+ℓ_r) stop-gradient** (clamp 1e-6) |
| Tổng hợp | **w̄_r = (1/K) Σ w_r^k trung bình đều** (Algorithm 1) |
| Tham gia | mọi client mỗi round |
| Optimizer | **AdamW, wd 1e-4, tạo mới mỗi client mỗi round** (một optimizer 2 nhóm tham số ≡ 2 optimizer, kiểm bit); clip 1,0 từng model; fp16 AMP |
| Learning rate | **cosine theo round 1e-3 → 1e-5, T = 50** (`proj/nilm.py::lr_at`) |
| Round × epoch | 50 × 1; batch **512 / 512 / 256** |
| Eval | mỗi round, **mọi N client** w_s^k trên đủ 10.761.343 dòng test, 10 metric/client + mean/std/min/max; **cộng w̄_r** (cột `global_*`) |
| Checkpoint | **chỉ trọng số** (state_dict w̄_r + N state_dict w_s), `weights_only=True`; không module, không optimizer; preds chỉ ở round 50 |
| W&B | project `nilm-veremi`, run id = `run_name` (`nilm_{K}c`); key nhúng theo uỷ quyền 2026-09-07 (vĩnh viễn trong version history Kaggle) |
| Đa phiên (16-09) | mỗi kịch bản là **một run** nối nhiều phiên Kaggle; hop sang tài khoản khác bằng **handoff bundle** (chỉ round cuối + `reports/handoff.json` chuỗi sha `weights/round_001..r`); cổng resume kiểm `sha(round_r) == chain[r]` và `prev_sha == chain[r-1]`; **sau phiên cuối merge thành một cây** rồi `verify_run --require-rounds 50` |

Deviation đầy đủ (13 mục, phải công bố kèm mọi con số): `rebuild.md` §2. Hai mục đáng nhớ:
(1) w_s và w_r cùng init + cùng dữ liệu ⇒ ở **round 1** chỉ khác nhau qua dropout, ℓ_d ≈ 0, chưng cất
có nghĩa từ round 2; (2) một backward cho L_s + L_r — đúng bằng hai gradient của Algorithm 1 vì
cross-term đã detach (kiểm bằng autograd).

## 2. Số đo đã có

**Local (sm_86, không phải bằng chứng cho sm_75):** train compiled ≈ 8 ms/step @256 (eager 41) trong
luồng cô lập; eval compiled-folded 380k rows/s @4096 (eager 254k); 1 worker ≡ 2 worker bit-identical;
crash replay và resume bit-identical.

**T4 (probe 20c 15-09, chi tiết `rebuild.md` §4):** bước mutual compiled **13,1 ms @512** (eager 63,5 → ×4,8); eval
compiled-folded **355k rows/s @16384**; VRAM 7,1 GiB/GPU; gate sm_75 pass (max|Δlogit| ≤ 7,3e-4, 0 flip decisive).

**Production (đo trên các phiên đã chạy, dùng để lập ngân sách):**

| | startup (prepack+compile) | round | trong đó eval | phiên 12 h ≈ |
|---|---:|---:|---:|---:|
| 20c | ~5 phút | **14,4 phút** (862 s) | 293 s | ~45 round |
| 50c | ~5 phút | **21,3 phút** (1 280 s) | ~700 s | ~31 round |
| 100c | **5,3 phút** (315 s) | **40,4 phút** (trung vị 2 425 s, max 2 659 s; round đầu phiên 2 496 s) | 1 425 s (59 %) | ~16 round |

Round-time không đổi giữa phiên thường và phiên resume từ handoff bundle. Finalize sau round cuối ≈ 100–160 s
(+ ghi preds 1,08 GB ở round 50). Skip AMP < 0,06 % bước; VRAM 7,0–7,2 GiB.

**Gate ngân sách trong driver** (`finalize_reserve_seconds=900`): sau mỗi round dừng nếu
`elapsed + 1,15 × round_max + 15 phút > max-hours`; trước round đầu dừng nếu `elapsed + 15 phút ≥ max-hours`. Nominal driver
kết thúc ≈ `max-hours − 13 phút`. Số round N chạy được trong phiên 100c cần `max-hours ≥ 315 + 2496 + (N−2)×2444 + 2870 + 900 s`
(N=2: 1,83 h; 4: 3,19 h; 14: 9,97 h). Wall ≈ `315 + 2496 + (N−1)×2444 + 150 s` (+ 1–3 phút overhead Kaggle).

**Kết quả (round 50, test toàn cục, chính thức):** 20c f1_macro w_s **0,6963** / w̄_r **0,8228**; 50c **0,6543** / **0,7923**;
100c **0,6111** / **0,7568** (acc 0,6289 / 0,7510; std client 0,050; 2/100 client < 0,50). Phát hiện (REPORT §5.1, §5.6): w_s đi lên
gần như đơn điệu suốt 50 round, tăng nhanh nhất ở 10 round cuối (+0,05 / +0,09 / +0,09) khi LR < 1e-4; ce_r/kl đạt cực tiểu
r≈27–36 rồi tăng trở lại, mạnh dần theo N (diễn giải, chưa ablation); w̄_r > mọi w_s^k trên test toàn cục. Tổng giờ round
12,2 / 17,9 / 33,8 h; giờ session 12,6 / 18,3 / 34,5 h.

## 3. Kaggle — tài khoản, quota, bẫy vận hành

**Tài khoản (6, snapshot OAuth + MCP token đủ, health pass 18-09):** odixeuit (**mới 17-09, 30 h; s5 đã dùng ~10 h**),
khanhmay0304, minhtran0601, odixe0502, minhtrit06, minhtriethihi. Quota còn (15:16Z 17-09, refresh **19-09 00:00Z**): odixeuit 30,00;
khanhmay0304 2,73 (đang chạy s4, hiển thị trừ trực tiếp); minhtran0601 1,88; odixe0502 1,50; minhtrit06 1,23; minhtriethihi 0,97 —
các mẩu < 1,9 h **không dùng nữa** (1 round 100c cần 0,85 h wall, không đáng hop).

**Chính sách (2026-09-10):** agent tự đổi tài khoản (`kaggle_account.py use/ensure --confirm`) nhưng phải nói rõ; push/kéo dưới
tài khoản khác **không cần đổi**: `kaggle_as.py <user> -- kaggle ...`. Không `kaggle auth login --force`. Tối đa 2 GPU session
đồng thời/tài khoản, **12 h/session**, `/kaggle/working` xoá khi session mới. `kernel_sources` **không** qua được tài khoản khác
(push báo thành công, `/kaggle/input` rỗng) → hop bằng dataset checkpoint.

**Bẫy đã gặp:**
- `kaggle kernels output` hay chết/treo giữa chừng → dùng `scripts/pull_output.sh` (retry, xoá file 0 byte, kill khi treo).
- `kaggle datasets create` phải có `-r zip -t` (mặc định bỏ thư mục và ghi lại CSV); stage `<out>/runs/<run_name>/…` vì Kaggle
  bỏ một cấp thư mục; mount tại `/kaggle/input/datasets/<owner>/<slug>/<run_name>/…`. Kiểm `datasets files` khớp kích thước
  (weights 100c = 166 849 317 B). Handoff 100c 144 MB zip upload ≈ 10 s; full tree 20c 988 MB ≈ 20 phút.
- Luôn chạy **probe CPU** (`gen_ckpt_probe.py`, 0 GPU, ~1 phút) và đòi `PROBE_OK nilm_100c <r>` trước khi push GPU kernel.
- **W&B không nhận hàng từ run resume** (`resume="allow"` cùng id, wandb 0.26.1 trên Kaggle mất heartbeat ở giây ~393 — chưa sửa);
  kernel vẫn chạy bình thường → theo dõi bằng trạng thái kernel; log chỉ đọc được sau COMPLETE (hoặc trên trang web Kaggle).

## 4. Cách chạy test local

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate nckh
CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/test_units.py          # 36 check, ~30 s
CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/test_smoke_real.py     # ~7 phút
CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/test_two_workers.py
CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/test_compile_gate.py
CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/test_ckpt_verify.py    # 31 check
python scripts/run_local_checked.py python tests/test_gpu_local.py                             # GPU local, Inductor
CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/test_notebook_sim.py   # thêm --gpu cho cell calibration
python scripts/validate_notebooks.py    # dir có file PUSHED = bản ghi đã push, chỉ *note* khi proj lệch
```
Một cây test một lúc; watchdog dừng thì giảm bài test, **không nới ngưỡng**. Test 2 worker/GPU local chỉ vừa watchdog khi
prepack chạy trong process con và VSCode không chiếm > 2,5 GB.

## 5. Theo dõi run, kéo output, verify

Trạng thái: `kaggle_as.py <acct> -- kaggle kernels status <owner>/<slug>`. W&B `21522798-uit/nilm-veremi` chỉ có hàng của phiên 1
(xem §3). Dấu hiệu phải dừng: backend `eager`, skip tăng dần, f1 về 0/NaN, `train_sec` +30 %, VRAM > 14 GiB, kernel ERROR.
**Đừng dừng vì f1 giảm.**

Khi COMPLETE: `bash scripts/pull_output.sh <acct> <owner>/<slug> papers/nilm-li-2024/runs/pulls/<name>` (chờ `PULL_OK`; round đạt:
`grep -o '\[r0[0-9][0-9]\][^"]*' <pull>/*.log`) → merge (§6.2) → `CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python
scripts/verify_run.py runs/merged/nilm_<K>c --require-rounds <r>` → `report_data.py` → sửa thân bài REPORT.

## 6. Đa phiên — nguồn gốc round và quy trình hop

### 6.1 Ai trả round nào (từ `runs/merged/nilm_*/logs/sessions.json`; giữ cho REPORT §7)

| kịch bản | phiên | tài khoản / kernel | resume từ | round | push | wall |
|---|---|---|---|---|---|---|
| 20c | s1 | `minhtriethihi/nilm-fl-veremi-20-clients` | — | 1–33 | 15-09 13:09Z | 8,01 h |
| | s2 | `minhtrit06/…-20-clients-s2` | full-tree dataset `minhtrit06/nilm-20c-ckpt-s1` | 34–49 | 16-09 05:30Z | 4,16 h |
| | s3 | `minhtrit06/…-20-clients-s3` | `--kernel-source` s2 | 50 | 17-09 09:05Z | 0,44 h |
| 50c | s1 | `minhtran0601/nilm-fl-veremi-50-clients` | — | 1–29 | 15-09 13:09Z | 10,60 h |
| | s2 | `minhtran0601/…-50-clients-s2` | `--kernel-source` s1 | 30–50 | 16-09 04:16Z | 7,66 h |
| 100c | s1 | `odixe0502/nilm-fl-veremi-100-clients` | — | 1–14 | 15-09 13:09Z | 9,58 h |
| | s2 | `minhtran0601/…-100-clients-s2` | handoff `minhtran0601/nilm-100c-ckpt-s1` (r14) | 15–26 | 16-09 06:15Z | 8,05 h |
| | s3 | `minhtrit06/…-100-clients-s3` | handoff `minhtrit06/nilm-100c-ckpt-s2` (r26) | 27–32 | 17-09 09:15Z | 4,14 h |
| | s4 | `khanhmay0304/…-100-clients-s4` | handoff `khanhmay0304/nilm-100c-ckpt-s3` (r32, PROBE_OK) | 33–36 | 17-09 14:25Z | 2,78 h (log), `--max-hours 3.55` |
| | s5 | `odixeuit/…-100-clients-s5` | handoff `odixeuit/nilm-100c-ckpt-s4` (r36, PROBE_OK) | 37–50 | 17-09 18:16Z | 9,85 h (log), `--max-hours 11` |

Merge: 20c 33 round chồng byte-identical, 50c 29; 100c 4 handoff (r14/r26/r32/r36) chuỗi sha liền mạch, merge 18-09 pass 50.
Probe kernels/datasets (`*-ckpt-probe`, `nilm-*-ckpt-s*`) xoá sau khi xong (giữ pull) — **chưa xoá**.

### 6.2 Quy trình một hop (phiên N vừa COMPLETE trên tài khoản A → phiên N+1 trên tài khoản B, round cuối r)

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate nckh
K=.agents/skills/kaggle-training-notebook/scripts/kaggle_as.py; P=papers/nilm-li-2024/runs; NB=papers/nilm-li-2024/notebook
python $K A -- kaggle kernels status A/nilm-fl-veremi-100-clients-sN          # phải COMPLETE
bash scripts/pull_output.sh A A/nilm-fl-veremi-100-clients-sN $P/pulls/100c_sN  # chờ PULL_OK
rm -rf $P/merged/nilm_100c && python .agents/skills/kaggle-training-notebook/scripts/merge_sessions.py --out $P/merged/nilm_100c $P/pulls/100c_s1 … $P/pulls/100c_sN
CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python scripts/verify_run.py $P/merged/nilm_100c --require-rounds r
CUDA_VISIBLE_DEVICES="" python scripts/stage_ckpt_dataset.py $P/merged/nilm_100c --owner B --slug nilm-100c-ckpt-sN --out $P/ckpt_ds/100c_sN --last-only
python $K B -- kaggle datasets create -p $P/ckpt_ds/100c_sN -r zip -t      # rồi `datasets status` = ready, `datasets files` 9 file
python scripts/gen_ckpt_probe.py --owner B --dataset B/nilm-100c-ckpt-sN --run-name nilm_100c --expect-round r --out $NB/100c_ckpt_probe_sN
python $K B -- kaggle kernels push -p $NB/100c_ckpt_probe_sN               # CPU ~1 phút; kéo log, phải có "PROBE_OK nilm_100c r"
python scripts/gen_notebook.py --owner B --clients 100 --session N+1 --require-resume --dataset-source B/nilm-100c-ckpt-sN --max-hours H
CUDA_VISIBLE_DEVICES="" python scripts/validate_notebooks.py
python .agents/skills/kaggle-training-notebook/scripts/embed_wandb_key.py $NB/100c_sN+1/nilm_100c.ipynb --metadata $NB/100c_sN+1/kernel-metadata.json
CUDA_VISIBLE_DEVICES="" python scripts/validate_notebooks.py
python $K B -- kaggle kernels push -p $NB/100c_sN+1                        # ghi file PUSHED vào 2 thư mục notebook mới; cập nhật §6.1 + đầu file
```

## 7. Caveat bắt buộc kèm mọi con số công bố

Kế thừa `knowledge/DATASET.md` §6, `ARCHITECTURE.md` §8: split theo thời gian mô phỏng; mất cân bằng
41:1 → đọc `f1_macro`; rò rỉ Sybil; scaler fit trên toàn bộ train (rò rỉ thống kê toàn cục trong FL);
fp16 lượng tử hoá đặc trưng; test không chia theo client (điểm đo **tổng quát hoá toàn cục** của model
cá nhân hoá — khác bài báo, vốn đo MAE trên test riêng từng hộ); một seed; không đặt số cạnh số của
bài báo (NILM hồi quy, REFIT/REDD). Riêng phương pháp này: bài báo là hồi quy với L2, ta dùng KL —
"lightweight FL" ở đây chỉ còn phần học tương hỗ, không có NAS nên **không** có claim về bộ nhớ/độ trễ.

## 8. Việc tiếp theo

1. ~~Hop s4 → s5~~ xong 18:16Z 17-09. ~~s5 COMPLETE → pull/merge/verify 50~~ **xong 18-09 ~05:40Z** (pull 139 file PULL_OK,
   merge s1..s5 50 round liền, `verify_run --require-rounds 50` pass). ~~report_data + viết lại REPORT~~ **xong 18-09** (bản cuối).
2. **Còn lại:** xoá probe kernels/datasets trên Kaggle (`*-ckpt-probe`, `nilm-*-ckpt-s*` ở minhtran0601/minhtrit06/khanhmay0304/odixeuit;
   dataset cây đầy đủ `minhtrit06/nilm-20c-ckpt-s1`) — giữ nguyên `runs/pulls/`. Chưa làm vì là thao tác xoá ngoài repo, cần chủ dự án xác nhận.
3. Việc mở (không bắt buộc, REPORT §7): baseline Local/FedAvg cùng cấu hình; ablation KL vs L2, stop-gradient mẫu số, LR hằng; thêm seed.
