# TEST_LOG — kết quả kiểm thử FD-IDS trên VeReMi NextGen

Mọi con số ở đây là **đo thật**, chép từ stdout của lần chạy tương ứng. Không có số nào ước
lượng. File này tách khỏi `CONTEXT.md` để `CONTEXT.md` giữ được *quyết định*, còn đây giữ
*bằng chứng*.

Quy ước bắt buộc: **"đã kiểm ở local" và "đã kiểm trên 2×T4" là hai cột khác nhau.** Cái trước
không chứng minh cái sau (`knowledge/LOCAL_ENV.md` §5).

Chạy lại toàn bộ suite local:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate nckh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
for t in test_ckpt test_fdids test_smoke_real; do
  python3 scripts/run_local_checked.py python tests/$t.py
done
python3 scripts/validate_notebooks.py
```

---

## 1. Local — 2026-09-10 (vòng 1, TRƯỚC rà soát)

Giữ nguyên làm lịch sử. Suite này pass **đồng thời** với 10 lỗi ở §2 (#9–#18):
pass ở đây chứng minh phạm vi mà nó thực sự kiểm, không hơn. Số liệu mới ở §1b.

CPU, WSL 8 GB, watchdog bật.

| # | bài kiểm | file | kết quả |
|---:|---|---|---|
| 1 | DAGSNet đúng kiến trúc | `proj/model.py` | **395.024** learnable + **3.295** buffer, khớp `ARCHITECTURE.md` |
| 2 | Checkpoint weights-only | `tests/test_ckpt.py` | **10/10 pass** |
| 3 | Toán FD-IDS | `tests/test_fdids.py` | **5/5 pass** |
| 4 | 10 metric vs sklearn | inline | `max|Δ| = 1,110 × 10⁻¹⁶` |
| 5 | End-to-end dữ liệu thật | `tests/test_smoke_real.py` | **pass**, peak RSS cây test 1.991 MiB |
| 6 | Notebook tĩnh | `scripts/validate_notebooks.py` | **pass** cho 20c/50c/100c, cả trước và sau khi nhúng W&B key |

### 1.1 Checkpoint weights-only — `tests/test_ckpt.py`

Yêu cầu của chủ dự án: lưu **trọng số**, không lưu nguyên mô hình, nhưng phải dựng lại được mô
hình tại đúng checkpoint đó.

| đo được | giá trị |
|---|---|
| **Dựng lại mô hình chỉ từ file trọng số** | **`max|Δ logits| = 0.0`** — trùng khít từng bit |
| Kích thước `weights/round_NNN.pt` | **1,65 MB** (chỉ tensor) |
| Kích thước `resume/round_NNN.pt` | **3,25 MB** (RNG, tách riêng) |
| So sánh: định dạng gộp cũ | 4,7 MB/round |
| `torch.load(weights_only=True)` | **thành công trên cả hai file** |
| Khoá trong file trọng số | đúng `{round, model, cfg, fingerprint, metrics}` — không có `optim`/`scaler` |
| Buffer BatchNorm | có mặt (`running_mean`, `running_var`, `num_batches_tracked`) |
| Sai số tham số bị bắt | `expect_params=999` → RuntimeError ✔ |
| Fingerprint lệch bị chặn | đổi `num_classes` → RuntimeError ✔ |
| Round chưa có marker | `last_complete_round` bỏ qua ✔ |
| RNG round-trip | tất định ✔ |

### 1.2 Toán FD-IDS — `tests/test_fdids.py`

| đo được | giá trị | ngưỡng |
|---|---|---|
| Proximal dạng đóng vs autograd | `max|Δ| = 4,657 × 10⁻¹⁰`, **rel = 1,721 × 10⁻⁸** | < 1e-6 ✔ |
| Round-trip flat vector ↔ state_dict | **`max|Δ logits| = 0.0`** | = 0 ✔ |
| Aggregation vs trung bình có trọng số | `max|Δ| = 1,192 × 10⁻⁷` | allclose ✔ |
| `L_soft` vs Eq. (4) viết thẳng | 0,91651726 vs 0,91651720 | < 1e-5 ✔ |
| Layout | 161 khoá float, 31 khoá int, `n_params = 395.024` | ✔ |

Bẫy đã xử lý trong bài kiểm này (theo `perf-federated.md` §3): hai đường phải **cùng mask
Dropout** (reseed) và **cùng buffer BN** (khôi phục **sau** `backward()`, vì `load_state_dict`
trong forward làm tăng version counter và autograd sẽ từ chối chạy).

### 1.3 End-to-end dữ liệu thật — `tests/test_smoke_real.py`

3 client đầu của `100_client`, mỗi client 20.000 dòng đầu; 40.000 dòng test đầu; CPU;
`compile=False`; 1 worker; 2 round.

| kiểm tra | đo được |
|---|---|
| `train/` đã chuẩn hoá sẵn | `mean(f_rcv_x_rel) = −0,611` (thô sẽ là ~1133) ✔ |
| `test/` **chưa** chuẩn hoá | `mean(f_snd_spd) = 7,6295` (full split: 7,4493) ✔ |
| fp16 an toàn | train `max|x| = 57,6`, test `355,0` (trần 65.504) ✔ |
| 2 round chạy trọn | r1 `f1_macro=0,025949`, r2 `0,027133` |
| Đủ 10 metric mỗi round | ✔ + 16 dòng per-class |
| Confusion phủ đủ | `cm.sum() == 40.000` mỗi round ✔ |
| Đẳng thức đơn nhãn | `precision_micro == f1_micro == accuracy` ✔ |
| Dựng lại trọng số round 2 | 395.024 tham số, `weights_only=True`, 1,65 MB ✔ |
| `history.csv` | đúng `[1, 2]`, không trùng dòng ✔ |
| Resume sau khi xoá marker round 2 | khởi động lại **đúng ở round 2**, `history.csv` vẫn `[1,2]` ✔ |

⚠ `f1_macro ≈ 0,026` ở đây **không có ý nghĩa khoa học** — 60.000 dòng, 2 round, khởi tạo ngẫu
nhiên. Đây là bài kiểm *pipeline*, không phải kết quả.

⚠ Round 2 chạy lại cho `f1_macro = 0,025921` thay vì `0,027133` của lần đầu. Đây là **đúng như
mong đợi**: round làm lại không bitwise-identical (`perf-federated.md` §8b). Không được mô tả
recovery là "tái lập chính xác round đã mất".

---

## 1b. Local — 2026-09-10 (vòng 2, SAU khi sửa R01–R10)

Cùng máy, cùng watchdog, cùng `CUDA_VISIBLE_DEVICES=""`. Suite được mở rộng để chạm đúng
những đường mà vòng 1 không chạm: `save_round_weights` (chứ không phải `save_round`),
`client_update` (chứ không phải công thức chép lại trong test), và import/resume có chèn lỗi.

| # | bài kiểm | kết quả | peak RSS |
|---:|---|---|---:|
| 1 | `tests/test_ckpt.py` | **14/14 pass** | 684 MiB |
| 2 | `tests/test_fdids.py` | **7/7 pass** | 759 MiB |
| 3 | `tests/test_smoke_real.py` | **pass** (1 worker + 2 worker + replay) | 2.992 MiB |
| 4 | `scripts/validate_notebooks.py` | **pass** 20c, 20c-probe, 50c, 100c, 100c-probe | 5 MiB |

⚠ RSS đỉnh của smoke test đã lên **2.992 MiB / trần 3.000 MiB** vì nay chạy thêm cấu hình
2 worker. Thêm bất kỳ bài nào nữa vào file đó sẽ chạm watchdog — tách file mới, đừng nới trần.

### 1b.1 Checkpoint và hợp đồng round — `tests/test_ckpt.py`

| đo được | giá trị |
|---|---|
| dựng lại mô hình từ trọng số | **max\|Δ logits\| = 0,0** |
| kích thước | weights **1,65 MB** / resume **0,01 MB** mỗi round |
| `weights_only=True` | cả hai file nạp được |
| trọng số NaN trong checkpoint | **bị chặn** (trước đây `argmax` biến NaN thành nhãn hợp lệ) |
| marker thiếu artifact | **không được tính** — `last_complete_round` trả round trước đó |
| khoảng trống round (1,2,3,_,5) | trả **3**, không phải 5 |
| fingerprint | **đổi** với 14 khoá khoa học, **đứng yên** với 6 khoá vận hành |
| thiếu khoá fingerprint | `KeyError`, không mặc định `None` |
| history bị cắt/xoá | `rebuild_history` dựng lại đủ 3 round từ metrics json |
| import bị ngắt ở round 3 | chỉ **2** được tin; retry hoàn tất **3** kèm history |
| nguồn import khác `lr` | import **0 round**; cfg đúng thì resume bình thường |

### 1b.2 Toán FD-IDS — `tests/test_fdids.py`

| đo được | giá trị |
|---|---|
| proximal dạng đóng vs autograd | `max\|Δ\| = 4,657 × 10⁻¹⁰`, rel `1,721 × 10⁻⁸` |
| round-trip flat ↔ state_dict | `max\|Δ logits\| = 0,0` |
| aggregation vs trung bình có trọng số | `max\|Δ\| = 1,192 × 10⁻⁷` |
| `L_soft` vs Eq. (4) viết trực tiếp | 0,91651726 vs 0,91651720 |
| **`client_update` trọn vẹn vs Eq. (6) qua autograd** | 700 dòng / B=256 / **E=1** → `max\|Δw\|/\|w\| = 4,606 × 10⁻⁷` |
| | 517 dòng / B=128 / **E=2** → `1,568 × 10⁻⁷` |
| coverage mỗi epoch | mỗi chỉ số xuất hiện **đúng một lần**, kiểm theo từng index |
| batch dư | tail 188 và tail **5** đều chạy, `skipped=0 nonfinite=0` |
| `check_updates` | chặn cả 4 ca: `applied=0`, gradient non-finite, trọng số NaN, thiếu client |

### 1b.3 End-to-end — `tests/test_smoke_real.py`

3 client cắt ngắn của 100c, 20.000 dòng/client, 40.000 dòng test, CPU, `compile=False`.

| đo được | giá trị |
|---|---|
| round 1 / round 2 f1_macro | 0,028752 / 0,030403 |
| **1 worker vs 2 worker, round 1** | **`max\|Δ trọng số\| = 0`** |
| **replay round 2 sau khi xoá marker** | **bit-identical**, f1_macro 0,030403 (vòng 1 đo 0,027133 → 0,025921) |
| artifact mỗi round | weights + resume + preds + logs + metrics + confusion + marker |
| `verify_run` | metrics == confusion == weights == history; **preds dựng lại đúng từng ô của CM** |
| history sau replay | vẫn đúng `[1, 2]`, không nhân đôi |

Hai dòng in đậm là bằng chứng trực tiếp cho R04: trước khi sửa, cùng dữ liệu cùng seed cho ra
trọng số khác nhau tuỳ lịch phân client; nay không còn.

### 1b.4 Kiểm rủi ro tail batch — đã kiểm, **không có rủi ro**

BatchNorm ở train mode hỏng khi mỗi channel chỉ có 1 giá trị, nên một client có `n mod B == 1`
sẽ làm sập cả round. Đếm trên dữ liệu thật:

| cấu hình | client | dòng nhỏ nhất..lớn nhất | tail nhỏ nhất | tail == 1 | tail == 0 |
|---|---:|---|---:|---|---|
| 20c, B=512 | 20 | 870.217 .. 5.890.990 | 29 | **không có** | 0 |
| 50c, B=512 | 50 | 199.063 .. 2.630.929 | 29 | **không có** | 0 |
| 100c, B=256 | 100 | 98.180 .. 1.333.839 | 2 | **không có** | 0 |

Cả ba đều tổng đúng **43.045.415** dòng. Ngoài ra DAGSNet forward train mode ở batch 1 và 2
đều chạy (BatchNorm1d còn 11 giá trị/channel sau khi transpose), nên đây là rủi ro kép đã loại.

---

## 1c. Local — 2026-09-10 (vòng 3, sau khi sửa R11–R18)

Cùng máy, cùng watchdog, `CUDA_VISIBLE_DEVICES=""`, mỗi cây test chạy tuần tự.
Suite tách thành **11 file** theo đúng yêu cầu §12: mỗi lỗi được tái hiện ở §11 nay có một
file riêng gọi đường production, thay vì phình `test_smoke_real.py` sát trần RAM.

| file | phạm vi | kết quả | peak RSS |
|---|---|---|---:|
| `test_ckpt.py` | checkpoint, fingerprint, marker, history | **14/14** | 685 MiB |
| `test_fdids.py` | Eq. (2)–(6), `client_update` vs autograd, `check_updates` | **7/7** | 759 MiB |
| **`test_amp_guard.py`** | R11 — overflow AMP vs divergence | **8/8** | 698 MiB |
| **`test_resume_import.py`** | R12 — marker nguồn, chuỗi hash, gap, history | **7/7** | 533 MiB |
| **`test_verifier.py`** | R13 — 12 ca tamper phải fail | **13/13** | 520 MiB |
| **`test_data_cache.py`** | R14 — `load_clients`/`load_test` thật + `cache_ok` | **8/8** | 4 MiB |
| **`test_teacher.py`** | R16 — teacher cache vs online | **3/3** | 718 MiB |
| **`test_budget.py`** | R17 — đồng hồ ngân sách, resume | **5/5** | 1.249 MiB |
| **`test_schedule.py`** | R04/R16 — 1 worker vs 2 worker | **3/3** | 1.996 MiB |
| `test_smoke_real.py` | end-to-end trên dòng VeReMi thật | **pass** | ~2.100 MiB |
| **`test_validator.py`** | R18 — 7 mutation phải bị chặn | **8/8** | 514 MiB |

### 1c.1 R11 — AMP overflow bị coi là divergence

Tái hiện bằng `client_update` thật + `GradScaler('cpu')`, chèn Inf vào backward đầu tiên:

| | trước | sau |
|---|---|---|
| `steps/skipped/applied` | 2 / 1 / 1 | 2 / 1 / 1 |
| `nonfinite` | **1** | **0** |
| trọng số cuối | hữu hạn | hữu hạn |
| `check_updates` | **REJECTED** — dừng cả round | **ACCEPTED** |

Đây là lỗi **giết run thật**: mỗi client khởi tạo GradScaler mới ở `2**16`, nên vài bước đầu
của gần như mọi client đều overflow. Round 1 trên T4 sẽ dừng ngay. Ngân sách skip warm-up
chốt trước ở **16**; đo được biên: 16 nhận, 17 từ chối.

Ca ngược lại vẫn bị chặn: khi scaler **không** skip mà gradient vẫn không hữu hạn (proximal
cộng sau `unscale_`), giá trị đó *có* vào trọng số — và `clip_grad_norm_` biến `Inf` thành
`clip_coef = 0`, rồi `Inf * 0 = NaN`, nên phép kiểm trọng số bắt trước. Bộ đếm `nonfinite`
là chẩn đoán, không phải hàng rào duy nhất.

### 1c.2 R12 — import nối hai lịch sử

Mỗi file trọng số nay ghi `prev_sha` = hash của **file** round trước. Hai run cùng cấu hình
có cùng fingerprint nhưng khác byte, nên chuỗi gãy đúng chỗ.

| ca | kết quả |
|---|---|
| nguồn có artifact round 1–2 nhưng chỉ marker round 1 | import **đúng 1**, không tự tạo marker 2 |
| working = A.r1, mount = B.r1–2 (cùng fingerprint, khác trọng số) | **RuntimeError**, nêu cả hai đường dẫn |
| cùng mount đó vào cây rỗng | import cả 2 round |
| round 2 bị thay bằng file từ nơi khác | chuỗi gãy → `last_complete = 1` |
| gap ở round 2 | `last_complete = 1`, history chỉ giữ round 1 |
| không còn round hợp lệ | `history.csv` **bị xoá**, không để lại |
| `resume/round_001.pt` ghi đè bằng rác | round không còn được tính |

### 1c.3 R13 — 12 ca tamper, tất cả từng **pass**

Fixture 2 round hợp lệ vẫn pass; mỗi bản sao sửa một chỗ đều phải fail:
xoá dòng history · nhân đôi dòng history · `accuracy = NaN` · per-class `F1 = 999` (support
để đúng) · xoá toàn bộ `preds/` · xoá toàn bộ `logs/` · ghi rác vào `resume/` · tráo
predictions · log client khai sai số bước · thiếu manifest · `cfg.n_test = 999` · thiếu 1/2
round dưới `require_rounds`.

Hai ca tinh vi nhất: **NaN** lọt qua mọi phép so viết dạng `abs(a-b) > tol` (so sánh với NaN
luôn False) — nay `isfinite` kiểm **trước**; và per-class bịa được toàn bộ mà `support` vẫn
đúng, vì support lấy thẳng từ tổng hàng — nay recompute đủ `precision/recall/f1`.

### 1c.4 R14 — hai loader production trên parquet thật

| đo được | kết quả |
|---|---|
| `find_root` hai sentinel | đúng root; **hai root khớp → từ chối**, không đoán |
| `load_clients` | `(100, 66)` fp16, spans liên tục, train **không** bị scale lại |
| `load_test` | z-score đúng một lần: `(x − mean)/std_used`, mean `−0.019` |
| thứ tự cột parquet đảo | vẫn lấy đúng theo **tên**, không theo vị trí |
| nhãn `−1` và `16` | **cùng bị chặn** trước khi ép `uint8` (nếu không: 255 và 0) |
| `cache_ok` | nhận cache đủ; từ chối cả 8 ca hỏng (thiếu file, cắt cụt, rác, X/y lệch dòng, sai dtype, spans hở, manifest khác scenario, thiếu manifest) |

### 1c.5 R16 — teacher cache **không** exact, và con số thật là bao nhiêu

Docstring cũ viết "Exact". Đo được thì không:

| | đo được |
|---|---|
| cache teacher, batch 16 … 4096 | `max\|ΔZt\| = 1,221 × 10⁻⁴` — **khác 0** |
| cache fp16 vs teacher online fp32 | `max\|ΔZt\| = 2,198 × 10⁻⁴` |
| ảnh hưởng lên KD loss | `\|Δloss\| = 0` (làm tròn về cùng float32) |
| ảnh hưởng lên gradient student | `rel\|Δgrad\| = 1,090 × 10⁻⁵` |
| teacher/anchor trong lúc client update | **không đổi**; student dịch 2,0728 |

Docstring đã sửa để ghi đúng ngưỡng thay vì khẳng định bằng 0.

### 1c.6 R04/R16 — bất biến theo lịch, đo lại ở quy mô lớn hơn

4 client kích thước lệch nhau (700/300/450/180) để LPT thật sự đảo thứ tự:

| | kết quả |
|---|---|
| round 1 và 2, 1 worker vs 2 worker | `max\|Δ trọng số\| = 0.0` |
| mọi metric và số bước | trùng khít |
| chạy lại cùng cấu hình 2 worker lần nữa | `max\|Δ trọng số\| = 0.0` |

### 1c.7 R17 — ngân sách phiên

| ca | kết quả |
|---|---|
| startup đã hết budget trước round 1 | **0 round**, không commit gì |
| chạy bình thường 2 round | `seconds` bao trọn commit; metrics json đủ cột dựng lại history |
| budget chỉ đủ 1 của 5 round | dừng sau round 1, round đã commit còn nguyên vẹn |
| resume | tiếp đúng round kế, history không lặp |
| đã xong hết | trả về sạch, 0 round |

### 1c.8 R18 — 7 mutation của validator

Baseline pass; cả 7 bị chặn: `CFG['lr'] = 0.25` sau khai báo · marker cache ghi trước
`np.save` · tráo dataset centralized · module nhúng lệch nguồn · tên chưa được cell nào bind ·
`rounds` đổi thành 5 · tắt `is_private`.

---

## 2. Lỗi phát hiện trong quá trình kiểm thử

Ghi lại vì mỗi lỗi đều **im lặng** — code vẫn chạy, vẫn ra 16 logit.

| # | lỗi | phát hiện bởi | sửa |
|---:|---|---|---|
| 1 | `metrics_from_confusion` trả `np.float64` → `torch.load(weights_only=True)` **từ chối cả file** | test_smoke_real | ép `float()` mọi giá trị |
| 2 | `find_root` chỉ bóc **một** cấp cha, sentinel `train/client_id=000` trả về `<root>/train` | kiểm tra trên cây thật | bóc đúng `len(Path(sentinel).parts)` cấp |
| 3 | Embedder W&B để lại `except` mồ côi → cell không parse | `validate_notebooks.py` | đổi hình dạng cell: `del _wandb_key` **sau** `except` |
| 4 | `mp` start method `spawn` re-import module test → worker chạy lại cả bài test | test_smoke_real | thêm `if __name__ == "__main__"` |
| 5 | `np.ascontiguousarray` cả mảng mmap → 5,7 GB host RAM/worker + tensor read-only | cảnh báo torch | copy theo chunk 4 M dòng |
| 6 | Assert "train đã z-score" dùng `max|mean|` trên 20k dòng đầu | test_smoke_real | 20k dòng đầu bị tương quan theo thời gian; đổi sang so sánh **độ lớn** trên `f_rcv_x_rel` |
| 7 | `cfg["local_epochs"]` có trong CFG nhưng vòng lặp hardcode 1 epoch — đặt E=2 sẽ bị **âm thầm bỏ qua** | rà lại CFG | cài đúng vòng `for _e in range(E)` theo Algorithm 1 dòng 15 |
| 8 | Worker chết chỉ để lại exit code; traceback thật nằm ở **tiến trình con**, parent không thấy | lỗi #7 làm lộ ra | worker bọc try/except, đẩy traceback qua queue, parent raise kèm nội dung |
| 9 | **R01** — `applied = max(1, nsteps - skips)` khiến điều kiện `applied == 0` **không bao giờ đúng**; client bị skip toàn bộ bước AMP vẫn được gộp | rà soát 2026-09-10 + stub | bỏ `max()`; tách `check_updates()` thành hàm thuần và test riêng 4 ca |
| 10 | **R02** — import checkpoint copy theo thứ tự `SUBDIRS`, `complete` đứng **thứ 3/8** → marker xuất hiện trước metrics/confusion; `history.csv` ở root **không nằm trong danh sách copy** | rà soát + chèn ngắt | copy qua staging, kiểm đủ artifact, **publish từng round rồi mới ghi marker**; history dựng lại từ metrics json |
| 11 | **R03** — gate compile lấy `ref` **trước** 3 bước warmup và `got` **sau**, tức so hai trạng thái BatchNorm khác nhau. Compiler identity vẫn báo lệch **0,214743** và 62 buffer đổi → loại nhầm đường compile (2,86×) **mọi lần** | rà soát + compiler identity | snapshot/restore weights+BN+RNG quanh probe; so cả logits **và gradient** từ cùng một trạng thái, trên **dữ liệu thật** |
| 12 | **R04** — Dropout đọc RNG của worker seed theo **rank**, nên kết quả phụ thuộc lịch phân client; replay round 2 cho f1 `0,027133 → 0,025921` | log cũ + chạy lại | seed `torch.manual_seed(seed, round, client)` ngay trước mỗi client. Đo lại: 1 worker vs 2 worker `Δ = 0`, replay **bit-identical** |
| 13 | **R05** — fingerprint chỉ hash **5** khoá, trong đó `batch_per_gpu`/`total_rounds` là **bí danh mà vòng train không hề đọc**. Mù với `lr`, `mu`, `beta`, `lam`, `T`, `seed`, `local_epochs`, `dropout`, `n_clients` và toàn bộ preprocessing | rà soát, sửa từng khoá | `FINGERPRINT_KEYS` 21 khoá + `data_id` (feature order + class order + scaler); bỏ hẳn hai bí danh; thiếu khoá là `KeyError` |
| 14 | **R06** — `append_history` mở CSV bằng `"w"`; crash giữa lúc ghi lại làm mất các dòng đã commit. Metrics json thiếu cột để dựng lại | đọc code | ghi tmp rồi `os.replace`; metrics json chứa **đủ** row; `rebuild_history()` dựng lại từ artifact |
| 15 | **R07** — prepack bỏ qua **toàn bộ** decode chỉ vì `train_X.f16.npy` tồn tại: crash sau file đầu làm mọi phiên sau train trên cache thiếu | đọc code | `manifest.json` ghi **sau cùng**, khoá theo `data_id`/`n_clients`/đường mount; không khớp thì xoá sạch và làm lại |
| 16 | **R08** — `preds/` được tạo nhưng **không bao giờ ghi**; eval gọi `argmax` mà không kiểm logits hữu hạn (`argmax` của một hàng NaN trả về 0 — một nhãn hoàn toàn bình thường) | đọc code | ghi `preds/round_NNN.u8.npy` mỗi round + đếm non-finite trên device; thêm `proj/verify.py` nối weights → CM → json → csv |
| 17 | **R09** — đồng hồ ngân sách bắt đầu **sau** spawn/compile, và `seconds` kết thúc **trước** checkpoint/CSV/W&B | đọc code | `t_origin=T0` lấy từ cell đầu; `seconds` bao trọn commit; ghi peak VRAM từng client |
| 18 | **R10** — cell tổng kết kiểm marker liên tục nhưng **không** kiểm `N == 50`; `best = max(test f1)` được in như kết quả chính | đọc code | báo round cuối là kết quả, best-test có nhãn *descriptive only*; `verify_run(require_rounds=)`; in luôn ước lượng 50 round |
| 19 | **Lỗi trong chính bản sửa R02**: `round_ok` không kiểm marker, nên retry sau khi crash **giữa lúc copy và lúc đánh dấu** thấy file đã đủ rồi bỏ qua luôn bước đánh dấu → round kẹt vĩnh viễn không marker | `tests/test_ckpt.py` #13 (ca chèn lỗi vừa viết) | tách điều kiện copy và điều kiện mark thành hai lệnh `if` độc lập |
| 20 | Cổng kiểm tĩnh so thứ tự trên **toàn bộ** notebook, nên `C.resolve_resume` nằm trong thân `driver.py` nhúng đã thoả mãn một khẳng định về **luồng thực thi** | chính cổng đó fail khi thêm check `data_id` | mọi check thứ tự chuyển sang `flow` = chỉ các cell **thực thi**, bỏ cell `%%writefile` |
| 21 | **R11** — `nonfin += ~isfinite(gn)` chạy **trước** khi biết scaler có skip bước hay không. Overflow AMP bình thường (scaler skip đúng, trọng số hữu hạn) bị `check_updates` từ chối → **dừng cả round**. Với GradScaler mới mỗi client ở `2**16`, lỗi này nổ ngay round 1 trên T4 | rà soát lần 3 + `client_update` thật | chỉ đếm non-finite trên bước scaler **đã áp dụng**; thêm ngân sách skip warm-up chốt trước (16); kiểm `applied+skipped == steps` |
| 22 | **R12a** — `_import_from` không đọc marker của **nguồn**: nguồn crash sau khi ghi artifact round 2 nhưng trước marker → import vẫn nhận 2 và **tự tạo marker** | rà soát lần 3 | copy cả `complete/` vào staging; dải import bị chặn bởi marker nguồn |
| 23 | **R12b** — hai run cùng cấu hình có cùng fingerprint, nên import giữ round 1 của A rồi lấy round 2 của B thành một "lịch sử" | rà soát lần 3 | mỗi file trọng số ghi `prev_sha` = hash **file** round trước; byte khác → chuỗi gãy; overlap khác byte → `RuntimeError` nêu cả hai đường dẫn |
| 24 | **R13** — verifier báo pass ở 12 ca tamper. Hai ca sâu nhất: `NaN` lọt qua mọi phép `abs(a−b) > tol`, và per-class bịa được toàn bộ mà `support` vẫn đúng | rà soát lần 3, mỗi ca một bản sao | `isfinite` kiểm **trước** phép so; recompute đủ per-class; bắt buộc `preds`/`logs`/manifest ở chế độ full; CSV phải đúng `1..last` không trùng; `round_ok` **đọc** file resume |
| 25 | **R14a** — `cache_ok` cũ chỉ so manifest: manifest khớp + thiếu `train_X` vẫn in "cache reusable", worker chết sau đó | rà soát lần 3, chạy chính nhánh hit | chuyển vào `proj/data.py::cache_ok`, kiểm đủ file + shape + dtype + X/y cùng số dòng + spans liên tục |
| 26 | **R14b** — `data_id` không nhìn thấy nội dung dữ liệu: đổi partition/nhãn mà giữ features/classes/scaler thì fingerprint không đổi | rà soát lần 3 | thêm `content_id` (row count từng client + histogram lớp) tính **sau** decode; `write_manifest` là cổng thứ hai, raise nếu lệch |
| 27 | **R15** — `y_true` chỉ nằm ở `/kaggle/temp`, mất theo phiên; scaler không có trong manifest; `preds`/`logs` không được import | rà soát lần 3 | `y_true` copy vào `reports/`; scaler ghi vào manifest; `RESUME_SUBDIRS` thêm `preds`, `logs` |
| 28 | **R16a** — gate compile chỉ chạy CE với nhãn ngẫu nhiên, bỏ KD và proximal (hai phần ba Eq. 6) | rà soát lần 3 | probe chạy đúng loss production: CE + KD + proximal + clip |
| 29 | **R16b** — `n_bad == 0` trên tập decisive **rỗng** vẫn đạt gate | rà soát lần 3 | in số lật top-1 trên **cả** batch; từ chối chứng nhận nếu tập decisive < 10% số dòng |
| 30 | **R16c** — hai worker gate độc lập, có thể chạy khác backend trong cùng một round | rà soát lần 3 | worker báo backend lúc ready; lệch nhau → driver ép cả hai về eager và ghi `cfg["backend"]` |
| 31 | **R16d** — docstring `teacher_logits` viết "Exact" | `tests/test_teacher.py` (bài mới viết) | đo được `max\|ΔZt\| = 1,2×10⁻⁴` giữa các batch; sửa docstring ghi đúng ngưỡng đã đo |
| 32 | **R17a** — `seconds` chốt trước metrics JSON/CSV/marker/W&B; budget chỉ kiểm **sau** round nên phiên hết giờ vẫn bắt đầu round mới | rà soát lần 3 | `time.monotonic()`; W&B log vào **trong** vùng đo; kiểm budget **trước** round đầu; `finalize_reserve_seconds` |
| 33 | **R17b** — continuation tính `overhead = (now − T0) − sum(sec)` trên **toàn bộ** history, cho overhead **âm** và projection ~1 phút cho 50 round | rà soát lần 3 (phản ví dụ) | projection tính từ `hist` của **phiên hiện tại**; <2 round thì báo không đủ số đo, không in projection |
| 34 | **R17c** — `vram_gb` là high-water mark từ đầu worker; `train_ce`/`train_kd` là trung bình **đều giữa client**, dễ đọc nhầm thành loss trung bình theo mẫu | rà soát lần 3 | `reset_peak_memory_stats` mỗi client và mỗi eval, tách `vram_train_gb`/`vram_eval_gb`; đổi tên thành `ce_client_mean`/`kd_client_mean` |
| 35 | **R18** — validator chỉ đọc khai báo `CFG` đầu tiên (`CFG['lr']=0.25` sau đó vẫn exit 0); kiểm thứ tự marker bằng `flow.index('test_y.u8.npy')` khớp phải tên file trong tuple `FILES`; dataset chỉ kiểm substring | rà soát lần 3, 3 mutation | chặn mọi ghi vào `CFG` ngoài whitelist; kiểm thứ tự marker bằng **AST** (`MF.write_text` sau mọi `np.save`); so **chính xác** cả hai dataset slug |
| 36 | **Sidecar `.stats.json`** — thư mục test centralized có `part-NNNNN.stats.json` cạnh mỗi part, và `ds.dataset(dir, format="parquet")` mở **mọi** file → `ArrowInvalid: Parquet magic bytes not found in footer`. Chết sau khi đã decode xong 43 M dòng train (~9 phút), trên **cả ba** run | **lần push đầu lên T4** | liệt kê tường minh `sorted(root.rglob("*.parquet"))`; thứ tự cố định cũng chốt luôn row order của test, tức của `y_true` và mọi vector prediction |
| 37 | `wandb_run.log(row)` gọi **trước** `row["seconds"] = …`, nên W&B nhận `seconds: 0.0` mọi round. Artifact (`metrics/*.json`, `history.csv`) và projection đều **đúng** vì lấy giá trị đặt sau — chỉ mất khả năng theo dõi thời gian round **trực tiếp** trên W&B | round 1 thật trên T4 | đặt `seconds` **trước** lời gọi W&B. Budget không phụ thuộc field này (nó so elapsed tuyệt đối của phiên), nên thứ tự này không làm hỏng phép tính ngân sách. **Không** push lại để sửa: restart cả ba run mất ~40 phút prepack + round 1 và reserve lại quota, cho một field chỉ dùng để nhìn |
| 38 | `backend` bị loại khỏi log W&B (`if k not in ("round", "backend")`), lại được quyết định **sau** `wandb.init(config=CFG)` và **sau** `write_manifest` ⇒ không có cách nào biết compile pass hay rơi eager khi run còn sống. Chỉ đọc được từ `weights/*.pt` sau khi run kết thúc | phân tích thông lượng round 1–2 trên T4 | `wandb_run.config.update({"backend": …}, allow_val_change=True)` + `summary["backend"]` ngay sau khi worker ready |
| 39 | **Gate compile không thể chứng nhận được** — `_compile` so một bước train của model eager với model compiled, `restore()` đặt lại RNG của torch nên eager lặp đúng mask dropout cũ, nhưng Inductor **functionalise RNG** và tự rút offset Philox riêng ⇒ mask dropout của hai bên không bao giờ trùng. Gate đo chênh lệch giữa **hai mask dropout** rồi gọi đó là lỗi compiler: `max|dlogit|` = **6,07e-01** ở p=0,1 so với **7,32e-04** ở p=0. `10*dz = 6,07` lớn hơn mọi margin ⇒ `n_dec = 0` ⇒ `cannot certify` ⇒ **rơi eager**. Cả ba run trên T4 đều đang chạy eager và trả giá ~2× | probe compile chạy **local trên sm_86** — nơi Triton không hề bị nghi ngờ | tắt dropout (`m.p = 0`) **cho cả hai bên** chỉ trong lúc so sánh, khôi phục trong `finally` và cả ở nhánh `except`. Warm-up phía trên đã capture graph ở p production nên production dùng lại đúng entry đó. `tests/test_compile_gate.py` (6 phép kiểm) |
| 40 | `torch.cuda.clock_rate` **qua được `hasattr`** nhưng ném `ModuleNotFoundError: nvidia-ml-py` khi gọi. Đặt ở cuối worker nên nó sẽ giết probe **sau khi** đã đo xong, mất toàn bộ JSON kết quả | chạy thử probe ở local trước khi tốn quota | bỏ hẳn lời gọi — cell 1 đã in clock bằng `nvidia-smi`. Cùng loại với #36: `hasattr` không phải là kiểm tra gọi được |
| 41 | `tests/test_validator.py` **hardcode** `fdids_20c.ipynb` làm notebook bị đột biến, trong khi validator đã đổi sang đọc `code_file` từ `kernel-metadata.json` (§14.4). Từ lần relaunch `--run-tag _v2`, mọi ca mutation chết `FileNotFoundError` — suite "86 phép kiểm pass" ghi ở CONTEXT là **cũ**, file này chưa chạy lại sau relaunch | chạy lại suite trước khi thêm mutation mới (2026-09-11) | lấy tên từ `kernel-metadata.json` y như validator. Thêm 2 mutation cho quy tắc attach mới: dataset checkpoint gắn vào run fresh, và `require_resume=True` mà không gắn gì — **10/10** (1 baseline + 9 mutation) |
| 42 | (skill helper) `kaggle_account.py use` kiểm danh tính bằng `"username: X" in [line.strip()]`, nhưng CLI 2.2.4 in `- username: X` ⇒ switch **thành công** vẫn bị báo "refresh token may have been revoked" và **exit 1** — với quy tắc "never launch after a nonzero result" thì đây là false negative chặn launch | `use minhtriethihi --confirm` lần đầu, 2026-09-11 | `lstrip("- ")` trước khi so; test `test_config_view_bulleted_output_of_cli_2_2_is_accepted`; cả hai cây |

Lỗi #1 chính là cái bẫy vừa được ghi vào skill (`weights_only=True` chỉ đúng khi file chỉ chứa
tensor và kiểu thuần) — và nó bị bắt ngay ở lần chạy đầu tiên.

Lỗi #3 và #7 đều là **im lặng theo hướng nguy hiểm nhất**: notebook vẫn chạy, vẫn ra kết quả,
chỉ là sai tham số. Cả hai bị bắt bởi cổng kiểm tĩnh và rà CFG, không phải bởi bài test số học —
đó là lý do `validate_notebooks.py` phải chạy **cả sau khi nhúng W&B key**, không chỉ trước.

**Bài học từ #9–#18:** cả 10 lỗi đều tồn tại trong lúc suite vòng 1 pass **toàn bộ**. Chúng
không bị bắt vì test kiểm *công thức chép lại trong test* chứ không kiểm *hàm production*
(`test_fdids` không gọi `client_update`), và kiểm *đường generic* chứ không kiểm *đường
FD-IDS thật* (`test_ckpt` gọi `save_round`, driver gọi `save_round_weights`). Đó là lý do
`save_round`/`load_for_resume` nay đã **bị xoá**: hai đường song song thì test luôn có nguy cơ
bám vào đường không chạy.

**Bài học từ #19 và #20:** cả hai là lỗi *mới sinh ra trong lúc sửa*, và cả hai bị bắt bởi
chính bài test/cổng viết cho lần sửa đó. Một bản sửa không kèm ca chèn lỗi chỉ là một giả
thuyết.

**Bài học từ #36 — vì sao 76 phép kiểm local không bắt được:** `test_smoke_real` **có** đọc
đúng thư mục test thật đó, nhưng qua `to_batches()` rồi `break` khi đủ 40.000 dòng — nó lấy
xong từ part đầu tiên và **không bao giờ chạm** tới sidecar. `to_table()` trên Kaggle đọc hết.
Một reader lười che đúng lớp lỗi này. Fixture parquet trong `test_data_cache.py` cũng chỉ có
file `.parquet` vì chính tay tôi tạo ra nó. Bài học: fixture do mình dựng chỉ kiểm được điều
mình đã nghĩ tới; phải soi **thư mục thật** xem nó chứa gì. Nay `test_data_cache.py` làm cả
hai: chèn sidecar vào fixture, **và** đếm file trong thư mục thật (8 file → 4 part).

**Bài học từ #21–#35:** vòng 2 sửa đúng những gì vòng 1 tìm ra, và **tạo ra một lỗi nghiêm
trọng hơn tất cả**: #21 sẽ dừng round 1 của cả ba kịch bản trên T4 — sau khi đã trả tiền cho
prepack, compile và một round train đầy đủ. Nó không thể bị bắt ở local vì `GradScaler` bị
tắt trên CPU: `prev is None` nên `applied` luôn True và không bước nào bị skip. Chỉ khi
**cố tình** dựng một scaler CPU và chèn overflow thì đường đó mới chạy. Một hàng rào chưa
từng chạy không phải là một hàng rào.

---

## 3. Kaggle — 2×T4

### 3.1 Lần push đầu — 2026-09-10, **cả ba FAIL**, một lỗi duy nhất

| kernel | tài khoản | version | kết quả |
|---|---|---:|---|
| `khanhmay0304/fd-ids-veremi-100-clients` | khanhmay0304 | 1 | **ERROR** |
| `khanhmay0304/fd-ids-veremi-50-clients` | khanhmay0304 | 1 | **ERROR** |
| `minhtrit06/fd-ids-veremi-20-clients` | minhtrit06 | 1 | **ERROR** |

Tất cả dừng ở cùng một chỗ — lỗi #36 ở §2:

```
ArrowInvalid: Could not open Parquet input source
'.../veremi-nextgen2026-centralized/upload/test/part-00001.stats.json':
Parquet magic bytes not found in footer.
```

Tổng chi phí: **0,08 h quota**. Rẻ vì nó chết ở prepack, không phải giữa round 40.

**Nhưng phần chạy được trước đó là số đo T4 đầu tiên của dự án, và đều đúng:**

| đo được trên T4 | giá trị |
|---|---|
| GPU | `0 Tesla T4 (7, 5) 14.6 GiB` · `1 Tesla T4 (7, 5) 14.6 GiB` — đúng 2×T4 sm_75 |
| runtime | torch **2.10.0+cu128**, python 3.12.13 (khác hẳn local 2.13.0+cu130) |
| CFG hiệu lực | in đủ 22 khoá, khớp bảng §1 của `CONTEXT.md` |
| module nhúng | 7/7 ghi ra `/kaggle/working/proj/` |
| mount FL | `/kaggle/input/datasets/odixe0502/veremi-fl-20client/20_client` — `find_root` giải đúng |
| **prepack train** | **`(43045415, 66)` `max\|x\|=570.5`** |
| `data_id` / `fingerprint` (20c) | `29f492a531052d2b` / `569ba49f98b498b2` |
| resume gate | `resume from round None` — đúng cho lần chạy đầu |

`max|x| = 570.5` khớp `knowledge/DATASET.md` (570,44) và xác nhận fp16 an toàn trên **toàn bộ**
43 M dòng, không phải trên mẫu. Đây là điều local **không** kiểm được (§5 `LOCAL_ENV.md`).

**Chưa đo được** vì chết trước khi tới đó: gate compile trên sm_75, backend hai rank,
thời gian round, peak VRAM, và do đó ngân sách thật.

### 3.2 Lần push thứ hai — sau khi sửa #36, **số đo T4 đầu tiên**

| run | r1 f1 / acc | r2 f1 / acc | ce r1→r2 | skip / steps | **round** | VRAM train/eval | teacher |
|---|---|---|---|---|---:|---|---:|
| 20c | 0,192371 / 0,434955 | 0,577886 / 0,773878 | 0,6490 → 0,5865 | 5, 6 / 84.083 | **17,9 ph** | 7,25 / 7,09 GiB | 221 s |
| 50c | 0,223400 / 0,479411 | 0,547154 / 0,752569 | 0,6503 → 0,5615 | 0, 0 / 84.098 | **18,0 ph** | 7,16 / 7,09 GiB | 230 s |
| 100c | chưa xong round 1 | | | | | | |

`skip` = 0–6 trên 84.083 bước, trần warm-up chốt trước là 16 → **bản sửa R11 được xác nhận
trên phần cứng thật**: code trước bản sửa sẽ đếm 5 bước đó là `nonfinite` và giết round 1.

**Benchmark local để truy nguyên 18 phút/round** (sm_86, batch 512, đủ AMP+KD+clip+Adam):

| đường | ms/step | 42.041 step/GPU |
|---|---:|---:|
| không proximal | 14,84 | 10,4 ph |
| proximal `cat`+vòng `add_` (đang dùng) | 16,88 | 11,8 ph |
| proximal `torch._foreach_*` | **15,16** | 10,6 ph |
| thêm gather `X[idx]` trên tensor thường trú | 19,07 | **13,4 ph** |

`cat_loop` vs `foreach`: `max|Δ| = 0.000e+00` — bit-identical, rẻ hơn ~10%. Đã áp dụng vào
`client_update`, kèm `tests/test_proximal.py` (4 phép kiểm, `max|Δ| = 0` trên 99 tham số).

⚠ **Kết luận "1,34×" ở bản ghi trước là SAI — đã sửa 2026-09-11.** Nó so một dự đoán
**thời gian/step thuần** (13,4 ph) với **wall-clock cả round** (18,0 ph), mà cả round còn
chứa teacher 221 s và eval. Trừ đúng hai phần đó: `(1074 − 221 − ~40)/42.041 ≈ 19,3 ms/step`
— **khớp gần như hoàn hảo với eager-local 19,07**, không hề có hệ số 1,34. Kết luận
"compile đã rơi eager" thì **đúng**, nhưng lập luận dẫn tới nó thì không; xem §3.3 cho
nguyên nhân thật (lỗi #39) và số đo trên chính T4.

⚠ **Bẫy đo (đã sập HAI lần, ghi lại cho kỹ):** endpoint history của W&B cho các run này
**trễ 25–30 phút**, trong khi stream system (GPU util/power) gần như tức thời.
* Lần 1: đọc "1 round, 32,6 phút trước" trong khi round 2 đã log xong.
* Lần 2 (2026-09-11): round 3 chưa xuất hiện sau 35 phút, GPU vẫn 47% và 77 °C ⇒ kết luận
  "round 3 chậm thật". **Sai.** Round 3 đã xong lúc 17:04:40Z với `gap` = **17,9 ph**,
  đúng nhịp. Lý lẽ hỏng là: *"system stream tươi ⇒ sender không tắc ⇒ history cũng tươi"* —
  hai stream được **nạp riêng**, độ tươi của cái này không nói gì về cái kia.

**Quy tắc:** tín hiệu đáng tin duy nhất là **một hàng MỚI thực sự xuất hiện**. "Chưa có hàng"
không bao giờ phân biệt được *trễ* với *chậm*. Dùng `gap` giữa hai `_timestamp` liên tiếp;
đừng dùng "thời gian kể từ round cuối"; `r.load(force=True)` không ép làm mới được.
Và `scan_history(keys=[...])` chỉ trả hàng nào có **đủ mọi key** — hỏi `round` (key bị loại
khỏi log, lỗi #38) sẽ ra **0 hàng**, trông y hệt như run chưa chạy gì.

---

### 3.3 Probe compile sm_75 — 2026-09-10, `minhtran0601`, ~0,15 h quota

Câu hỏi quyết định cả ngân sách: `_compile` có chứng nhận được `torch.compile` trên T4 không?
Không cần prepack — model là launch-bound nên đo chi phí phóng kernel trên tensor ngẫu nhiên
cỡ một shard client thật là đủ. **`certified=True` ở MỌI cấu hình.**

| cấu hình | eager ms/step | compiled ms/step | **speedup** | gate |
|---|---|---|---:|---|
| B=512, 1 proc | 22,13–22,30 | 9,58–10,42 | **2,31×** | 78,0 s |
| B=512, 2 proc (20c/50c) | 21,54–23,66 | 9,72–10,55 | **2,22–2,35×** | 19,0 / 75,7 s |
| B=256, 2 proc (100c) | 23,09–24,54 | 8,78–10,06 | **2,63–2,65×** | 87,3 / 87,6 s |

`max|dlogit|` 6,10e-04 … 7,32e-04, **0 top-1 flip** trên toàn bộ 512/256 hàng. `cpu_count = 4`
— xác nhận đúng cái hộp 4 vCPU mà chẩn đoán launch-bound dựa vào. `cpu/wall ≈ 0,97` ở **cả
hai** chế độ: tiến trình bão hoà CPU dù compiled hay eager, đúng dấu hiệu launch-bound.

Nguyên nhân gốc là **lỗi #39, do chính gate của tôi**, không phải Triton trên sm_75. Bắt được
ở **local sm_86** trước khi tốn quota — nơi Triton không hề bị nghi ngờ, nên nó loại ngay giả
thuyết "sm_75 hỏng".

**Ảnh hưởng lên một round (20c/50c):** train 853 s → ~370 s; teacher 221 s và eval ~40 s
**không** co lại (teacher chạy batch 16.384, compute-bound, không phải launch-bound; proximal
`_foreach` nằm **ngoài** graph compiled nên phần tiết kiệm của nó cộng thêm). Round
**17,9 ph → ≈10,5 ph**; 50 round: **14,9 h → ≈8,8 h/kịch bản**.

⚠ Probe đã push **mang `fdids.py` bản cũ** (proximal vòng lặp): phần lợi của `_foreach`
**chưa** nằm trong các số trên.

---

### 3.4 Lần push thứ ba (v3) — compile BẬT, 2026-09-11

`backend=compiled` trên **cả ba** run, đọc được từ `summary["backend"]` **khi run còn sống** —
điều v1/v2 không làm được (lỗi #38, nay đã vá và đã xác nhận trên phần cứng thật).

| run | round | `seconds` | teacher | skip/steps | f1_macro | accuracy | ce |
|---|---:|---:|---:|---|---|---|---|
| 20c | 1 | **520,3 s** | 226,3 s | 5 / 84.083 | 0,161194 | 0,377478 | 0,6491 |
| 20c | 2 | **521,9 s** | 229,9 s | 10 / 84.083 | 0,582295 | 0,765267 | 0,5862 |
| 50c | 1 | **532,0 s** | 235,4 s | 0 / 84.098 | 0,234190 | 0,510238 | 0,6502 |
| 100c | 1 | **737,9 s** | 230,8 s | 1 / 168.200 | 0,207831 | 0,451180 | 0,6216 |

Startup (`_runtime − seconds` của round 1): 20c **12,6 ph** · 50c **13,1 ph** · 100c **16,9 ph**.
100c: `(737,9 − 230,8)/84.100 = ` **≤6,0 ms/step**, nhanh hơn 20c/50c theo step — đúng như probe
báo (2,63× ở batch 256 so với 2,2–2,35× ở batch 512). `ce` 0,6216 khớp eager 0,621527.

**17,9 ph → 8,7 ph/round = 2,06×** trên **cả round**, tốt hơn dự đoán 10,5 ph. Trừ teacher:
`(520,3 − 226,3)/42.041 ≤ **7,0 ms/step**` — nhanh hơn cả 9,7–10,6 ms/step probe đo trên dữ
liệu ngẫu nhiên. `gap` giữa hai `_timestamp` (8,7 ph) **khớp** `seconds` đo trong tiến trình:
hai đồng hồ độc lập xác nhận lẫn nhau.

**Kiểm chứng số học — đường học trùng với eager:**

| 20c | eager (v2) | compiled (v3) |
|---|---|---|
| r1 `ce` | 0,6490 | **0,6491** |
| r2 `ce` | 0,5865 | **0,5862** |
| r2 `f1_macro` | 0,577886 | **0,582295** |

`ce` khớp tới 4 chữ số ở **cả hai** round. `f1`/`accuracy` lệch nhẹ vì mask dropout của
Inductor **thật sự khác** eager — đúng bản chất của lỗi #39 — và đã hội tụ lại gần nhau ở
round 2 sau khi lệch nhiều hơn ở round 1. Đây là dấu hiệu đúng: **`ce` là tín hiệu ổn định,
metric phân loại ở round đầu thì nhiễu.**

Xác nhận thêm hai bản sửa trên phần cứng thật: `seconds` **có giá trị** (lỗi #37) và
`skip = 5, 10, 0` được đếm đúng là warm-up chứ không phải phân kỳ (bản sửa R11).

**Skip đã bão hoà rồi ĐẢO CHIỀU — mối lo đóng lại.** 20c: 5, 10, 18, 24, 32, 37, **40**, 38,
31, 32. Đúng như mô hình log: `GradScaler` khởi từ 2^16 và **chia đôi** mỗi lần overflow, nên
skip/client ≈ `log2(init_scale / stable_scale)` — **logarit** theo độ lớn gradient, không tuyến
tính theo round. Ngoại suy tuyến tính (5→32 trong 5 round ⇒ ~350 ở round 50) là **sai mô hình**.
50c: skip = 0 suốt, vì 50 client × ~1.682 bước thì scaler ổn định mà không bỏ bước nào.

### 100c bị NGHẼN CPU, không phải nhiệt, không phải rò bộ nhớ

| run | GPU util | smClock | power | temp | round |
|---|---:|---:|---:|---:|---|
| 20c | **100%** | 1290 MHz | 67,3 W | 73 °C | 522 s, phẳng |
| 50c | **100%** | 1245 MHz | 66,9 W | 77 °C | 530 s, phẳng |
| 100c | **58%** | **1575 MHz** | 67,6 W | 77 °C | 738 → 903 s, tăng |

Cả ba cùng chạm trần 67 W và 77 °C ⇒ **không phải nhiệt**. `vram_train_gb` đứng yên 7,13 suốt
và train+eval **không đơn điệu** (507, 512, 590, 557, 619, 619, 642, 749, 691, 671) ⇒ **không
phải rò bộ nhớ trong code**.

**Cách đọc quyết định: util PHẢI đọc cùng với clock.**
* 20c/50c — **100% util nhưng clock bị kéo xuống** ~1270 MHz: GPU bận liên tục, bị trần công
  suất ghì lại. Đây là trạng thái *bão hoà tính toán*.
* 100c — **58% util mà clock gần đỉnh** 1575/1590 MHz: GPU **boost được vì nó đang rảnh**.
  Đây là trạng thái *đói CPU*.

Nguyên nhân: batch 256 ⇒ **168.200 bước/round** so với 84.083, tức **gấp đôi số lần phóng
kernel** trên cùng **4 vCPU**. CUDA graph đã cắt phần lớn chi phí phóng nhưng phần Python còn
lại vẫn không nuôi kịp hai T4. Nó cũng giải thích **độ dao động**: tiến trình nghẽn CPU nhạy
với mọi thứ khác dùng chung máy, nên 100c chạy lêu bêu trong khi 20c/50c đứng trong 3 s suốt
20 round.

**Ngân sách cập nhật theo số đo thật:** 20c ≈ 41 round trong 6,4 h; 50c **đủ 50 round** trong
9,7 h (cần ~7,8 h, còn dư); 100c ≈ **23 round** trong 6,2 h (không phải 28 — round đã trôi lên
~900 s).

### 3.6 Phiên 1 (v3) kết thúc — kéo output và kiểm, 2026-09-11

Cả ba kernel `COMPLETE`, W&B `finished`. Kéo bằng `kaggle kernels output --page-size 200`
(CLI 2.2.4 tự lật trang, giữ đường dẫn tương đối; file 0 byte duy nhất ngoài `complete/*.done`
là `proj/__init__.py`, vốn rỗng). `verify_run.py --y-true reports/y_true.u8.npy` **mode full**:

| run | round | thời gian phiên | round steady | dừng vì | verify | f1 tốt nhất / cuối |
|---|---:|---:|---:|---|---|---|
| `fdids_20c_v2` (minhtrit06) | **42**/50 | 6,15 h | 520–524 s | budget 6,40 h, sau r42 | pass 1..42 | 0,7822 @r7 / 0,7684 |
| `fdids_50c_v2` (khanhmay0304) | **50**/50 ✅ | 7,44 h | 528–538 s | đủ 50 round | pass 1..50 | 0,7205 @r6 / 0,6815 |
| `fdids_100c_v2` (khanhmay0304) | **25**/50 | 5,86 h | 738–981 s | budget 6,20 h, sau r25 | pass 1..25 | 0,6870 @r9 / 0,6649 |

Executed notebook (18 output/cell, `workers ready on compiled`, dòng `[r001]..[rNNN]`) lưu
cạnh nguồn: `notebook/<K>c/fdids_<K>c_v2.executed.s1.ipynb`, mode 600 — **chứa W&B key**.

### 3.7 Continuation qua checkpoint DATASET — đo 2026-09-11, `minhtriethihi`

`kernel_sources` không qua được ranh giới tài khoản, nên chọn đường dataset. Đo được:

| bước | đo được |
|---|---|
| `KGAT_` token = API token của CLI | `KAGGLE_API_TOKEN=<đường dẫn file>` được SDK 2.2.4 đọc từ file; `IntrospectToken` trả `active=True, username=minhtriethihi`; quota 30,00 h. **Không cần** OAuth để push/kéo. Cả 4 bearer cũ cũng `active=True` — kết luận "chết" ở §14.5 CONTEXT đến từ probe trực tiếp, vốn đã ghi là không tin được |
| `datasets create -r zip -t` | 510 MB → **105 MB** zip (20c), 308 MB → 68 MB (100c); upload 13 s / 10 s; `ready` sau ~1 phút. Kaggle **bỏ đúng một cấp** thư mục: local `runs/fdids_20c_v2/…` ⇒ mount `…/fdids_20c_v2/…` |
| đối chiếu tên + kích thước | 297/297 và 178/178 file khớp **từng byte kích thước**; `history.csv` 14.853 B nguyên vẹn (`-t` chặn chuyển đổi tabular) |
| mount thật | `/kaggle/input/datasets/minhtriethihi/fdids-20c-v2-ckpt/fdids_20c_v2` — có thêm cấp `datasets/<owner>/` so với `/kaggle/input/<slug>` cũ; `_attached_source` rglob nên không việc gì |
| CPU probe (`scripts/gen_resume_probe.py`) | chạy **đúng** `ckpt.resolve_resume` production với cfg từ `reports/manifest.json`: `42 marker, 42 verified, imported 1..42` và `25/25/1..25`; `RESUME PROBE OK` cả hai; ~35 s mỗi kernel, **0 h GPU** |
| fingerprint | CFG của notebook continuation (AST) + `data_id` từ manifest ⇒ `5f74ba70f2c0af90` (20c) và `43fb1276b4aa83b1` (100c), **khớp** fingerprint lưu trong weights |
| W&B resume | probe run tạm: `resume="allow"` trên run `finished` giữ step 1..3, nhận 4..5, config đổi (`max_seconds`, `require_resume`) không lỗi; đã xoá run probe |
| push | `minhtriethihi/fd-ids-veremi-{20,100}-clients` v1, 01:46:10Z; cả hai `RUNNING` trong <40 s, W&B `running` trong <1 phút |

Bài học: **đường continuation phải được chứng minh ở CPU trước** — nó tốn 35 s và trả lời
cả câu hỏi layout mount lẫn tính toàn vẹn từng round, còn cổng `require_resume` trên GPU chỉ
trả lời "có hay không" sau khi xếp hàng. Và **`last == expected`** là điều cổng GPU *chưa*
kiểm: một import dừng sớm sẽ **train lại** các round còn thiếu dưới cùng `run_name` và W&B
sẽ vứt step không đơn điệu mà không nói gì; probe kiểm đúng điều đó, notebook production
thì chưa (ghi ở CONTEXT §15.5).

### 3.8 Phiên 2 (continuation trên `minhtriethihi`) — 20c XONG 50/50, 2026-09-11

| hạng mục | 20c phiên 2 (`minhtriethihi/fd-ids-veremi-20-clients` v1) |
|---|---|
| cổng resume | `42 marker(s), 42 verified, imported 1..42` → `resume from round 42` (log live) |
| prepack | 132,0 s (cache mới, đủ 43.045.415 / 10.761.343 dòng) |
| compile | `compile OK` cả hai rank, số **giống hệt** phiên 1 (7,32e-04 / 6,10e-04); `workers ready on compiled` 246 s |
| round | **509–510 s** × 8 (phiên 1: 520–524 s; máy khác, ít nghẽn hơn) |
| phiên | **1,20 h** cho 8 round (startup ~4 ph + import + prepack) |
| skip | 21–27/84.083, ổn định |
| W&B | `resume="allow"` nối đúng run: step 43..50 vào history, hàng đầu chỉ trễ ~8 ph. `state` **nhảy** `running`↔`crashed` suốt phiên — bẫy đã ghi ở skill `wandb.md` (heartbeat đứng lúc spawn worker; system stream vẫn chảy) |
| f1_macro r43..r50 | 0,7669 · **0,7294** · 0,7499 · 0,7606 · 0,7588 · 0,7641 · 0,7662 · **0,7655** |
| verify (s2 pull) | `--require-rounds 50` **pass, mode full** |
| merge s1+s2 | `merge_sessions.py`: history khớp 42/42 hàng overlap; **weights và preds round 1..42 giống hệt từng byte** giữa hai pull (`diff -rq` độc lập); 50 round liên tục; `logs/sessions/{1,2}` giữ manifest + log kernel từng phiên |
| verify (merged) | `runs/fdids_20c_v2` **pass 1..50, mode full** |

**Round 44 tụt 0,7669 → 0,7294 (−3,75 điểm; phiên 1 tụt lớn nhất chỉ −1,43 ở r19).** Đã soi
`logs/round_04{3,4,5}.json`: **không có client nào bất thường** — ce max 0,32 cùng client 12 ở
cả ba round, gnorm max ≤ 0,40, nonfinite 0, skip 0–4/client. `ce_client_mean` r45 tăng lên
0,2293 là **hệ quả** (client bắt đầu từ global model xấu hơn của r44), không phải tín hiệu
độc lập. Hồi về 0,764–0,766 ở r48–r50. Kết luận: dao động một round của phép gộp trên dữ liệu
non-IID, không phải lỗi continuation (weights/code/data/RNG đều đã kiểm giống phiên 1).

**Đường cong 20c đủ 50 round:** đỉnh `f1_macro` **0,7822 @r7**; r50 **0,7655**; 10 round cuối
trung bình 0,7596 (kéo xuống bởi r44); `accuracy` 0,8161 (r4) → 0,7327; `f1_weighted` → 0,7297.
10 metric r50: acc 0,7327 · P_macro 0,7837 · R_macro 0,7887 · F1_macro 0,7655 · P_w 0,7971 ·
R_w 0,7327 · F1_w 0,7297 (micro = acc). Ai trả tiền round nào: 1–42 `minhtrit06`, 43–50
`minhtriethihi` (`logs/sessions.json`).

**Sửa skill `merge_sessions.py` (chung, cả hai cây)** vì nó từ chối sai một merge đúng: (1)
`history.csv` chỉ tìm ở `metrics/`, driver này ghi ở gốc run ⇒ "longest is 0 rounds"; (2)
`reports/manifest.json` bị so từng byte dù nó là provenance **từng phiên** (max_seconds,
require_resume, startup) — nay cùng nhóm `PER_SESSION` với `config.json`, giữ ở
`logs/sessions/<n>/`; (3) `logs/round_*` là artifact từng round (per-client stats), nay merge
với kiểm identity như weights, phần còn lại của `logs/` vẫn giữ theo phiên.

**100c phiên 2 đang chạy** (lúc 03:35Z: r34, 676–681 s/round, f1 0,659–0,669, ce 0,263–0,266,
skip 1–6). Còn 16 round ⇒ ≈ 06:35Z. Pull `.s1` giữ lại để merge.

### 3.9 Phiên 2 — 100c XONG 50/50, merge, report HOÀN TẤT, 2026-09-11 ~10:00Z

| hạng mục | 100c phiên 2 (`minhtriethihi/fd-ids-veremi-100-clients` v1) |
|---|---|
| cổng resume | `25 marker(s), 25 verified, imported 1..25` → `resumed at round 26` (executed notebook) |
| prepack / compile | 132,3 s; `compile OK` hai rank 7,32e-04 / 9,77e-04, `workers ready on compiled` 250 s |
| round | **671–753 s** × 25 (phiên 1: 738–981 s); Σ `seconds` 4,77 h; phiên **4,84 h** kể cả startup |
| kết thúc | `COMPLETE`, log cuối 17.453 s sau khi bắt đầu ⇒ **≈06:37Z**; W&B `Synced 5 file(s)` |
| skip | 1–6/168.200 |
| f1_macro r26..r50 | 0,6684 … 0,6695 (r31) … 0,6491/0,6478 (r37–38) … 0,6473 (r44) … **0,6606** (r50); tụt lớn nhất −1,75 điểm @r44 |
| verify (s2 pull) | `--require-rounds 50` **pass, mode full** (609 MB, 378 file; grep key: chỉ có tên biến `WANDB_API_KEY` trong warning của wandb, không có giá trị) |
| merge s1+s2 | `merge_sessions.py`: history khớp 25/25 hàng overlap; 50 round liên tục; new rounds per session [25, 25]; `logs/sessions/{1,2}`; ba thư mục rỗng `checkpoints client_log protos` đã `rmdir` |
| verify (merged) | `runs/fdids_100c_v2` **pass 1..50, mode full**; cây file giống hệt bản merge 20c (chỉ khác tên log kernel) |
| dọn | xoá `runs/pulls/` (cả `.s1` lẫn `.s2`); executed notebook `notebook/100c/fdids_100c_v2.executed.s2.ipynb` (0600, **có key**) |

**10 metric r50 100c:** acc 0,6552 · P_macro 0,6856 · R_macro 0,7149 · F1_macro **0,6606** · P_w 0,7287 ·
R_w 0,6552 · F1_w 0,6223 (micro = acc). Đỉnh 0,6870 @r9; 10 round cuối trung bình 0,6584. Ai trả tiền
round nào: 1–25 `khanhmay0304`, 26–50 `minhtriethihi`.

**Report HOÀN TẤT** (`python scripts/make_report.py` → `report.md` 649 dòng, 10 hình): cổng độc lập
3 × 50 hàng, header đúng thứ tự `METRIC_KEYS`, **0 lệch** với `metrics/round_NNN.json` ở 4 chữ số,
10 ảnh tồn tại, không còn nhãn INTERIM. Sửa generator để prose không còn hardcode hai kịch bản:
caption §0 (mức plateau từng cấu hình, tính từ 10 round cuối: 20c ≈ 0,76 / 50c ≈ 0,68 / 100c ≈ 0,66,
đơn điệu theo số client), §8 (đỉnh r6–9; plateau thấp hơn đỉnh 3–5 % tương đối; accuracy xói mòn
6–9 điểm, F1 weighted 8–11 điểm; continuation liệt kê 20c 42 round + 100c 25 round chồng lấn).
**Đính chính một câu sai của bản interim:** "loss huấn luyện giảm rồi đi ngang, không tăng trở lại"
chỉ đúng cho **CE**; số hạng **KD** chạm cực tiểu ở r7–8 rồi **tăng dần** tới r50 ở cả ba cấu hình
(20c 0,1298 → 0,1585; 50c 0,1577 → 0,2099; 100c 0,1901 → 0,2341) — nay caption từng hình, tiêu đề
panel phải và §8 nêu CE/KD tách bạch, số tính từ artifact.

### 3.5 Mẫu ghi mỗi lần chạy
```
### <slug> — <ngày>, phiên <n>, tài khoản <acct>
| hạng mục | đo được |
| GPU xác nhận | device_count / capability / tên |
| compile | max|Δlogit|, rel|Δgrad|, số dòng decisive; có rơi eager không và vì sao |
| prepack | giây, và cache có hit manifest không |
| startup (T0 → workers ready) | giây |
| round time | round 1 / steady / xấu nhất, đã bao gồm commit I/O |
| peak VRAM | GiB mỗi GPU (từ `logs/round_NNN.json`) |
| skipped / nonfinite | tổng theo round; kỳ vọng skip>0 vài round đầu do GradScaler warmup |
| f1_macro theo round | ... |
| verify_run | pass/fail, require_rounds |
| quota tiêu | giây |
```

Những thứ **chỉ** kết luận được ở đây, không phải ở local (`LOCAL_ENV.md` §5):
`torch.compile(reduce-overhead)` trên sm_75, topology 2 worker/2 GPU, mọi con số thời gian
tuyệt đối, prepack đủ 43 M dòng, peak VRAM thật, và do đó `max_seconds`.
