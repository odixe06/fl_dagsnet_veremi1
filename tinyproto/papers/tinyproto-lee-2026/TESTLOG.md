# TESTLOG — TinyProto-FP, những gì thực sự đã chạy

**Review mới nhất 2026-09-08:** đã kiểm 7 file thật trong `production/`, cổng μ với footer/scaler
local thật và 15 config/metric đã pull. 20 regression pass (peak 620 MiB); diagnostic pass
(peak 645 MiB), tái hiện lỗi W&B log, bốn thiếu sót validator và cổng AMP có thể pass khi không
có hàng decisive. Log `review-logs/production-review-{regressions,checks}.log`.
Xem [PRODUCTION_REVIEW.md](PRODUCTION_REVIEW.md) và CONTEXT §8: các câu lịch sử bên dưới về
“mọi lỗi W&B đều bị nuốt”, “AMP an toàn toàn bộ” hoặc “đã kiểm resume chéo tài khoản” cần đọc
với giới hạn mới. Chưa sửa mã thực thi hoặc chạy Kaggle trong review này.

Ghi chép bằng chứng. Mỗi dòng ứng với một log trong `review-logs/`. Nguyên tắc: **một mức kiểm
chứng không chứng minh cho mức khác.** Đúng/sai ở local không nói gì về tốc độ; hai GPU chạy được
không nói gì về việc 50 round có vừa quota hay không.

Ba mức tách bạch:

| Mức | Phần cứng | Trả lời được câu hỏi gì |
|---|---|---|
| L1 local | CPU, 1 worker, `compile=False`, fixture 4 client × 2.048 dòng | thuật toán/artifact **đúng hay sai** |
| L2 Kaggle bounded | 2 × T4, mẫu dữ liệu **thật**, 4 client/kịch bản, 3 round | đường đi hai GPU, AMP, compile, resume **chạy được hay không** |
| L3 Kaggle full-data | 2 × T4, dữ liệu đầy đủ | **nhanh/chậm**, `eval_group`, `max_hours` |

---

## L1 — local (CPU, 1 worker)

Máy phát triển: Ubuntu 24.04 trên WSL2, **7,6 GiB RAM**, RTX 3050 4 GB. Mọi lệnh chạy qua
`scripts/run_local_checked.py` (khoá một cây test, trần RSS 2.907 MiB, sàn MemAvailable 1.536 MiB).
Bộ test local **cố ý chạy trên CPU một worker** để vừa RAM; nó **không** kiểm chứng đường hai GPU.

| Log | Nội dung | Kết quả |
|---|---|---|
| `unit-final.log` | 10 metric đối chiếu sklearn | max\|Δ\| = **1,1e-16**; `ALL UNIT TESTS PASS` |
| `resume-final.log` | 6 kịch bản crash/resume, có so bitwise liên tục vs resume | `ALL RESUME TESTS PASS` |
| `tamper-final.log` | 9 kiểu hỏng artifact cố ý | **9/9 CAUGHT** |
| `verify-final.log` | verifier trên run sạch | **235 check**, 0 fail |
| `real-data-smoke.log` | forward/backward CE + regularizer trên dòng **thật** | PASS cả 20/50/100 |
| `suite-final.log` | toàn bộ ở trên | `REVIEW SUITE PASSED`, đỉnh RSS 2.072 MiB |
| `suite-session2.log` | chạy lại **sau** các sửa đổi phiên 2 | `REVIEW SUITE PASSED`, đỉnh RSS 2.095 MiB, exit 0 |

Kiểm thêm ở phiên 2 cho cổng resume mức sweep (`src/sweep.py::plan_sweep`), 4 trường hợp:
đếm đúng round đã commit của ứng viên đã chạy; ứng viên chưa từng chạy đọc ra 0; tìm được state
qua mount lồng nhau kiểu `/kaggle/input`; và **không** khớp chéo giữa các kịch bản (sweep 50 client
không nhận state của 20 client). Mô phỏng notebook `tinyproto-fp-mu-sweep` chạy hết, chọn winner,
in đúng hướng dẫn đính kèm thay vì chép tay.

Kiểm cổng chọn μ cho production (`sweep.selected_mu` / `choose_winner`), **12/12 pass**: chấp nhận
tài liệu hoàn chỉnh đúng chữ ký; từ chối sweep dở dang, tài liệu không có winner, **sai kịch bản**,
**sai data fingerprint**, đổi batch, μ = 0 và μ = NaN; và `choose_winner` trả `None` khi grid thiếu
ứng viên, có ứng viên lỗi, hoặc có ứng viên bị cắt ngắn round — nghĩa là một μ **không thể** được
chọn từ grid chưa chạy đủ.

Giới hạn đã biết: fixture nhỏ chứng minh **số học và hành vi crash**, không phải hiệu năng.

---

## L2 — Kaggle 2 × T4, dữ liệu thật (đã PASS)

Notebook `odixe0502/tinyproto-fp-2t4-validation`. 4 client/kịch bản, 8.192 dòng/client,
1.024 dòng test, 3 round, hai worker `cuda:0`/`cuda:1`, AMP + compile bật.
**Điểm số ở mức này không phải kết quả nghiên cứu** — mẫu quá nhỏ.

### v1 (2026-09-07) — `kaggle-odixe-v1.log`
Bị **huỷ** giữa kịch bản 100 (t = 267 s; ngân sách 2.700 s), do session bị dừng, không phải lỗi code.
Kịch bản 20 và 50 đã xong và pass. **Phát hiện quyết định:** `OK: 2 x Tesla T4 (sm_75) confirmed` —
chứng minh lỗi 0-GPU trước đó **thuộc về tài khoản `khanhngoc0304`**, không phải code hay metadata.

### v2 (2026-09-08) — `kaggle-odixe-v2.log` → **PASSED**
Chạy hết 293 s, `remote_validation.json` có `"status": "passed"`.

| Kịch bản | batch | round | compile | max \|Δlogit\| | bất đồng | verifier |
|---|---|---|---|---|---|---|
| 20 | 512 | 3/3 | bật, 2/2 worker | 3,58e-07 | **0** | 335 check, 0 fail |
| 50 | 512 | 3/3 | bật, 2/2 worker | 4,47e-07 | **0** | 335 check, 0 fail |
| 100 | 256 | 3/3 | bật, 2/2 worker | 3,87e-07 | **0** | 335 check, 0 fail |

Đã chứng minh thêm: resume qua ranh giới session trên kịch bản 20 (`rounds_this_session=2`
→ dừng sau round 2 → `require_resume=True` → `client state restored from round 2` → round 3);
`require_resume` từ chối khi không có gì để resume; mask CPS trên Kaggle **trùng khít** số đo local
(overlap 0/7,275/10, hamming min 80).

Môi trường worker đã xác nhận: python 3.12.13, torch 2.10.0+cu128, CUDA 12.8, 2 × Tesla T4
14,6 GB sm_75 40 SM, 4 vCPU, 31,3 GB RAM, 19,5 GB trống ở `/kaggle/working`.
Ghim trong `knowledge/runtime.json` kèm xuất xứ.

**Lưu ý đọc log:** Kaggle phát lại output của một cell khi cell kế tiếp bắt đầu, nên các khối
giống hệt nhau liên tiếp là **hiện tượng của log**, không phải chạy hai lần — `history.csv` của mỗi
run chỉ có đúng một dòng mỗi round, và verifier đã kiểm điều đó.

---

## L3 — Kaggle full-data calibration

Notebook `odixe0502/tinyproto-fp-calibration`. Đo prepack 43 M dòng, thời gian một step
(eager vs compile × AMP × batch 512/256), eval (unfolded / folded / vmap G = 2,4,8) trên test thật,
rồi chạy **3 round dữ liệu đầy đủ** kịch bản 20 client để lấy thời gian round thật.

**Trạng thái: ĐÃ XONG (2026-09-08).** `max_hours=11.0`, `eval_batch=8192` và `eval_group=1`
trong 7 notebook nay đều **suy từ số đo dưới đây**, không còn là giá trị đặt trước.

### Kết quả calibration — 2026-09-08, `calibration.json` + `kaggle-odixe-calibration.log`

Chạy hết **3.430 s** (~57 phút), COMPLETE. torch 2.10.0+cu128, 2 × Tesla T4.

**Đọc dữ liệu.** train 818.144 dòng/s (4,65 M dòng trong 5,7 s); test 10.761.343 dòng trong 14,9 s.
Một worker nạp 21,9 M dòng train + toàn bộ test = **4,02 GiB** thường trú trong **54 s**.
`abs_max` 535,5 (train) / 570,4 (test) → fp16 an toàn. `raw_mean_f_snd_spd` = 7,4493, khớp
`knowledge/DATASET.md` — scaler đã áp đúng cho test và **không** áp lại cho train.

**Thời gian một step (ms).** `torch.compile(reduce-overhead)` là thắng lợi lớn nhất:

| batch | eager AMP | eager fp32 | compile AMP | compile fp32 | compile+AMP nhanh hơn eager+AMP |
|---|---|---|---|---|---|
| 512 | 18,01 | 13,43 | **6,21** | 9,65 | **2,90×** |
| 256 | 19,44 | 14,53 | **5,73** | 5,73 | **3,39×** |

Lưu ý ngược đời: **eager + AMP chậm hơn eager fp32** (18,01 vs 13,43). Chỉ khi có compile thì AMP
mới có lợi. Đây là lý do phải đo thay vì áp lời khuyên AMP chung.

**Fold BatchNorm.** Chính xác: max |Δlogit| = 4,77e-07, max |Δfeat| = 1,43e-06, **0 bất đồng argmax**.
Nhanh hơn **1,56×** (297.289 vs 190.083 client-dòng/s ở eval_batch 16384).

**eval_batch.** 8192 → **304.397** client-dòng/s; 16384 → 297.289; 32768 → 295.629.
Đã đổi cấu hình sang **8192** (nhanh hơn 2,4% và tốn ít bộ nhớ hơn).

**`eval_group` — QUYẾT ĐỊNH: giữ 1.** Đường vmap **chậm hơn**, không phải nhanh hơn:

| G | speedup vs folded-sequential | `cm_proto_mismatch` | đỉnh bộ nhớ |
|---|---|---|---|
| 2 | **0,50×** | 4.546 | 5,29 GB |
| 4 | **0,64×** | 4.554 | 5,29 GB |
| 8 | **0,62×** | 4.554 | 5,29 GB |

**Đọc đúng con số:** `cm_proto_mismatch` là **tổng trị tuyệt đối chênh lệch giữa hai ma trận
confusion**, không phải "số ô khác nhau" cũng không phải số dòng dự đoán khác nhau. `mismatch_pct`
= 0,028% là con số đó chia cho tổng số lượt dự đoán. Muốn biết tỉ lệ dự đoán thực sự bất đồng thì
phải so từng `y_pred`, chưa làm. Kết luận **loại vmap dựa trên TỐC ĐỘ** (0,50–0,64×) — đây là lý
do đủ và không phụ thuộc cách diễn giải con số lệch. Việc nó không trùng số là lý do phụ.

**Thời gian round thật, 20 client, dữ liệu đầy đủ** (đã gồm commit checkpoint):

| round | tổng | train | eval | proto f1_macro | reg loss |
|---|---|---|---|---|---|
| 1 | 810,8 s | 453,2 s | 353,9 s | 0,5960 | 0,0000 |
| 2 | 762,8 s | 408,9 s | 350,5 s | 0,5937 | 0,1655 |
| 3 | 760,9 s | 408,7 s | 348,8 s | 0,4575 | 0,7101 |

**Steady-state 763 s/round → 50 round = 10,59 h.** Setup mỗi phiên ~100 s (decode 54 s + compile 44 s).
⚠️ Số 763 s đo ở **`eval_batch=16384`** (giá trị lúc chạy calibration). Cấu hình nay là **8192**,
mới chỉ nhanh hơn 2,4% trong **microbenchmark**, **chưa** được xác nhận trong một round thật.
Đỉnh bộ nhớ GPU 4,75 GB / 14,6 GB — còn rất nhiều chỗ. Chia GPU 10/10 client, lệch 1,963%.

**Tín hiệu khoa học cần chú ý:** ở k = 1,0 (`mu_kind=inv_mean_nij`), reg loss tăng
0,000 → 0,166 → 0,710 trong khi proto f1_macro **tụt** 0,596 → 0,594 → 0,458. Nghĩa là
**μ ở k = 1,0 quá mạnh** và đang kéo hỏng prototype cục bộ. Đây chính là lý do phải sweep;
grid `[0,03; 0,1; 0,3; 1; 3]` nhiều khả năng chọn k nhỏ hơn 1. Ba con số trên **không** phải kết quả
nghiên cứu — mới 3 round và μ chưa chọn.

### Phép chiếu ngân sách, suy từ số đo (không phải phỏng đoán)

eval tỉ lệ tuyến tính theo số client; train giữ nguyên với 20/50 (cùng tổng dòng, cùng batch 512),
còn 100 client có gấp đôi số step nhưng ở batch 256 (5,73 ms/step).

| Kịch bản | train/round | eval/round | round | 50 round | phiên ở `max_hours=11` |
|---|---|---|---|---|---|
| 20 | 409 s (đo) | 341 s | **755 s** | **10,5 h** | **1** |
| 50 | ~409 s | 851 s | ~1.265 s | ~17,6 h | 2 |
| 100 | ~754 s | 1.704 s | ~2.463 s | **~34,2 h** | 4 |

Ba sweep μ rẻ hơn nhiều vì chấm trên validation 2% (~861 k dòng) chứ không phải 10,76 M dòng test:
ước tính ~2,4 h + ~2,7 h + ~5,0 h ≈ **10,1 h**.

**Tổng ≈ 72 h GPU** cho toàn bộ nghiên cứu.

⚠️ **Giới hạn của phép chiếu.** Chỉ hàng 20 client dựa trên round đã đo (và ở eval_batch 16384).
Hàng 50/100 là **suy ra**, chưa từng chạy. Con số này **chưa trừ**: setup mỗi candidate/phiên
(~100 s × số lần gọi), thời gian import khi resume, round cuối ghi prediction (~2,15 GB ở
100 client), và dao động runtime của Kaggle. Hãy coi ~72 h là **sàn**, không phải số giờ quota
sẽ bị trừ. Riêng kịch bản 100 client (~34,2 h) **đã vượt quota 30 h/tuần của một tài khoản**.

---

## Những gì vẫn CHƯA được chứng minh

- Chưa có kịch bản production nào chạy đủ **50 round trên dữ liệu đầy đủ**. Đã chạy **3 round**
  dữ liệu đầy đủ cho kịch bản 20 client (trong calibration) — đủ để lấy thời gian, **không** đủ để
  báo cáo kết quả.
- Kịch bản 50 và 100 client **chưa từng chạy trên dữ liệu đầy đủ**; thời gian của chúng là phép
  chiếu từ số đo 20 client, chưa phải số đo trực tiếp.
- μ: **20 và 50 client đã chọn xong** (k=0.1 và k=0.3), 100 client đang chạy. `mu_value` trong
  notebook production vẫn là chỗ giữ chỗ — giá trị thật được đọc từ `mu_sweep.json` đính kèm và
  `require_mu_selection=True` **từ chối chạy** nếu thiếu hoặc sai chữ ký/protocol.
- `eval_group=1` nay đã **được quyết bằng số đo**, không còn là mặc định an toàn: vmap chậm hơn
  0,50–0,64× và lệch 0,028% ô confusion.
- Dừng sớm không phải là hoàn thành. Chỉ `verify_run.py --require-complete` mới chứng nhận một
  kịch bản đã xong.

---

## Rà soát trước chạy thật R1–R6 (2026-09-08, phiên 3) — đã sửa và có regression

Sáu vấn đề được nêu trong rà soát trước khi chạy thật. Bằng chứng tái hiện ban đầu:
`review-logs/prelaunch-review-checks.log` (các dòng `OBSERVED`). Sau khi sửa, mỗi vấn đề có
regression bền vững trong `tests/review_regressions.py` để không tái phát.

| # | Vấn đề | Đã sửa thế nào | Test bảo vệ |
|---|---|---|---|
| **R1** | Cổng nhận μ chỉ kiểm status + μ dương: vẫn nhận sweep 1 round, grid 1 ứng viên, results rỗng, fraction/seed validation khác, winner ngoài grid; signature bỏ qua scaler | Đóng băng `MU_PROTOCOL` (version, grid, rounds, fraction, seed, criterion) trong `src/sweep.py`; summary mang theo protocol; `selected_mu` kiểm protocol, đủ ứng viên, **tính lại winner** từ results để đối chiếu; `scaler_fingerprint` vào signature | `MuProtocolContract` — 9 biến thể sai đều bị từ chối, 3 kiểu winner giả bị bắt |
| **R2** | Resume sweep chỉ hỏi "có ứng viên nào không": mất output riêng một ứng viên vẫn qua cổng và ứng viên đó bị lên lịch chạy lại; cổng lại nằm **sau** bước đọc toàn bộ train | `read_claims` + `plan_sweep(claims=...)`: ứng viên từng được ghi nhận commit mà thiếu thư mục → **dừng**, không chạy lại. Ghi `mu_sweep_progress.json` sau mỗi ứng viên. Cổng chuyển lên **trước** khi decode | `SweepResumeIntegrity` — 4 ca gồm mất một phần, đọc claim, không khớp chéo kịch bản |
| **R3** | Validator vẫn PASS khi xoá cả `docker_image` lẫn `nb.metadata.kaggle` | Validator đối chiếu digest với `knowledge/runtime.json`, kiểm hai nơi khai accelerator, python 3.12, và production phải gắn đúng sweep của kịch bản mình | `NotebookRuntimeContract` — 6 kiểu phá hợp đồng đều phải FAIL |
| **R4** | So sánh compile/fold chạy **ngoài** autocast, tức chứng nhận một precision mà run không hề dùng | `try_compile` so thêm **trong** autocast; tiêu chí nêu trước: fp16 không thể bit-equal nên yêu cầu **argmax không đổi** trên hàng có margin đủ rộng. `verify_fold(amp=True)` tương tự | **ĐÃ XÁC NHẬN trên T4 thật** (cả 3 sweep): `amp_disagreements: 0` ở mọi worker/vòng — xem §Kiểm chứng AMP |
| **R5** | Dự toán đĩa sweep chỉ tính **một** ứng viên trong khi output giữ **5** | Tính `per_run × len(MU_GRID)` + cache `train_index`/validation, in rõ thành phần | Số in ra ở mọi lần chạy sim notebook |
| **R6** | Nhiều chỗ diễn đạt vượt quá số đo | Xem "Đính chính" ngay dưới | — |

### Đính chính R6 (quan trọng khi đọc lại các số cũ)

- **`max_hours` không đồng nhất như đã viết.** Trước phiên 3, calibration là **10.0** còn sweep và
  production là 11.0; câu "cả 7 notebook đều 11.0" là **sai**. Nay đã thống nhất 11.0.
- **Cổng vào round đầu dùng ước lượng 60 giây.** `driver.run` đặt `worst_round = 0` mỗi lần gọi,
  nên round **đầu tiên** của mỗi phiên (và của mỗi ứng viên sweep) được cho phép chạy khi chỉ còn
  hơn 60 s — trong khi một round 100 client dài ~41 phút, tức có thể vượt `max_hours`. Đã sửa:
  nạp `logs/timing_*.json` của chính run đó, nếu chưa có thì dùng `expected_round_s` từ calibration.
  `max_hours` **không phải timeout cứng**, chỉ là cổng *vào* round.
- **`cm_proto_mismatch` bị mô tả sai** — xem lại ở bảng vmap phía trên.
- **Quan sát μ(k=1) không phải kết luận nhân quả** và **không** được dùng để thu hẹp grid.
- **763 s/round đo ở `eval_batch=16384`**, không phải 8192 hiện đang cấu hình.
- **`notebook_sim` thay chuỗi `eval_batch: 16384`** nên sau khi đổi sang 8192 phép thay không còn
  khớp; đã cập nhật, và thêm thay thế cho `expected_round_s`.
- **Suite có 6 regression tại thời điểm rà soát**, không tự động chứa các ca "4/4"/"12/12" từng
  chạy ad hoc ở phiên 2. Nay các ca đó đã thành test bền vững (bảng trên), suite chạy **17 test**.

---

## Kết quả sweep μ (2026-09-08) — CẢ BA KỊCH BẢN ĐÃ XONG

Chạy trên `odixe0502`, 2 × T4. Kết quả máy đọc được lưu ở
`review-logs/mu-sweep-result.json` và `review-logs/mu-sweep-50client-result.json`.
Cả **ba** đều `status: complete`, protocol **v1**, validation fraction 0.02 / seed 42,
đủ 5 ứng viên × 4 round. Kết quả máy đọc được: `review-logs/mu-sweep{,-50client,-100client}-result.json`.

### 20 client — thắng **k = 0.1**, μ = **7.585696e-07**

| k | round | proto f1_macro (cuối) | proto f1_macro (tốt nhất) | clf f1_macro | μ tuyệt đối |
|---|---|---|---|---|---|
| 0.03 | 4 | 0.6302 | 0.6302 | 0.5174 | 2.275709e-07 |
| **0.1** | 4 | **0.6331** | 0.6331 | 0.5165 | **7.585696e-07** |
| 0.3 | 4 | 0.6310 | 0.6323 | 0.5187 | 2.275709e-06 |
| 1.0 | 4 | 0.3420 | 0.5903 | 0.5025 | 7.585696e-06 |
| 3.0 | 4 | 0.2392 | 0.5878 | 0.3897 | 2.275709e-05 |

### 50 client — thắng **k = 0.3**, μ = **5.646573e-06**

| k | round | proto f1_macro (cuối) | proto f1_macro (tốt nhất) | clf f1_macro | μ tuyệt đối |
|---|---|---|---|---|---|
| 0.03 | 4 | 0.5021 | 0.5051 | 0.3995 | 5.646573e-07 |
| 0.1 | 4 | 0.5045 | 0.5045 | 0.4007 | 1.882191e-06 |
| **0.3** | 4 | **0.5162** | 0.5162 | 0.4074 | **5.646573e-06** |
| 1.0 | 4 | 0.2736 | 0.4830 | 0.4012 | 1.882191e-05 |
| 3.0 | 4 | 0.1926 | 0.4762 | 0.2598 | 5.646573e-05 |

### 100 client — thắng **k = 0.3**, μ = **1.126461e-05**

| k | round | proto f1_macro (cuối) | proto f1_macro (tốt nhất) | clf f1_macro | μ tuyệt đối |
|---|---|---|---|---|---|
| 0.03 | 4 | 0.4111 | 0.4184 | 0.3242 | 1.126461e-06 |
| 0.1 | 4 | 0.4139 | 0.4228 | 0.3273 | 3.754870e-06 |
| **0.3** | 4 | **0.4297** | 0.4326 | 0.3341 | **1.126461e-05** |
| 1.0 | 4 | 0.2462 | 0.4054 | 0.3371 | 3.754870e-05 |
| 3.0 | 4 | 0.1679 | 0.4021 | 0.2225 | 1.126461e-04 |

Đường cong người thắng (k=0.3): 0.4019 → 0.4326 → 0.4324 → 0.4297
(reg 0.0000 → 0.0975 → 0.0177 → 0.0124).

### Regularizer APS PHÂN KỲ khi k ≥ 1 — bằng chứng định lượng

Quỹ đạo `reg_loss` của kịch bản 100 client nói rõ điều mà hai kịch bản kia chỉ gợi ý:

| k | reg r2 | reg r3 | reg r4 | proto f1_macro r1 → r4 |
|---|---|---|---|---|
| 0.03 | 0.0336 | 0.0031 | 0.0023 | 0.4002 → 0.4111 |
| 0.1 | 0.0862 | 0.0085 | 0.0027 | 0.4020 → 0.4139 |
| 0.3 | 0.0975 | 0.0177 | 0.0124 | 0.4019 → 0.4297 |
| 1.0 | 0.2224 | 0.9556 | **13.0189** | 0.4007 → 0.2462 |
| 3.0 | 1.6772 | **258.7784** | **125175.0369** | 0.4021 → 0.1679 |

Ở k ≤ 0.3 reg loss **giảm dần** (prototype cục bộ hội tụ về prototype toàn cục đã scale).
Ở k ≥ 1 nó **tăng theo cấp số nhân** — μ·ĉ_G lớn tới mức mục tiêu regularizer kéo đặc trưng
ra xa vùng mà CE có thể bù lại. Đây **không** phải overfitting mà là mất ổn định số học.

**Hệ quả cho lần sau:** grid `[0.03, 0.1, 0.3, 1, 3]` đã bao trọn điểm gãy. Muốn tinh chỉnh thêm
thì mở rộng **xuống dưới hoặc chèn giữa** (0.03–0.3); mở rộng **lên trên 1 là lãng phí quota**.
Điều này cũng xác nhận (chứ không thay thế) quan sát 3 round ở calibration — khác nhau ở chỗ giờ
có grid đầy đủ trên validation, không phải một điểm đơn lẻ.

### So sánh ba kịch bản

| Kịch bản | k thắng | μ tuyệt đối | mean(n_ij) | val proto f1_macro |
|---|---|---|---|---|
| 20 client | 0.1 | 7.585696e-07 | 131,827.1 | 0.6331 |
| 50 client | 0.3 | 5.646573e-06 | 53,129.6 | 0.5162 |
| 100 client | 0.3 | 1.126461e-05 | 26,632.1 | 0.4297 |

`μ = k / mean(n_ij)`. **50 và 100 client trùng k nhưng μ tuyệt đối lệch 2×** — nên chuyển k
giữa các kịch bản vẫn là sai; production dùng μ tuyệt đối riêng của từng kịch bản.
f1 giảm dần theo số client (0.63 → 0.52 → 0.43) đúng như kỳ vọng: cùng tổng dữ liệu chia cho
nhiều client hơn thì mỗi client có ít dữ liệu hơn và non-IID nặng hơn.

### Đọc kết quả

- **Hai kịch bản chọn k KHÁC NHAU (0.1 vs 0.3).** Đây là bằng chứng trực tiếp cho quyết định
  sweep riêng từng kịch bản: dùng lại k của 20 client cho 50 client sẽ lấy giá trị **không tối ưu**.
- **k ≥ 1 sụp đổ ở cả hai kịch bản.** Cột "tốt nhất" cho thấy chúng từng đạt ~0.48–0.59 rồi tụt
  mạnh ở round cuối — regularizer dần lấn át. Điều này **nhất quán** với quan sát 3 round ở
  calibration, nhưng giờ mới có grid đầy đủ trên validation để nói được.
- Đường cong của hai người thắng đều lành: f1 tăng đơn điệu, reg loss giảm dần khi prototype hội tụ.
  20 client: 0.5864 → 0.6268 → 0.6270 → 0.6331 (reg 0.0000 → 0.0703 → 0.0054 → 0.0023).
  50 client: 0.4750 → 0.5120 → 0.5157 → 0.5162 (reg 0.0000 → 0.0905 → 0.0175 → 0.0125).
- **Các con số này là điểm trên validation 2% sau 4 round — KHÔNG phải kết quả nghiên cứu.**
  Kết quả nghiên cứu là 50 round trên test cố định, chưa chạy.

### Thời gian round THẬT của sweep (đo từ `logs/timing_*.json`, không phải suy ra)

| Kịch bản | round 1 | round ổn định | mỗi ứng viên | cả grid (5 ứng viên) |
|---|---|---|---|---|
| 20 client | 466 s | ~430 s | ~29 phút | 146 phút |
| 50 client | 578 s | ~482 s | ~34 phút | 169 phút |
| 100 client | **898 s** | **~750 s** | **53 phút** | **~4,4 h** |

Chi tiết 100 client (ứng viên k=0.03, từ `history.csv` + `timing_*.json`):

| round | tổng | train | eval | peak GPU | applied steps | skipped |
|---|---|---|---|---|---|---|
| 1 | 898,0 s | 746,5 s | 148,6 s | 3,61 GB | 164.831 | 3 |
| 2 | 771,8 s | 621,8 s | 147,1 s | 3,61 GB | 164.810 | 24 |
| 3 | 748,0 s | 598,2 s | 147,6 s | 3,61 GB | 164.781 | 53 |
| 4 | 750,1 s | 599,7 s | 147,6 s | 3,61 GB | 164.774 | 60 |

**Setup mỗi ứng viên chỉ ~3,6 phút** (`session_elapsed_s` 1120,4 − round 898,0 = 217 s), không
phải ~26 phút. Round 1 đắt hơn ~150 s là do **warm-up của `torch.compile`**, không phải setup.
Skipped steps 3–60 trên ~164.800 là GradScaler tự hiệu chỉnh — bình thường, đã nằm trong ngưỡng
warm-up cho phép.

> **ĐÍNH CHÍNH (quan trọng).** Bản trước của mục này ghi "setup mỗi ứng viên ~26 phút" và suy ra
> production 100 client ~36,2 h. **Cả hai đều sai.** Con số 26 phút đến từ việc đọc W&B qua một
> đối tượng `wandb.Api()` tạo một lần rồi dùng lại trong vòng lặp — nó **cache** và trả số liệu
> cũ, nên round 1 trông như kết thúc muộn hơn thực tế. Bài học: `wandb.Api()` phải được tạo
> **mới mỗi lần poll**, và **`logs/timing_*.json` trong artifact mới là nguồn có thẩm quyền** về
> thời gian, không phải W&B summary.

### Phép chiếu production 100 client (từ round ỔN ĐỊNH)

Tốc độ eval đo ở 100 client: `100 × 860.908 / 147,5 s` = **583.666 client-dòng/s**
(so với 617k đo ở 20 client — chi phí cố định mỗi client không co giãn tuyến tính).

| | Phép chiếu cũ (từ 20 client) | **Đo ở 100 client** |
|---|---|---|
| train/round | ~754 s | **599 s** (ổn định) |
| eval/round | 1.704 s | **1.844 s** |
| round | ~2.463 s | **~2.443 s (40,7 phút)** |
| 50 round | ~34,2 h | **~33,9 h** |
| số phiên ở `max_hours=11` | 4 | **4** |

Phép chiếu cũ hoá ra khá sát; nó **quá cao** ở phần train và **quá thấp** ở phần eval, hai sai số
gần như triệt tiêu nhau. Điều đó **không** biến nó thành phương pháp tốt — chỉ là may.
Tổng ngân sách nghiên cứu: **~72 h** (sweep ~7,4 h đã dùng + production ~64 h).

### Kiểm chứng artifact

- `verify_run.py --require-complete` trên **cả 15** ứng viên (3 kịch bản × 5 điểm lưới), **0 fail**:
  20 client **869 check/ứng viên**, 50 client **1.709 check/ứng viên**,
  100 client **3.109 check/ứng viên**. Toàn bộ 15 run báo `completed: 4 rounds`.
- Cổng μ production chạy trên tài liệu **thật**: chấp nhận đúng sweep của kịch bản mình
  (20client → 7.585696e-07, 50client → 5.646573e-06) và **từ chối** tài liệu của kịch bản kia.
  Đây là kiểm chứng end-to-end của R1 trên dữ liệu thật, không phải fixture.

### Kiểm chứng AMP trên T4 thật (đóng R4)

R4 yêu cầu compile/fold phải được chứng nhận **trong** autocast, đúng precision mà run dùng.
Bằng chứng lấy từ compile report của cả ba kernel sweep (`kaggle-mu-sweep-*-final.log`):

| Kịch bản | `amp_max_abs_dlogit` | `amp_decisive_rows` | `amp_disagreements` |
|---|---|---|---|
| 20 client | 4.883e-04 … 7.324e-04 | 497–502 | **0** |
| 50 client | 4.883e-04 … 7.324e-04 | 486–498 | **0** |
| 100 client | 4.883e-04 | 245–247 | **0** |

So sánh fp32 cùng lúc: `max_abs_dlogit` ~2,7e-07 … 3,9e-07, `disagreements: 0`.

Đọc kết quả: sai số fp16 đúng bằng bội của 2⁻¹¹ = 4,883e-04 — tức granularity của chính
định dạng, không phải lỗi kernel. Điều quan trọng là **argmax không đổi trên mọi hàng có
margin đủ rộng, ở cả ba kịch bản, mọi worker, mọi vòng**. Đây là điều kiện đã nêu **trước**
khi chạy, nên nó là kiểm chứng chứ không phải diễn giải hậu nghiệm.
`compile=True` + `amp=True` do đó an toàn cho production.

### Hạ tầng chéo tài khoản cho production song song (2026-09-08)

Yêu cầu: chạy 3 kịch bản đồng thời trên 4 tài khoản, có kịch bản phải chia 2–4 session, kết quả
cuối vẫn phải như chạy một mạch.

Đã phát hiện và xử lý **hai** chặn đường, cái thứ nhất không nằm trong ghi chép cũ:

1. Sweep μ nằm ở `odixe0502`; production ở tài khoản khác **không đọc được kernel output riêng
   tư** — và điều đó chặn ngay **session 1**, không phải chỉ ở ranh giới resume như ghi chú cũ
   nói. Giải bằng 3 dataset μ dùng chung + `--mu-dataset`.
2. `--mu-dataset` ban đầu định gộp 3 file vào 1 dataset. **Không làm được:** cell config glob
   đúng tên `mu_sweep.json`, mà `mu_sweep_progress.json` cũng có khoá `scenario` → nới glob
   thành `mu_sweep*.json` sẽ cho 2 match và `assert len(matches) == 1` nổ. Chuyển sang **một
   dataset mỗi kịch bản**, giữ nguyên glob tên chính xác.

| Kiểm chứng | Kết quả |
|---|---|
| Route A (cùng tài khoản, giữ nguyên đường cũ) | 7 notebook, **0 problem** |
| Route B (dataset μ, `--owner khanhmay0304`) | 7 notebook, **0 problem** |
| Resume chéo tài khoản (`minhtrit06` nối `khanhmay0304`) | **0 problem**; `kernel_sources` đúng 1 phần tử, `require_resume: True` |
| Guard `--resume-from` thiếu `--only` | báo lỗi, thoát khác 0 |
| Validator từ chối: 2 nguồn resume / mất resume / sweep sai kịch bản / mất dataset μ | **từ chối cả 4** |
| `review_regressions.py` | **20 test pass** (thêm 3 test chéo tài khoản) |
| 3 file `mu_sweep.json` trong dataset vs bản lưu | **giống hệt**, 3 chữ ký **khác nhau** |

Một lỗi tự gây trong lúc làm: check `require_resume` ban đầu grep thẳng file `.ipynb`. `.ipynb`
là JSON nên dấu nháy bị escape, substring không bao giờ khớp — validator báo sai. Đã chuyển sang
parse bằng `nbformat` rồi mới so. Chính validator mới bắt được lỗi này, không phải người đọc lại.

`max_hours` **không** nằm trong `FINGERPRINT_KEYS` (đổi mỗi session an toàn); `mu_value` **có**
(μ buộc giống nhau qua mọi session). Đã kiểm trực tiếp trong `src/ckpt.py`.

### Lỗi W&B trong production (phát hiện 2026-09-08 khi lập kế hoạch chạy song song)

`LAUNCH_CELL` của production gọi `D.run(cfg, paths, OUT_ROOT)` **thiếu `wandb_run=`**. Notebook
sweep có (`gen_notebooks.py` dòng ~853), production thì không. Hậu quả: production đăng nhập W&B,
in "logged in", rồi **không gửi một scalar nào**. Không lần chạy nào lộ ra vì tới giờ mới chỉ
chạy sweep — production 50 round chưa từng chạy.

Đã sửa: production `WANDB.init(id=cfg["run_name"], resume="allow", group=f"train-{scenario}")`.
`id` theo tên run để 4 session của 100 client nối thành một biểu đồ; `driver.run` log `step=rnd`
nên step đơn điệu qua ranh giới session. `notebook_sim tinyproto-fp-train-20client --rounds 2`
chạy hết, 2 round commit — xác nhận nhánh `WANDB is None` vẫn an toàn.

Bài học: "đã login W&B" **không** đồng nghĩa "đang stream". Phải kiểm chính lời gọi `D.run`.

### Thực hiện review P1–P6 + smoke test 2 tài khoản (2026-09-08)

**Regression: 31 test pass** (từ 20), peak **1.635 MiB**, tuần tự qua `run_local_checked.py`,
CPU, không Inductor/local CUDA.

| Mục | Test mới | Nội dung |
|---|---|---|
| P1 | `TelemetryNeverStopsTraining` | Logger ném lỗi ở round 1 → run vẫn đủ 2 round commit, đủ `timing_*.json`, và **chỉ gọi log một lần** (telemetry bị tắt chứ không thử lại mỗi round) |
| P3 | `SessionChainContract` | 6 biến thể metadata sai + s1-gắn-session-trước đều bị từ chối; bộ 7 file production thật pass |
| P5 | `AmpProbeGate` | Phản ví dụ của review (delta=1000, decisive=0) **bị chặn**; coverage thiếu bị chặn; probe fp16 thực tế vẫn pass |
| P6 | `SourceManifestContract` | Đổi một module hoặc đổi digest image đều bị bắt |

#### P5 — phát hiện thêm: phép kiểm argmax trước đây **bất khả kích hoạt**

Ngưỡng cũ `decisive = margin > 10*delta`. Muốn đảo argmax thì nhiễu phải ≥ margin/2, tức
`delta ≥ margin/2`. Thay vào: `margin > 10*delta ≥ 5*margin` — bất khả với margin dương. Nghĩa là
**mọi hàng bị đảo argmax đều tự động bị loại khỏi tập decisive**, và `disagreements` luôn bằng 0
vì lý do số học chứ không phải vì kernel đúng. Các log sweep báo `amp_disagreements: 0` vì thế
**không có giá trị chứng minh** như đã hiểu trước đây; cái thực sự bảo vệ là `allclose` fp32.

Đã đổi sang margin **cố định** `DECISIVE_MARGIN=0.01`, thêm trần `AMP_MAX_DELTA=0.05` và sàn
coverage `MIN_DECISIVE_FRAC=0.50`, tất cả chốt **trước** khi probe chạy. Logic tách ra hàm thuần
`probe_verdict()` để test được trên CPU bằng tensor cố ý sai — trước đây chỉ chạy được trên GPU,
tức chỉ quan sát được khi nó đã pass.

#### Smoke test 2 × T4 trên hai tài khoản chưa từng chứng minh

| Tài khoản | GPU | Dataset | Resume | Verifier |
|---|---|---|---|---|
| `khanhmay0304` | 2 × Tesla T4 sm_75, 14,6 GB, 40 SM | đọc được | round 2 → 3 | **335 check, 0 fail** |
| `minhtrit06` | 2 × Tesla T4 sm_75, 14,6 GB, 40 SM | đọc được | round 2 → 3 | **335 check, 0 fail** |

Mỗi lần ~0,10 h quota. Log: `review-logs/kaggle-smoke-*.log`. Bốn dataset VeReMi **vốn đã public**
(kiểm bằng tìm kiếm không cờ `-m`), nên phần P2 về quyền đọc dataset khép lại mà không cần share.

#### Đã push session 1 của cả ba kịch bản

`khanhmay0304/…-100client-s1`, `minhtran0601/…-20client-s1`, `odixe0502/…-50client-s1`.
Cả ba lên W&B ngay (`state=running`), xác nhận `wandb_run=wb` + sửa P1 hoạt động trong production
— trước đó production login W&B nhưng không stream gì.

### Production session 1 — round đầu của cả ba kịch bản (2026-09-08, 21:31)

| Kịch bản | round | proto_f1 | clf_f1 | ce | reg | s/round |
|---|---:|---:|---:|---:|---:|---:|
| 20 client | 5 | 0,6292 | 0,5184 | 0,0450 | 0,001722 | **738–739** |
| 50 client | 2 | 0,5225 | 0,4225 | 0,0841 | 0,089788 | **1.317** |
| 100 client | 1 | 0,4161 | 0,3397 | 0,1887 | 0,000000 | **2.405,9** |

**`reg` hội tụ, không phân kỳ.** 20c: 0,0723 → 0,0055 → 0,0024 → 0,0017 qua round 2→5. Đây là
bằng chứng trực tiếp rằng winner của giao thức v1 (k = 0,1) ổn định ở thang dài, chứ không chỉ
trong 4 round của sweep — và ngược hẳn với k ≥ 1 (bùng lên 13,0 rồi 125.175). `reg = 0` ở round 1
là **đúng**: chưa có prototype toàn cục nên chưa có gì để phạt.

**μ chọn trên validation chuyển được sang test cố định.** Sweep 20c round 4 (validation) 0,6331
so với production round 4 (test cố định) 0,6284; 50c sweep 0,5162 so với production round 2 đã
0,5225. Không thấy dấu hiệu μ chỉ hợp với tập validation của sweep.

**Số đo thay cho phép chiếu.** Ngoại suy tuyến tính theo số client (từ 739 s ở 20c và 1.317 s ở
50c) đoán 100c = 2.280 s; thực đo round 1 là **2.405,9 s**, lệch **5,5%**. Ngoại suy dùng tạm
được để lập kế hoạch, **không** dùng để chốt ngân sách.

**Độ trễ push → round 1 lớn hơn "setup" trong kernel.** 20c mất ~33 phút từ lúc push tới khi
round 1 lên W&B, trong đó round chỉ 13 phút → ~20 phút là **queue Kaggle + decode dữ liệu**.
Con số "setup 3,6 phút" đo ở sweep là setup **trong** kernel; không dùng nó để đoán thời điểm có
kết quả. 100c mất ~86 phút tới round 1.

### Thu hẹp phạm vi source manifest (P6)

Bản đầu ghim cả `gen_production.py` và `validate_notebooks.py`. Sai phạm vi: manifest trả lời câu
"session N có chạy **cùng thuật toán** với session 1 không?", mà hai file đó không thể đổi câu trả
lời — `gen_production.py` chỉ chọn owner/`max_hours`/chuỗi session (không nằm trong fingerprint),
`validate_notebooks.py` chỉ kiểm chứ không chạy. Ghim chúng biến mọi lần chỉnh ngân sách hợp lệ
(chính review cho phép) thành báo động giả về "đổi thuật toán".

Đã thu về **15 file**: `src/*.py`, `gen_notebooks.py` (nó nhúng `src/` vào notebook session sau và
dựng CFG), ba JSON μ, cùng digest image. Lần verify khi đang chạy xác nhận đúng một file đổi là
`gen_production.py` — tức `src/` và μ **y nguyên** từ lúc run bắt đầu.

**Fixture `/tmp/tinyproto_fixture` mất khi phiên khởi động lại** (nằm ở `/tmp`, không phải
scratchpad) làm 8 test lỗi — **không phải regression**. Dựng lại bằng `tests/make_fixture.py` rồi
**31 test pass**, peak 1.629 MiB. Test P1 mới có guard `skipTest` nên chỉ skip; các test cũ không
có guard nên lỗi thẳng.

---

## Session 1 kết thúc — pull, verify, ghép (2026-09-09)

### `verify_run.py` trên artifact đã kéo về

| Run | Round | Check | Fail | Ghi chú |
|---|---:|---:|---:|---|
| `tinyproto_fp_20client` (pull thô) | 50/50 | 10.573 | 0 | `--require-complete` pass |
| `tinyproto_fp_20client` (**sau khi ghép**) | 50/50 | **10.573** | 0 | cùng số check → ghép không mất gì |
| `tinyproto_fp_50client` (s1) | 31/50 | 13.048 | 0 | `stopped_early: wall_clock_budget` |
| `tinyproto_fp_100client` (s1) | 16/50 | 12.348 | 0 | `stopped_early: wall_clock_budget` |

`source_manifest.py verify`: **manifest OK, 15 file + runtime image khớp**.
Regression sau mọi thay đổi phiên này: **31 test pass**, peak **1.594 MiB**.

### Cross-check W&B ↔ artifact

W&B `proto/f1_macro` round 1 của 100 client = 0,416139; `metrics/round_001.json`
`aggregate.proto.mean_over_clients.f1_macro` = **0,416138722036093**. Khớp. Ba `mu_value` trong
`config.json` khớp đúng bảng §5b của CONTEXT (20c 7,585696e-07 · 50c 5,646573e-06 ·
100c 1,126461e-05) → cổng μ đã làm đúng việc trên cả ba tài khoản.

### Timing (`logs/timing_*.json`, nguồn có thẩm quyền)

| Kịch bản | round | mean 8 cuối (s) | worst (s) | overhead trước r1 (s) | elapsed (h) | quota tiêu (h) |
|---|---:|---:|---:|---:|---:|---:|
| 20 client | 50 | 788,5 | 816,3 | 172,9 | 10,589 | 10,60 |
| 50 client | 31 | 1.234,7 | 1.343,1 | 209,8 | 10,851 | 10,86 |
| 100 client | 16 | 2.336,3 | 2.411,9 | 188,4 | 10,418 | 10,42 |

**Đính chính hai con số đã báo cáo ở lần theo dõi round đầu:**

1. Tôi đã báo 100c steady ≈ **2.370 s** (suy từ round 1 trừ warm-up). Đo thật trên 16 round:
   **2.336,3 s** (mean 8 round cuối). 50 round ≈ **32,4 h**, không phải 32,9 h.
2. Tôi đã báo 20c **739 s/round** từ 5 round đầu. Trên đủ 50 round, round chậm dần:
   738 s ở round 5 → **788,5 s** trung bình 8 round cuối, max 816,3. Tổng 10,589 h chứ không
   phải 10,26 h. Ngoại suy từ vài round đầu **thiếu ~3%** cho một run 50 round.

### Ba số đo bác bỏ giả định cũ

1. **`elapsed` khớp quota tiêu thụ tới 0,01 h ở cả ba kịch bản.** Trước đó tôi đã lo phần queue
   Kaggle + decode nằm ngoài `timing_*.json` sẽ ăn thêm quota. Không: quota chỉ tính từ lúc
   kernel chạy, nên `max_hours` là thước đo quota dùng được trực tiếp.
2. **`save_preds_rounds=[50]` là đúng thiết kế.** `preds/` của 50c và 100c chỉ có `y_true.npy`;
   ban đầu tôi tưởng thiếu file. 20 client (đủ 50 round) có `round_050_proto.npy` +
   `round_050_clf.npy` như mong đợi.
3. **`merge_sessions.py` của skill ghép sai layout dự án này** — xem CONTEXT §9. Phép kiểm
   liền mạch 0-index tình cờ abort trước khi kịp báo thành công giả; nếu chỉ sửa lỗi đó thôi thì
   đã có một thư mục "run hoàn chỉnh" **không chứa trọng số nào**. Cả hai lỗi đã sửa ở `.agents/`
   và `.claude/`, và phép ghép 20 client sau khi sửa verify lại đúng **10.573 check / 0 fail**.

---

## Probe bàn giao chéo tài khoản + nghiệm thu resume session 2 (2026-09-09)

### Probe: `kernel_sources` chéo tài khoản — KHÔNG hoạt động

`minhtrit06/tinyproto-fp-handoff-probe` (CPU, 0 quota GPU) gắn
`khanhmay0304/tinyproto-fp-train-100client-s1`. Push in ra `not valid kernel sources`, **vẫn báo
thành công**, kernel chạy và `/kaggle/input` **rỗng** — probe liệt kê 0 mục rồi `SystemExit`.
Kết luận ở CONTEXT §10. Đây là cấu hình từng nằm trong `PLAN` cho s3→s4.

### Nghiệm thu resume — cả hai đường

| Kịch bản | Đường | round cuối s1 (proto / reg) | round đầu s2 (proto / reg) |
|---|---|---|---|
| 50 client | kernel output, cùng tài khoản | r31 0,441924 / 0,045055 | **r32** 0,442666 / 0,049703 |
| 100 client | **dataset**, chéo tài khoản | r16 0,408794 / 0,013400 | **r17** 0,406100 / 0,014246 |

Phép kiểm có sức phân biệt là **`reg` khác 0**: round 1 của một run mới luôn có `reg = 0` vì chưa
có prototype toàn cục. Cả hai session 2 đều có `reg` nối tiếp giá trị cuối của s1 và tiếp tục xu
hướng tăng, nên đây là resume thật chứ không phải khởi động lại.

### Đo overhead import — bác bỏ điều tôi đã báo cáo

Tôi đã báo "import dataset 2,5 GB mất ~33 phút". **Sai.** Đo bằng `_timestamp` của W&B trừ
`startedAt` trong `wandb-metadata.json`:

| Đường | overhead trước round đầu |
|---|---:|
| kernel output, cùng tài khoản (50c) | **230 s** |
| dataset 2,5 GB, chéo tài khoản (100c) | **232 s** |

Hai đường nhanh như nhau. Cái tôi đo nhầm là **độ trễ history của W&B**: kernel 100c `startedAt`
04:18:35, round 17 xong ở `_timestamp` 05:04, nhưng API trả `last_step=16` tới tận ~05:45. Lúc
06:19 độ trễ còn lớn hơn — 50c log gần nhất **44 phút trước**, 100c **75 phút trước**, trong khi cả
hai kernel đều `RUNNING` và nhịp round đều đặn.

**Hai quy tắc rút ra, đã ghi vào skill (`wandb.md`, `multi-account.md`):**

1. Sống/chết lấy từ `kaggle kernels status`, **không** từ `state`/`heartbeatAt` của W&B — một run
   `resume="allow"` giữ nguyên trạng thái kết thúc của session trước cho tới khi log được gì đó.
   Lúc 04:14 W&B báo 50c `crashed` với heartbeat 2 giây sau `startedAt`; kernel thực ra `RUNNING`.
2. Thời gian lấy từ `_timestamp` (khi đang chạy) hoặc `logs/timing_*.json` (sau khi pull), **không**
   từ lúc con số xuất hiện trên W&B.

### Round của session 2 chậm hơn session 1

50 client: s1 ổn định **1.235 s**, s2 ổn định **1.387 s** — **+12%**. Chiếu ngân sách bằng số của
s1 cho ra 6,6 h ("thoải mái"); bằng số thật của s2 cho ra **7,41 h** trên quota 8,06 h, và biên tới
ngưỡng break ở round 49 chỉ còn ~1.455 s ≈ một round. **Ngân sách phải tính bằng số đo của chính
session đang chạy.** Dự báo: 50c xong trọn 50 round ngay trong s2; 100c s2 dừng ở round 32.

### Regression

**34 test pass**, peak 1.644 MiB. `CrossAccountResumeContract` đã được viết lại vì nó
**khẳng định đúng hành vi mà probe vừa bác bỏ**; thêm 3 test: đường dataset hợp lệ, kernel source
chéo tài khoản bị chặn, hai đường loại trừ nhau.

---

## Session 2: 50 client HOÀN THÀNH, 100 client tới round 32 (2026-09-10)

### `verify_run.py`

| Run | Round | Check | Fail |
|---|---:|---:|---:|
| `tinyproto_fp_50client` s2 (pull thô) | 50/50 | 21.133 | 0 |
| `tinyproto_fp_50client` (**đã ghép s1+s2**) | 50/50 | **21.133** | 0 |
| `tinyproto_fp_100client` s2 (pull thô) | 32/50 | 24.668 | 0 |

Cả hai đều `--require-complete` với 50 client. `source_manifest.py verify` OK trước khi push s3.

### Ghép 50 client = bằng chứng resume qua 31 round chồng lấn

`merge_sessions.py` thoát lỗi nếu **bất kỳ** artifact nào có ở cả hai pull mà khác byte — nghĩa là
session sau đã train lại thay vì tiếp tục. Ghép 50 client: **không một khác biệt nào**,
`new rounds per session: [31, 19]`, round 1..50 liền mạch.

### Bàn giao bằng dataset giữ nguyên byte — chứng minh không cần đọc lại 2,8 GB

`complete/round_NNN.done` mang manifest sha256 của **mọi** artifact mà round đó chứng nhận. Cả hai
pull 100 client (s1 và s2) đều đã pass `validate_integrity`, nên **marker giống nhau ⇒ byte giống
nhau**. Đối chiếu marker rounds 1..16 giữa `_pull_100client-s1` và `_pull_100client-s2`:
**16/16 giống hệt, 145 artifact được chứng nhận, 0 khác biệt**.

→ Chuỗi `s1 (khanhmay0304) → dataset → s2 (minhtrit06)` bảo toàn chính xác từng byte của 16 round.

### Ba lỗi vận hành (chi tiết ở CONTEXT §11)

1. Pull thiếu file **và** để lại file 0 byte; đếm theo tên file bỏ sót file hỏng.
   `find <pull>/runs -type f -size 0 -delete` rồi pull lại với `-o --file-pattern`.
   Giới hạn vào `runs/`: `proj/__init__.py` rỗng là hợp lệ, không phải rác.
2. Run **hoàn chỉnh** vẫn mang `stopped_early.json` của session trước (50/50 round nhưng file ghi
   `last_round: 31`). Chỉ `verify_run.py --require-complete` mới trả lời được câu hỏi này.
3. `merge_sessions.py` ghép `resume/` theo hợp → `verify_run.py` bắt
   `superseded resume blobs were not pruned`. Đã sửa: chỉ lấy `resume/` của session cuối.
   **Đây là verifier làm đúng việc của nó** — lỗi nằm ở phép ghép, không ở run.

### Độ chính xác của dự báo ngân sách

| Dự báo | Thực tế |
|---|---|
| 50c s2 xong trọn 50 round, elapsed 7,41 h | **xong 50 round**, quota tiêu **7,40 h** |
| 100c s2 dừng ở round 32, elapsed 10,50 h | **dừng ở round 32**, elapsed **10,57 h** |

Dự báo dùng steady s/round của **chính session đang chạy** (không phải của s1) và mô phỏng đúng
cổng ngân sách `elapsed + 1,15 × worst_round >= max_hours*3600`.

---

## Session 3 + 4 — 100 client HOÀN THÀNH, ghép 4 phiên, báo cáo (2026-09-11)

### Pull + verify

| Pull | Round | Kích thước | Kiểm size>0 | `verify_run.py` |
|---|---|---:|---|---|
| `_pull_100client-s3` (minhtrit06) | 1–49 | 7,7 GB | 49/49 mọi thư mục, 0 file 0 byte (trừ `proj/__init__.py`) | **37.758 / 0** |
| `_pull_100client-s4` (minhtran0601), lần 1 | — | 3,2 GB | weights **5**/50, `round_006.pt` **0 byte**, 7–50 thiếu; các thư mục khác 50/50 | — |
| `_pull_100client-s4`, sau `-o --file-pattern` | 1–50 | 9,9 GB | 50/50, 0 byte = 0 | **38.733 / 0** `--require-complete` |
| **ghép s1+s2+s3+s4** → `runs/tinyproto_fp_100client/` | 1–50 | 9,9 GB | `new rounds per session: [16, 16, 17, 1]`, 97 round chồng lấn 0 khác biệt | **38.733 / 0** `--require-complete` |

`stopped_early.json` trong s4 ghi `last_round: 49` (của s3) dù s4 chạy trọn — bẫy đã biết,
verify là thẩm quyền.

### Resume ở cả 4 ranh giới: `reg` nối tiếp, không round nào chạy lại

r16 0,013400 → r17 0,014246 (dataset) · r32 0,065881 → r33 0,075387 (kernel output) ·
r49 0,927422 → r50 1,097369 (dataset). Log s4: `[resume] imported 49 completed rounds from
/kaggle/input/datasets/minhtran0601/tinyproto-fp-resume-100client-r49/…`, `starting at round
50 of 50`, workers ready t=313 s.

### Dataset bàn giao s3→s4

`minhtran0601/tinyproto-fp-resume-100client-r49`: `-r zip -t`, 7,09 GB upload 11 m 14 s,
`ready` sau ~4 phút; **353/353 file khớp kích thước từng byte** với local (0 lệch, 0 thừa, 0
thiếu). Kaggle cắt một cấp `runs/` như đã ghi ở §10.

### W&B khớp artifact

Round 49 `train/grad_norm` trên W&B = 6,2732686998 — file `history.csv` = 6,2733. Round 50 chưa
lên W&B lúc kiểm (lag của run resume), số round 50 lấy từ artifact.

### Kết quả 100 client (mean qua 100 client, fixed test 10.761.343 dòng)

| | round đỉnh | round 50 |
|---|---:|---:|
| proto f1_macro | 0,4474 (r2) | **0,3435** (−23,2%) |
| clf f1_macro | 0,3529 | 0,3289 |
| proto − clf | +0,0945 | +0,0146 |
| reg_loss | đáy 0,010512 (r9) | **1,097369** (104×) |
| grad_norm | đáy 0,2514 (r14) | **7,2834** (29×) |
| train_loss | đáy 0,006650 (r24) | 0,017671 (2,7×) |

### Kiểm chứng cơ chế phân kỳ (τ) — chi tiết ở `report.md` §7

τ_j = ‖μĉ_G[j]⊙m_j‖ / mean‖ĉ_L[i,j]⊙m_j‖ đo ở round 1 từ `protos/round_001.pt`:

| | τ_max | lớp τ>1 | norm r1→r50 hai lớp đó | lớp τ kế tiếp |
|---|---:|---|---|---|
| 20c | 0,420 | 0 | 0,42× / 0,41× | 0,65× |
| 50c | 1,283 | `trafficCongestionSybil`, `benign` | 20,92× / 17,55× | 1,08× |
| 100c | 1,271 | cùng hai lớp | 31,68× / 27,13× | 1,48× |

Recall `proto` round đỉnh→50: `trafficCongestionSybil` rơi mạnh nhất ở cả 50c (0,729→0,446) và
100c (0,658→0,380); 11/16 lớp mất >0,05 ở mỗi kịch bản phân kỳ; ở 20c chỉ **một** lớp
(`trafficCongestionSybil`, τ cao nhất) mất >0,05. Sweep 4 round: k=1,0 / 3,0 có reg r4 lớn hơn
nhóm ổn định 7 bậc (đã thấy phân kỳ nhanh), k=0,3 thắng k=0,1 ở 50c/100c, thua 0,0021 ở 20c.

### Bộ test

Fixture `/tmp/tinyproto_fixture` mất sau crash → 8 lỗi (`MuProtocolContract`, `ReviewRegression`);
13/13 contract test về notebook/PLAN pass ngay; dựng lại fixture → **34/34**.

### Báo cáo

`report.md` (gốc repo), 1.689 dòng, 23 bảng spliced bởi `make_report_tables.py --splice`
(1 m 30 s, gồm chạy `verify_run.py` cho cả ba run). CSV: `report/per_client_{20,50,100}client.csv`
(2.000 / 5.000 / 10.000 hàng), `mean_over_clients_*.csv` (100 hàng mỗi file), `summary.json`.
