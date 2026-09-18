# Bàn giao dự án AFPHA–DAGSNet

> **📌 Note 2026-09-13 (chủ dự án, ghi từ phiên `~/nckh/pfedes`): thống nhất MỘT lịch learning rate cho mọi dự án anh em.**
> Lịch: AdamW, **cosine theo round** `lr(t) = lr_min + (lr − lr_min)/2·(1 + cos(π(t−1)/(T−1)))`, **lr = 1e-3, lr_min = 1e-5, T = 50**,
> hằng trong một round (công thức `afpha`; bản tham chiếu `~/nckh/pfedes/papers/pfedes-yi-2025/proj/pfedes.py::lr_at`).
> pFedES đã chạy lại với lịch này (kernel `*-cos`, 13-09). **Dự án này CHƯA sinh/push notebook mới** — chỉ làm khi chủ dự án yêu cầu;
> quota tuần 13→19-09 đã dành cho pFedES (~92 h). Chỗ cần sửa khi làm: `src/afpha.py` (`LR_MAX`/`LR_MIN`, `lr_for_round`) — đã là cosine 1e-3→1e-5 T=50 ⇒ **đã khớp**, không cần đổi giá trị.
> Lý do và bằng chứng (đỉnh sớm sau ~2 epoch local, cosine chặn đà trôi nhưng không kéo lại đỉnh): `~/nckh/pfedes/CONTEXT.md` §12.

**Cập nhật: 2026-09-07. Thư mục: /home/odixe/nckh/afpha.**

**Trạng thái 2026-09-08: CẢ BA KỊCH BẢN ĐÃ HOÀN THÀNH 50/50 round, verifier 10/10 mỗi run,
report đã viết ở [report.md](report.md).** R1–R5 của mục 14 đã sửa và có test tiêm lỗi chứng
minh (mục 14.1–14.3). Kết quả và phân tích ở mục 16.
Đặc tả đã chốt; 9 review finding của 2026-09-06 đã có bản sửa và kiểm chứng luồng thông thường
(mục 12), kèm 7 lỗi
mới phát hiện trong lúc sửa (mục 8). **Calibration đã chạy thật trên T4×2 và COMPLETE**
(mục 6): 3 round trên toàn bộ 43M dòng thật, 2 GPU, full test 10,76M dòng, accuracy
0,45 → 0,77 → 0,81. Ngân sách đã chốt từ số đo thật, `MAX_HOURS` = 6,0/6,0/9,5.
4 notebook đã build lại và validate 4/4 bằng `scripts/validate_notebooks.py`.

**W&B trên Kaggle — mắt xích này đã đóng (2026-09-07), và nó ĐÚNG LÀ hỏng thật.**
Version 1 của notebook 100 client ERROR sau **27,7 s** quota với
`UsageError: No API key configured`: cell đặt `os.environ["WANDB_API_KEY"]` rồi gọi
`wandb.login(relogin=True)`, mà `relogin=True` bỏ qua biến môi trường (chi tiết mục 8 điểm 11).
Ở local luôn pass vì `~/.netrc` che mất lỗi. Thiết kế fail-fast hoạt động đúng như thiết kế:
chết **trước** prepack nên tốn 27,7 s thay vì nhiều giờ. Đã sửa bằng cách truyền `key=` tường
minh, **tái hiện được lỗi ở local trước** (HOME tạm rỗng, không netrc, không terminal) rồi mới
push lại — không push mù. Version 2 xác thực W&B thành công và tạo được run.
`import wandb` trên image Kaggle bình thường; nhánh phòng vệ `pip install` vẫn giữ.

## 1. Cách làm việc với người dùng

Đọc [AGENTS.md](AGENTS.md) trước khi sửa code. Người dùng giao tiếp bằng tiếng Việt,
yêu cầu hỏi rõ và chốt những điểm chưa chắc trước khi triển khai; việc nào phải làm
thủ công thì nêu cụ thể. Không hỏi lại các quyết định đã xác nhận dưới đây.

Làm thay đổi tối thiểu, không sửa lan sang phần không liên quan. Hoàn thành các việc
độc lập đã được giao trong khi chờ câu trả lời. Không tự spawn sub-agent nếu không
có chỉ dẫn cho phép áp dụng. Không gửi token vào chat/log, không tự chuyển tài khoản
Kaggle, không launch training chỉ vì người dùng yêu cầu viết script.

Python và Kaggle CLI luôn chạy trong conda env `nckh`. `python3` hệ thống không có numpy;
mọi script phải chạy bằng `conda run -n nckh python ...`.

## 2. Quyết định đã chốt

| Thiết lập | Đã xác nhận |
|---|---|
| Kịch bản | 20, 50, 100 client |
| Round | 50, đánh số 1–50 |
| Local epoch | 1 lượt đầy đủ trên dữ liệu mỗi client, mỗi round |
| Batch mỗi client | 20/50 client: 512; 100 client: 256 |
| Khởi tạo | Train từ đầu, mặc định PyTorch; không warm-start |
| Client và server | Cùng kiến trúc DAGSNet |
| Đánh giá | Chỉ global sau aggregation, toàn bộ test mỗi round |
| Metrics | Bộ 10 metrics dưới đây |
| Checkpoint | Lưu state_dict, không serialize nguyên model |
| Phần cứng | Kaggle Tesla T4 ×2 |
| **Đặc tả AFPHA** | **Đề xuất A trong rebuild.md, chốt nguyên 2026-09-06** |
| **Tài khoản chạy** | **minhtran0601** (dataset của odixe0502 là public, attach được) |
| **Calibration** | **Chạy 1 notebook calibration trên T4×2 trước khi launch** |
| **Giám sát** | **Bắt buộc dùng W&B** |
| **Cách launch** | **Chạy đồng thời cả 3 notebook train, theo dõi song song** |

```text
accuracy
precision_macro  precision_micro  precision_weighted
recall_macro     recall_micro     recall_weighted
f1_macro         f1_micro         f1_weighted
```

## 3. Dữ liệu

| Kịch bản | Root local | Kaggle |
|---|---|---|
| 20 | dataset/fl_client/alpha05/20_client | odixe0502/veremi-fl-20client |
| 50 | dataset/fl_client/alpha05/50_client | odixe0502/veremi-fl-50client |
| 100 | dataset/fl_client/alpha05/100_client | odixe0502/veremi-fl-100client |
| Test | dataset/centralized/test | odixe0502/veremi-nextgen2026-centralized |

Đã xác nhận qua MCP ngày 2026-09-06: cả 4 dataset version 1, Ready. Layout dự kiến dạng phẳng:
`/kaggle/input/veremi-fl-20client/20_client/train/client_id=NNN/part-*.parquet` và
`/kaggle/input/veremi-nextgen2026-centralized/upload/test/part-*.parquet`.
`fldata.find_root` dò sentinel tới **độ sâu 5** và bắt buộc đúng một match. Đã test bằng
fixture: layout phẳng **và** layout có prefix `datasets/<owner>/` đều resolve đúng; thiếu
sentinel hoặc nhiều match đều bị từ chối. Mount runtime thật được xác nhận trong calibration.

**Train đã z-score; test còn thô.** Chỉ 66 cột `f_*` đúng thứ tự trong
architecture/meta.json; scaler train áp cho test, không fit lại. Nhãn 0–15.
Parquet có 87 cột (66 `f_*` + `label` int8 + 20 cột bị loại).

Số đo từ client_stats.json (dùng để tính ngân sách, xem mục 5):

| Kịch bản | rows | client nhỏ nhất | lớn nhất | batch | **step/round** |
|---|---:|---:|---:|---:|---:|
| 20 | 43.045.415 | 870.217 | 5.890.990 | 512 | **84.083** |
| 50 | 43.045.415 | 199.063 | 2.630.929 | 512 | **84.098** |
| 100 | 43.045.415 | 98.180 | 1.333.839 | 256 | **168.200** |

Test 10.761.343 dòng, đủ 16 lớp. Audit full đã chạy trước đó
([audit.json](papers/khan-2025-afpha/audit.json)); không cần chạy lại.

Caveat khoa học phải giữ trong mọi báo cáo: client là receiver unit chứ không phải một xe;
scaler fit trên toàn bộ train trước FL nên không tuyên bố preprocessing phân tán; split
theo thời gian chứ không tách theo xe; điểm global test không phải personalized; không so
trực tiếp với số CAN/CIC-IDS của bài báo; feature lưu ở fp16 (mục 6).

## 4. Đặc tả AFPHA đã chốt — Đề xuất A

Bài báo mục 4.6 chỉ mô tả AFPHA bằng lời (FedAvg + FedProx + HFL). **Mọi con số dưới đây là
lựa chọn triển khai, không phải hyperparameter công bố của tác giả. Không gọi đây là exact
reproduction.** Nguồn đầy đủ: [rebuild.md](papers/khan-2025-afpha/rebuild.md).

- 100% client tham gia; cụm cố định 5 client → 4/10/20 cụm; hoán vị `default_rng(42)`.
- Aggregation sample-weighted trong cụm rồi sample-weighted ở server.
- Local loss `CE + (mu_i/2)·||w − w_t||²` trên learnable parameters.
- `mu` round 1 = 0.01; `mu_i,t+1 = 0.01·(1 + d_i/(d_i + d̄ + 1e-12))`, `d̄` weighted theo `n_i/N`.
- Adam (0.9, 0.999), eps 1e-8, wd 0, reset mỗi client mỗi round.
- LR cosine `1e-5 + (1e-3−1e-5)/2·(1 + cos(π(t−1)/49))`, cố định trong round.
- Clip global grad norm 1.0, sau unscale và sau khi cộng proximal gradient.
- BN `running_mean/var` trung bình theo `n_i/N`; `num_batches_tracked` lấy max.
- Không class weights, không resampling.

**Hai điều phải nói kèm:** hai cấp sample-weighted bằng FedAvg về đại số (đã kiểm chứng số,
mục 7) — phân cấp là mô tả hệ thống chứ không phải khác biệt tối ưu hoá; và trung bình
`running_var` là quy ước hợp nhất buffer, không phải pooled variance.

## 5. Code đã viết

```
src/dagsnet.py     model, 395.024 params, chép nguyên từ ARCHITECTURE.md §6
src/metrics.py     10 metrics tính từ confusion matrix
src/flatpack.py    state_dict <-> flat vector (params đứng đầu)
src/afpha.py       cụm, LR, adaptive mu, aggregation phân cấp, drift
src/fldata.py      find_root (depth 5), Schema, scan_counts, parquet -> fp16, scaler cho test
src/worker.py      1 process/GPU: resident data, compile+validate, train_client, evaluate
src/fl_train.py    driver: scan -> fingerprint -> import/resume -> prepack -> round loop
                   -> aggregate -> eval -> commit; guard client, budget, W&B
scripts/build_notebooks.py  sinh 4 notebook (staging dir + kernel-metadata) ; nhúng key W&B
scripts/verify_run.py       9 contract check, đọc config.json, mọi round, mọi kịch bản
scripts/make_fixture.py     fixture streaming (bounded RAM), 3 kịch bản + mount lồng
notebooks/afpha-dagsnet-calibration/ , afpha-dagsnet-train-{20,50,100}client/
```

Notebook nhúng nguyên văn src/*.py + meta.json + scaler.json, nên thứ chạy trên Kaggle
đúng bằng thứ đã test local. **Sửa src/ thì phải chạy lại build_notebooks.py.**

Thiết kế tốc độ: dữ liệu train cả kịch bản nằm thường trú trên VRAM từng GPU dưới dạng
fp16 (5,68 GB), không có DataLoader, không copy PCIe trong vòng train; hai worker
persistent nhận client động theo thứ tự lớn-trước; aggregation chạy ở process cha theo
thứ tự cụm cố định. Chi tiết và số đo: [perf-federated.md](.agents/skills/kaggle-training-notebook/references/perf-federated.md).

Guard đã có: client bị **từ chối** (abort round, giữ nguyên round đã commit) nếu trọng số
không hữu hạn, loss/grad-norm **trên các step được áp dụng** không hữu hạn, mọi step bị AMP
bỏ, hoặc số step bỏ vượt `max(8 warm-up, 5%)`. Không bao giờ âm thầm bỏ client — đặc tả
chốt participation 100%.

## 6. Benchmark đã đo trên T4×2 (calibration thật, 2026-09-07)

Nguồn: kernel `minhtran0601/afpha-dagsnet-calibration`, COMPLETE, tốn **1489 s** quota
(0,41 h). Torch 2.10.0+cu128, CUDA 12.8, 2× Tesla T4 capability (7,5) — hardware gate pass.

### 6.1 Micro-benchmark tổng hợp

| batch | mode | AMP | ms/step | mẫu/s | peak GiB |
|---:|---|---|---:|---:|---:|
| 512 | eager | có | 18,524 | 27.640 | 0,377 |
| 512 | default | có | 14,903 | 34.355 | 0,091 |
| **512** | **reduce-overhead** | **có** | **6,487** | **78.933** | **0,081** |
| 512 | reduce-overhead | không | 10,075 | 50.818 | 0,384 |
| 256 | eager | có | 19,109 | 13.397 | 0,213 |
| 256 | default | có | 14,813 | 17.282 | 0,072 |
| **256** | **reduce-overhead** | **có** | **6,028** | **42.469** | **0,062** |
| 256 | reduce-overhead | không | 5,767 | 44.392 | 0,211 |

`reduce-overhead` nhanh **2,86×** so với eager ở batch 512 trên T4 — cùng kết luận với 3050,
khác con số vì baseline khác. Nó vẫn là quyết định tối ưu quan trọng nhất của dự án.

**Phát hiện mới, trái với 3050:** ở **batch 256** fp32 **nhanh hơn** AMP (5,767 vs 6,028 ms,
tức 4,3%); ở batch 512 AMP vẫn thắng rõ (6,487 vs 10,075). Batch 256 chính là kịch bản
100 client. Đề nghị **vẫn giữ AMP cho cả ba**: 4,3% tương đương ~0,3 h, không đáng để một
kịch bản chạy khác numerics với hai kịch bản còn lại khi cả ba sẽ được đem ra so sánh với
nhau. AMP=False được đo để ghi nhận, không phải candidate (đúng review finding #7).

### 6.2 Chạy thật: 3 round, 20 client, 43.045.415 dòng, 2 GPU, full test 10.761.343 dòng

| | round 1 | round 2 | round 3 |
|---|---:|---:|---:|
| train (s) | 340,1 | 304,6 | 301,4 |
| aggregate (s) | 0,0 | 0,0 | 0,0 |
| eval (s) | 35,5 | 15,1 | 16,1 |
| **tổng (s)** | **375,6** | **319,7** | **317,5** |
| accuracy | 0,452001 | 0,770118 | 0,807165 |
| f1_macro | 0,202941 | 0,605268 | 0,710131 |

Round 1 đắt hơn ~58 s vì trả tiền compile và lần eval đầu. Steady state **319,7 s/round**.
Prepack **125,9 s** (train decode 89,2 s + test decode 36,7 s) — rẻ hơn dự đoán nhiều.
Resident sau load **6,32 GiB**/GPU. Aggregate và commit I/O đo được **0,0 s**: không đáng kể.
`ms_per_step` thật ở batch 512 = **7,245** (micro-bench 6,487; chênh là chi phí thật của
proximal, sampler và ranh giới client).

**Ngoại suy từ 3050 đã được kiểm chứng đúng**: dự đoán ~320 s/round, đo được 319,7 s;
dự đoán eval ~17 s trên 2 GPU, đo được 15–16 s. Không có bất ngờ về hiệu năng.

**Accuracy tăng 0,45 → 0,77 → 0,81 sau 3 round trên toàn bộ test set thật** — đây là bằng
chứng đường ống học được với cấu hình calibration, khác hẳn fixture (mục 7).
**Đính chính re-review:** calibration truyền `--rounds 3`, nên LR thực tế là
`0.001 → 0.000505 → 0.00001`: một lịch cosine hoàn chỉnh 3 round. Đây **không phải** ba
round đầu của lịch 50 round, vốn có LR xấp xỉ `0.001 → 0.000998983 → 0.000995936`.
Số đo tốc độ/phần cứng vẫn hữu ích để lập ngân sách; không dùng đường accuracy này để
khẳng định hội tụ hoặc ổn định số học của lịch 50 round.

### 6.3 Ngân sách đã chốt

| Kịch bản | step/round | ms/step | train (h) | tổng (h) | `suggested` | **MAX_HOURS đặt** |
|---|---:|---:|---:|---:|---:|---:|
| 20 | 84.083 | 7,245 | 4,23 | 4,44 | 4,9 | **6,0** |
| 50 | 84.098 | 7,245 | 4,23 | 4,44 | 4,9 | **6,0** |
| 100 | 168.200 | 6,733 | 7,86 | 8,07 | 8,6 | **9,5** |

Quota còn **29,59 h** (đã dùng 1489 s). Kỳ vọng tiêu **16,95 h**; nếu cả ba chạy hết deadline
thì **21,5 h**, vẫn nằm trong quota và vẫn dưới trần session 12 h của Kaggle.

**Vì sao MAX_HOURS cao hơn `suggested_max_hours`:** `suggested` chỉ chừa ~0,5 h ≈ một round,
một lần chậm 6% là mất. Quota không phải ràng buộc (29,6 h so với 17,0 h cần), nên chừa ~35%
headroom: một session resume thừa tốn một lần push và một lần prepack, còn headroom thừa
không tốn gì nếu không dùng tới. Driver dừng **trước** round sẽ vượt deadline nên không mất
round đã commit trong cả hai hướng.

## 7. Đã kiểm chứng những gì

Model: 395.024 params, 3.295 buffer, forward (4,66)→(4,16), state_dict 192 key.

Số học:

- `metrics_from_confusion` khớp sklearn tới 1,1e-16 trên 3 bộ dữ liệu, kể cả khi có
  lớp không bao giờ được dự đoán (`zero_division=0`).
- Proximal cộng vào gradient **bằng đúng** dạng autograd: max|Δ| 7,45e-09, tương đối 3,8e-08.
- Compiled vs eager logits: max|Δ| 4,88e-04, **0 bất đồng argmax** trên các dòng decisive.
- Aggregation phân cấp = FedAvg sample-weighted phẳng, max|Δ| 6,0e-08 (kiểm cho cả 20/50/100).
- `mu` luôn trong [0.01, 0.02); drift toàn 0 → đúng 0.01. LR round1=1e-3, round-cuối=1e-5.
- flatpack round-trip state_dict chính xác tuyệt đối.

End-to-end trên fixture dữ liệu thật, **cả ba kịch bản** (20/50/100 client × 1200 dòng,
test 16.000 dòng đủ 16 lớp), batch đúng 512/512/256:

- `scripts/verify_run.py` pass **9/9 trên cả ba**, kiểm **mọi round** (không chỉ round cuối):
  trọng số nạp `weights_only=True` + `strict=True` đủ BN buffer; y_pred dựng đúng confusion,
  phủ trọn test; tổng step = Σ ceil(n_i/batch) → tail batch được giữ; history không trùng dòng;
  **seed mỗi client khớp công thức (round, client_id)** → lịch gán GPU không rò vào RNG.
- **Mount**: resolve đúng cả layout phẳng và layout có prefix `datasets/<owner>/`;
  **từ chối** khi thiếu sentinel và khi có nhiều match.
- **Guard client**: tiêm lỗi thật (đặt 66 feature của 1 client = 6,0e4 — hữu hạn trong fp16
  nhưng tràn trong mạng) → driver từ chối với thông báo nêu đủ 4 vấn đề, 0 round bị nhiễm.
  Trên dữ liệu sạch guard **không** báo nhầm.
- **Resume trong session**: dừng sạch theo max_hours rồi chạy tiếp, nối đúng round.
- **Resume qua session mới**: output cũ đặt read-only (mô phỏng `/kaggle/input`), working
  trống → import đúng số round, chạy tiếp tới hết, history liền mạch 1..5, verifier 9/9.
- **Từ chối resume sai**: `--require-resume` không tìm thấy → fail **trước prepack**;
  đổi `--rounds` (đổi lịch cosine) → fingerprint khác nên không import.
- **Crash giữa round**: xoá marker + phá file preds → round đó làm lại, artifact hỏng bị
  ghi đè, history không trùng.
- **2 worker + eval sharding**: confusion cộng đúng, y_pred ghép đúng thứ tự test.
- **Worker bị SIGKILL giữa round**: driver thoát rc=1 sau **27,4 s** với thông báo nêu
  signal, giữ nguyên round đã commit (trước đây sẽ treo tới timeout 2 giờ).
- **W&B thật**: log đủ 10 key `eval/*` cộng `progress/*`, `train/*`, `round/*`;
  `wandb_run.json` chỉ chứa định danh.
- Module trích **từ chính notebook** chạy lại cho cùng kết quả, trên mount lồng, 2 worker.
- Notebook: nbformat valid, **kernelspec có mặt**, 9 file nhúng khớp byte với nguồn,
  metadata `machine_shape=NvidiaTeslaT4`, `is_private`, id khớp slug sinh từ title.

**Kiểm chứng trên phần cứng thật (2026-09-07)** — mạnh hơn mọi kiểm chứng fixture ở trên:
`scripts/verify_run.py` chạy trên **output T4 thật** của calibration (thư mục `calib/` pull về
từ Kaggle) pass **9/9, mọi round**, với dữ liệu production đầy đủ: 20 client × 43.045.415 dòng,
**84.083 step == Σ ceil(n_i/512)**, dự đoán phủ trọn **10.761.343** dòng test, metrics khớp
confusion tới **max|Δ| = 0,0**, trọng số nạp lại `strict=True`, seed khớp công thức
(round, client). Đây là lần đầu contract được kiểm trên dữ liệu thật quy mô đầy đủ chứ không
phải fixture, và trên chính notebook đã push.

**Giới hạn của các kiểm chứng này:** fixture chỉ 24k–120k dòng nên accuracy ~0,07–0,10 chỉ
chứng minh đường ống, **không phải kết quả khoa học**. Máy local là 1 GPU sm86 4 GB; hai
worker được ép lên cùng device nên **bản thân fixture local không chứng minh** residency
5,68 GB thật hay hai GPU độc lập; calibration T4×2 ở mục 6 đã bổ sung bằng chứng đó.
Chạy lại một round **không bit-identical** (cuDNN atomics + quỹ đạo AMP scale);
recovery khôi phục một round hợp lệ, không phải round đã mất.

**Ràng buộc môi trường local:** WSL chỉ có 8 GB RAM. Một `fl_train` với 1 worker đạt đỉnh
~2,2 GB RSS; chạy hai job song song đã làm crash máy (OOM killer). Chỉ chạy **một job một
lúc**, và `make_fixture.py` phải stream record batch chứ không `pq.read_table` cả part
(một part VeReMi giải nén hơn 1 GB).

## 8. Bài học đã rút ra (đã đưa vào skill)

Những cái bẫy đã **thực sự** làm hỏng kết quả, không phải suy đoán:

1. **Kiểm chứng proximal báo lệch 43%** — do `Dropout(0.1)` bốc mask khác nhau giữa hai
   forward và BN mutate running stats. Test gradient ở train mode phải cố định RNG **và**
   khôi phục BN buffer, khôi phục **sau** `backward()`.
2. **`reduce-overhead` bị loại oan** vì harness gọi 3 `backward()` liên tiếp không
   `zero_grad` → lỗi CUDA graph "gradient tensor overwritten". Harness phải mô phỏng đúng
   một step đầy đủ.
3. **`torch.mean` trên CUDA trả 0,99999994 cho một khớp hoàn hảo** với 77/600 giá trị n
   (nhân nghịch đảo đã làm tròn); trên CPU luôn đúng. Tiêu chí `agree < 1.0` vì thế loại
   nhầm build đúng **một cách ngẫu nhiên**, âm thầm rơi về eager và mất 3,8× tốc độ.
   Phải **đếm bất đồng bằng số nguyên**.
4. **resume_state.pt ghi SAU completion marker** → crash giữa hai lần ghi làm marker và
   state nói hai round khác nhau. Sửa: resume state ghi theo từng round, **trước** marker.
5. **`shutil.copy2` giữ nguyên quyền read-only của mount nguồn** → mọi file import từ
   `/kaggle/input` thành `r--r--r--`, lần ghi `config.json`/`history.csv` kế tiếp là
   PermissionError, **sau khi** import đã báo thành công. Dùng `copyfile` + `chmod`.
6. **Thống kê của step bị AMP bỏ là vô nghĩa** — grad norm là inf theo định nghĩa. Gộp vào
   trung bình biến một lần warm-up bình thường thành tín hiệu diverge giả. Chỉ tính trên
   step **được áp dụng**, và dùng `torch.where` chứ không nhân (`inf * 0 = nan`).
7. **Ngưỡng skip chỉ theo phần trăm bắt oan client khoẻ**: một lần warm-up là 33% của
   client 3 step. Phải có thêm hạn mức tuyệt đối.
8. **Notebook thiếu `metadata.kernelspec` không bao giờ chạy** — papermill của Kaggle báo
   "No kernel name found". `nbf.v4.new_notebook()` để metadata rỗng. Lỗi này không tốn
   quota (fail trước cell 1) nhưng đốt một version.
9. **Kaggle lấy slug từ title, không phải từ `id` trong metadata** — chỉ cảnh báo nhẹ rồi
   tạo notebook ở URL khác, khiến mọi lệnh status/output sau đó trỏ vào slug không tồn tại.
10. `lr_for_round` chia cho `total_rounds - 1` → crash khi `--rounds 1`.
11. **`WANDB_API_KEY` + `relogin=True` KHÔNG phải là đăng nhập.** `relogin=True` bảo wandb
    vứt credential đang có và xác thực lại, và nhánh đó **không đọc** `WANDB_API_KEY` — nó đi
    tìm netrc rồi prompt. Kernel Kaggle không có cả hai nên chết với
    `UsageError: No API key configured`. Ở máy dev thì y hệt code đó **pass**, vì `~/.netrc`
    còn lại từ lần `wandb login` trước đã âm thầm thoả mãn lookup. Phải truyền `key=` tường
    minh. Lỗi này đốt version 1 của notebook 100 client ngày 2026-09-07 (mất 27,7 s quota) và
    **không một test local nào thấy được** cho tới khi xoá netrc. Cách tái hiện: đặt `HOME` và
    `WANDB_CONFIG_DIR` sang thư mục tạm rỗng, xoá `WANDB_API_KEY`, rồi khẳng định dạng cũ raise
    còn `wandb.login(key=...)` trả True và `wandb.Api().default_entity` resolve được.

Ngoài ra: fingerprint phải chứa **mọi hằng số khoa học + định danh dữ liệu**, và phải tính
được **trước** prepack (đếm từ footer parquet) để `--require-resume` fail trong vài giây;
state_dict 192 entry qua `mp.Queue` sẽ cạn file descriptor ở 100 client (~19k fd/round) nên
phải flatten thành 1 vector; worker bị OS kill không đặt gì lên queue nên phải poll liveness.

## 9. Kaggle MCP và W&B

**MCP đã xác thực, kiểm lại 2026-09-07**: `get_accelerator_quota` trả GPU 108000s (30 h),
dùng 1489.116s, reserved 0s,
refresh 2026-09-12T00:00:00Z. Token: `~/.kaggle/accounts/minhtran0601.mcp-token`, khớp header
trong `.mcp.json` và `.vscode/mcp.json`. Hook `SessionStart` trong `.claude/settings.json`
refresh OAuth CLI cho Claude; Codex dùng `http_headers_helper`, không cần export token.
Không in token ra chat/log.

Lưu ý quyền MCP: `get_notebook_session_status` với `kernelSlug` **không tồn tại** trả
`Permission 'kernels.get' was denied` chứ không phải "not found" — đừng đọc thành lỗi token.

**W&B**: entity `21522798-uit`, project `afpha-dagsnet-veremi`, run id cố định
`afpha-dagsnet-{20,50,100}client`. `resume="allow"` chỉ nối lịch sử W&B, **không** khôi phục
checkpoint. Key local trong `~/.netrc`.

**W&B API (đọc tiến trình từ local, xác nhận 2026-09-07):** `wandb` 0.29.0 trong env `nckh`;
`wandb.Api()` tự lấy key từ `~/.netrc`, `api.default_entity` trả **`21522798-uit`** → xác thực
đọc hoạt động. `api.runs('21522798-uit/afpha-dagsnet-veremi')` hiện lỗi vì **project chưa tồn
tại** (chưa run nào) — đây là trạng thái mong đợi trước lần train đầu, không phải lỗi token.

Công cụ đọc: [inspect_run.py](.agents/skills/wandb-training-monitor/scripts/inspect_run.py),
`--self-test` pass. Dùng đúng `ENTITY/PROJECT/RUN_ID`, không đoán run mới nhất:

```bash
conda run -n nckh python .agents/skills/wandb-training-monitor/scripts/inspect_run.py \
  --run 21522798-uit/afpha-dagsnet-veremi/afpha-dagsnet-100client
```

Hai bẫy API đã ghi trong skill và vẫn áp dụng: `scan_history(min_step=...)` nhận **giá trị
step** chứ không phải offset dòng (heartbeat ở step 250/500/750 nên `lastHistoryStep - 40`
cắt gần hết lịch sử); và các dòng **thưa** — dòng tổng kết round không có key `train/*`, phải
coalesce tiến về trước trước khi đọc "giá trị mới nhất", nếu không ranh giới round trông như
đứng máy. Run đọc được qua API cũng ghi `wandb_run.json` trong run dir, chỉ chứa định danh.

**Uỷ quyền inline key — thường trực, mọi dự án (người dùng, 2026-09-07):** được phép nhúng
thẳng W&B API key vào notebook, lý do notebook để private và dùng tại local. Người dùng đã
**mở rộng phạm vi**: áp dụng cho **mọi dự án** của họ, không còn phải hỏi lại từng dự án.
Đã ghi vào skill (wandb.md §1 + hai SKILL.md). Điều kiện kèm theo, không phải tuỳ chọn:
notebook đích phải `is_private: true` và generator phải assert trước khi ghi; key đọc từ
`~/.netrc` lúc build, không in ra, không truyền qua tham số shell; file `.ipynb` sinh ra phải
nằm trong `.gitignore`.

Đã triển khai: `build_notebooks.py` chèn key vào **chỉ** notebook `is_private: true`;
notebook vẫn **ưu tiên Kaggle secret** khi secret được gắn. Ba notebook train chứa key;
calibration không cần nên không chứa.

Giá phải trả, phải nhắc lại mỗi lần dùng: key nằm **vĩnh viễn** trong version history của
Kaggle; chia sẻ hoặc public notebook là lộ key; đổi key phải build và push lại mọi notebook.

**Cách gỡ an toàn** (đầy đủ ở wandb.md §1.1, thứ tự là phần quan trọng nhất): (1) **revoke
key ở W&B trước** — version đã push lên Kaggle không xoá sạch được, xoá version mới nhất
không xoá các version cũ, nên revoke tại nguồn là bước duy nhất thật sự vô hiệu hoá chuỗi
key; (2) `wandb login --relogin`; (3) build lại (`--no-embed-wandb-key` để không kèm key, hoặc
build thường để nhúng key mới) — build lại **chỉ** đổi file local, không đổi gì đã có trên
Kaggle; (4) push lại; (5) xoá `.ipynb` sinh ra và output đã pull; nếu key từng vào git commit
thì `.gitignore` vô ích, phải rewrite history + force push **cộng thêm** bước 1 chứ không thay
thế; (6) grep 8 ký tự đầu của key cũ trong repo/staging/output, xác nhận không còn match ngoài
`~/.netrc`. Nên rotate **giữa** các run: key bị revoke có thể làm hỏng phần upload metric của
run đang chạy (training không ảnh hưởng vì W&B nằm ngoài đường train).

Nhờ inline key, việc launch **không còn phụ thuộc** vào việc secret có sống sót qua API push.
Cell probe trong calibration vẫn giữ để ghi nhận sự thật đó.

## 10. Việc người dùng phải làm tay

MCP đã xác thực, notebook đã có `machine_shape=NvidiaTeslaT4`.
Theo yêu cầu re-review, người dùng kiểm tra/sửa các mục 14 trước khi bắt đầu huấn luyện;
W&B/API key người dùng tự kiểm tra, không phải điểm chặn của lượt này.

Tuỳ chọn: nếu muốn dùng Kaggle Secret thay cho inline key, tạo secret tên `wandb_key`
(Add-ons → Secrets) và bật cho từng notebook — notebook sẽ tự ưu tiên secret; sau đó dựng
lại bằng `--no-embed-wandb-key` nếu muốn gỡ key khỏi file.

Việc cần quyết định: **ra lệnh cho phép launch**. Chưa có lệnh thì không push notebook train,
không chạy training.

## 11. Thứ tự việc còn lại

0. Xử lý các finding R1–R5 ở mục 14, ưu tiên R1/R2/R5; sinh lại notebook và xác minh payload
   sau sửa. Re-review chỉ ghi nhận vấn đề, chưa sửa code train/verifier.
1. ~~Push + chạy calibration~~ — xong, COMPLETE, kết quả ở mục 6.
2. ~~Đặt `MAX_HOURS` từ `suggested_max_hours` và quota còn lại~~ — xong: 6,0/6,0/9,5
   (lý do chọn cao hơn `suggested` ở mục 6.3), đã build lại và validate 4/4.
3. Khi người dùng ra lệnh: push 3 notebook train
   (`kaggle kernels push -p notebooks/afpha-dagsnet-train-<n>client`). Kaggle thường giới hạn
   **2 session GPU song song**; chưa xác nhận con số cho tài khoản này. Nếu session thứ ba bị
   từ chối thì chạy 2 trước rồi tiếp cái còn lại — quota tính theo tổng thời gian session nên
   chạy song song không tốn thêm quota, chỉ rút ngắn wall time.
4. Theo dõi: W&B + `kaggle kernels status` + `get_accelerator_quota`.
   **Sửa 2026-09-07 (đo thật trên calibration):** `time_reserved` **ở nguyên 0s** suốt một
   session đã xác nhận RUNNING → **không dùng `time_reserved > 0` làm tiêu chí liveness**.
   Thứ tăng là `time_used`, tăng **live** trong lúc chạy (504s → 1251s qua hai lần poll).
   Liveness đáng tin = `_timestamp` W&B mới + `progress/global_step` tăng, đối chiếu với
   `time_used` tăng qua hai lần đọc cách nhau vài phút (chỉ tính theo account, không chỉ ra
   được notebook nào). Trạng thái batch version không chứng minh session còn sống.
5. Nếu một session hết 12 h trước round 50: bật `REQUIRE_RESUME = True`, thêm output của
   chính notebook đó vào `kernel_sources`, push lại. Driver import round đã xong từ
   `/kaggle/input` và chạy tiếp.
6. Pull output, chạy `scripts/verify_run.py` trên từng run, viết report đủ 10 metric mọi
   round kèm caveat mục 3 và mục 4.

## 12. Trạng thái 9 review finding (cập nhật 2026-09-07)

Chín finding của lượt review 2026-09-06 đã có bản sửa và kiểm chứng được ghi ở mục 7–8.
**Re-review 2026-09-07 xác nhận các cải tiến đó, nhưng phát hiện các trường hợp biên còn
thiếu ở mục 14, nhất là import/history, định danh dữ liệu và độ mạnh của verifier.**
Không dùng bảng lịch sử dưới đây để bỏ qua các finding mới. Đề xuất A không bị mở lại.

| # | Finding | Trạng thái |
|---|---|---|
| 1 | Resume qua session Kaggle mới không hoạt động | **Sửa**: `import_previous` + `--resume-search` + `--require-resume`; test với mount read-only |
| 2 | Verifier hardcode 20 client, chỉ kiểm round cuối | **Sửa**: đọc `config.json`, kiểm mọi round, pass trên 20/50/100 |
| 3 | Client bị AMP bỏ toàn bộ step vẫn được nhận | **Sửa**: `_check_client`, test bằng tiêm lỗi thật; sửa luôn false-positive nó tạo ra |
| 4 | Mount có prefix `datasets/<owner>/` không tìm được | **Sửa**: depth 5; test phẳng/lồng/thiếu/mơ hồ |
| 5 | Metadata chưa chọn T4×2 | **Sửa**: `machine_shape` + staging dir + id khớp slug + kernelspec |
| 6 | Calibration không chứng minh hiệu năng production | **Sửa**: calibration chạy **chính driver** trên dữ liệu thật, 2 GPU, full test |
| 7 | Kết quả calibration không ràng buộc cấu hình train | **Sửa**: AMP=False chỉ để ghi nhận, không phải candidate; xuất `suggested_max_hours` |
| 8 | RNG và fingerprint resume chưa đủ | **Sửa**: seed dropout theo (round, client) và verifier kiểm; fingerprint gồm mọi hằng số + định danh dữ liệu; prepack cache ràng buộc cả test root |
| 9 | Budget và worker chết chưa kiểm chứng | **Sửa**: budget theo round tệ nhất **gồm commit I/O**; poll liveness, phát hiện SIGKILL trong 27,4 s |

Về #8, việc `resume_state` không lưu RNG state là **có chủ ý và đã đủ**: seed của mỗi client
là hàm thuần của `(seed, round, client_id)`, không phụ thuộc thứ tự chạy, nên không có trạng
thái RNG nào cần mang qua ranh giới round. Verifier kiểm đúng điều này. Vẫn **không hứa
bit-identical** vì cuDNN atomics và quỹ đạo AMP scale.

Bảy lỗi **mới** phát hiện trong lúc sửa (chi tiết mục 8): `torch.mean` trên CUDA, `copy2`
giữ quyền read-only, thống kê step bị AMP bỏ, ngưỡng skip theo phần trăm, `lr_for_round`
chia 0, notebook thiếu kernelspec, Kaggle lấy slug từ title.

## 15. Kết quả 100 client — ĐÃ XONG 50/50 round (2026-09-07)

Kernel `minhtran0601/afpha-dagsnet-train-100client` v2 **COMPLETE**, **50/50 round**, tiêu
**26.875 s = 7,47 h** (dưới MAX_HOURS 9,5 h nên dừng tự nhiên, không bị cắt). Output đã pull về
`runs/100client/`. W&B run `21522798-uit/afpha-dagsnet-veremi/afpha-dagsnet-100client`.

`scripts/verify_run.py --require-complete` → **10/10 pass trên cả 50 round**, gồm gate hoàn thành,
CSV khớp JSON từng round, mọi tensor hữu hạn, per_class khớp confusion, dự đoán phủ trọn
10.761.343 dòng, 168.200 step = Σ ceil(n_i/256), ID client đúng 0..99, LR đúng lịch cosine,
chuỗi mu khớp `resume/*.pt`, seed khớp công thức (round, client).

**9/9 file src chạy trên Kaggle khớp byte với `src/` + `architecture/` local** → ba kịch bản
chạy đúng cùng một code, so sánh được với nhau.

### 15.1 Hiệu năng thực tế

| | giá trị |
|---|---|
| Round 1 (có compile) | 664,3 s |
| Steady state (median 50 round) | **525,2 s** (min 516,9 / max 664,3) |
| Tổng train | **7,37 h** |
| Prepack | 189 s (train 156 + test 33) |
| Eval median | 14,3 s | 
| Aggregate max | 0,20 s |
| Resident/GPU | 6,31 GiB (free 7,53 / 14,56) |
| Compile mode | **`reduce-overhead` được chấp nhận trên cả 2 GPU** |

Ngoại suy calibration là 8,07 h, thực tế **7,37 h** — nhanh hơn 8,7%. (Ước tính giữa chừng
8,54 h dựa trên round 1–2 là quá bi quan: round 1–2 chưa phải steady state.)
Eval shard 5.380.672 + 5.380.671 = 10.761.343, rời nhau và phủ trọn.

### 15.2 Sức khoẻ huấn luyện — sạch

5.000 client-round (50 × 100) đều được log. `skip_pct` **max 0,78% / mean 0,09%** (ngưỡng
abort 5%). `ce_mean` hữu hạn toàn bộ, 0,0157–0,4610. `mu` ∈ [0,014155; 0,015684] nằm trong
spec [0,01; 0,02). `drift` 0,50–7,36, không phân kỳ. Log không có dòng fallback compile,
không NaN/Inf. Các warning còn lại đều vô hại (debugger frozen modules, mistune/nbconvert của
Kaggle, mmap array read-only ở `worker.py:30` là cố ý, `ConnectionError` của probe secret là
đúng thiết kế vì không gắn secret).

Một warning **cosmetic** cần sửa ở lượt sau, **không sửa bây giờ**: `wandb.login(anonymous=...)`
đã deprecated và không có tác dụng. Sửa bây giờ sẽ làm code của 20/50 client khác code mà
100 client đã chạy, phá mất tính so sánh được giữa ba kịch bản.

### 15.3 Quỹ đạo metric — có một đặc điểm phải nêu trong báo cáo

| round | accuracy | f1_macro | precision_macro | recall_macro |
|---:|---:|---:|---:|---:|
| 1 | 0,51396 | 0,24810 | 0,37485 | 0,27325 |
| 3 | **0,74921** | 0,63536 | 0,66888 | 0,63247 |
| 10 | 0,70548 | 0,69126 | **0,69892** | 0,71107 |
| 30 | 0,67027 | 0,66598 | 0,66300 | 0,71972 |
| 40 | 0,66970 | 0,65908 | 0,64542 | 0,72252 |
| **50** | 0,69671 | **0,69493** | 0,68276 | **0,73793** |

Round 50 là **f1_macro tốt nhất** cả run, nhưng accuracy đỉnh ở round 3 (0,749) rồi giảm và hồi
lại. `recall_macro` tăng gần đơn điệu 0,273 → 0,738. Đây là đánh đổi kinh điển trên dữ liệu mất
cân bằng: model bỏ dần độ đúng ở lớp đa số để lấy recall lớp hiếm. f1_macro giảm so với round
trước ở 22/49 round — dao động, không phải phân kỳ.

**Điểm quan trọng nhất: `benign` recall chỉ 0,248.** Trong 2.391.136 dòng benign, chỉ 593.103
được nhận đúng; phần còn lại rơi vào `dataReplay` (20,8%), `positionMirroring` (17,7%),
`timeDelayAttack` (16,7%). Diễn biến: benign recall đỉnh 0,770 ở round 2, tụt còn 0,140 ở
round 30, hồi lên 0,248 ở round 50.

**Đây là giới hạn của bài toán/feature, không phải lỗi pipeline** — và ba lớp nuốt benign chính
là ba lớp f1 thấp nhất (`timeDelayAttack` 0,163, `positionMirroring` 0,181, `dataReplay` 0,425).
Lý do có cơ sở: cả ba tấn công này phát đi **nội dung message hợp lệ** (replay message thật,
mirror vị trí thật của xe khác, gửi message thật bị trễ). Chúng chỉ khác benign ở **ngữ cảnh
thời gian/quan hệ**, thứ mà model 66 feature per-message **không nhìn thấy được**. Ngược lại,
các lớp có chữ ký per-message rõ ràng đều gần hoàn hảo: `dosAttack` 0,983,
`randomPositionOffset` 0,966, `trafficCongestionSybil` 0,956, `accelerationMultiplication` 0,952.

Hệ quả phải nói thẳng trong báo cáo: **f1_macro 0,695 không đồng nghĩa với một IDS dùng được** —
bỏ sót 75% lưu lượng benign là tỉ lệ báo động giả không chấp nhận được trong vận hành. Không lớp
nào bị bỏ trắng (`predicted` > 0 và f1 > 0 ở cả 16 lớp).

**Không đổi phương pháp vì phát hiện này.** Đề xuất A đã chốt và ba kịch bản phải so sánh được
với nhau; đây là kết quả cần báo cáo, không phải lỗi cần vá.

## 16. Cả ba kịch bản ĐÃ XONG — report đã viết (2026-09-08)

| Kịch bản | round | acc cuối | F1_macro cuối | F1_macro đỉnh (round) | tổng train |
|---|---:|---:|---:|---:|---:|
| 20 client | 50/50 | 0,748660 | 0,775221 | 0,784998 (r16) | 4,77 h |
| 50 client | 50/50 | 0,711642 | 0,716229 | 0,720220 (r7) | ~4,3 h |
| 100 client | 50/50 | 0,696708 | 0,694934 | 0,694934 (r50) | 7,37 h |

**Cả ba pass `verify_run.py --require-complete` 10/10.** Output ở `runs/{20,50,100}client/`.
Với cả ba run, 9 file source chạy trên Kaggle khớp **byte** với `src/` + `architecture/` local.

Report: **[report.md](report.md)** — sinh bằng `scripts/make_report.py`, đọc thẳng từ artifact,
không gõ tay số nào. Gồm: caveat khoa học, thiết lập, kết quả tổng hợp, phân tích, hiệu năng,
**ba bảng đầy đủ 50 round × 10 metrics**, per-class round 50, kiểm chứng, nguồn artifact.
Đã kiểm chéo độc lập: mọi số round 50 trong report dựng lại được từ chính confusion matrix thô,
`cm.sum() == 10.761.343` ở cả ba, và đẳng thức micro (`acc == P_mic == R_mic == F1_mic`) đúng.

### 16.1 Ba phát hiện chính

1. **Chất lượng giảm đơn điệu theo số client** trên cả acc lẫn F1_macro, cả giá trị đỉnh lẫn
   giá trị cuối. Cùng dữ liệu, cùng code, cùng seed — chỉ khác mức phân mảnh. Đây là hiệu ứng
   ba kịch bản được thiết kế để đo.
2. **Accuracy đạt đỉnh rất sớm (round 3–6) rồi giảm, trong khi `recall_macro` tăng gần đơn điệu**
   suốt 50 round (ví dụ 20 client: 0,231 → 0,802). Đánh đổi mất cân bằng lớp, không phải phân kỳ.
   Hệ quả: acc và F1_macro đạt đỉnh ở **hai round khác nhau** — trích số phải nói rõ round nào.
   Với 20 và 50 client, checkpoint tốt nhất **không phải** checkpoint cuối.
3. **`benign` recall chỉ 0,248–0,274 ở cả ba kịch bản** — gần như không đổi trong khi mọi metric
   khác đổi rõ theo số client. Đó là bằng chứng đây **không phải hiệu ứng FL** mà là giới hạn
   feature. Bốn lớp yếu nhất giống hệt và đúng thứ tự ở cả ba: `timeDelayAttack`,
   `positionMirroring`, `benign`, `dataReplay` — và ba lớp đầu chính là nơi `benign` bị nhận
   nhầm vào. Cả ba tấn công đó phát **nội dung message hợp lệ**, chỉ khác ở ngữ cảnh thời
   gian/quan hệ mà model 66 feature per-message không thấy được. Đối chứng: các lớp có chữ ký
   per-message rõ ràng đều F1 ≥ 0,95 ở cả ba kịch bản.

**Kết luận vận hành phải giữ trong mọi bản trình bày:** F1_macro 0,70–0,78 không đồng nghĩa với
một IDS dùng được — bỏ sót ~75% lưu lượng benign là tỉ lệ báo động giả không chấp nhận được.
Muốn cải thiện phải thêm feature theo chuỗi/ngữ cảnh, không phải chỉnh siêu tham số FL.

### 16.2 Việc còn lại

- Warning cosmetic chưa sửa (cố ý): `wandb.login(anonymous=...)` đã deprecated, vô tác dụng.
  Không sửa trong lúc ba kịch bản đang chạy vì sẽ phá tính so sánh được. **Giờ đã xong cả ba
  nên sửa được an toàn.**
- Tài khoản Kaggle CLI hay bị hook `SessionStart` đẩy sang account khác (đã thấy trôi sang
  `khanhngoc0304` rồi `odixe0502`). Kernel private của `minhtran0601` khi đó trả `denied`.
  `KAGGLE_CONFIG_DIR` **không** khắc phục được (CLI không tôn trọng nó cho credential OAuth).
  Cách xử lý: `kaggle_account.py use minhtran0601`. Người dùng đã cho phép làm việc này
  2026-09-08 để tải output; **không mặc định coi là uỷ quyền vĩnh viễn**.


## 13. Chuẩn hóa skill và trạng thái bàn giao cuối lượt review (2026-09-06)

Đã đối chiếu [OpenAI Docs về skill](https://learn.chatgpt.com/docs/build-skills) và skill
`skill-creator` có sẵn. Hai skill Codex ở `.agents/skills/` đã đúng cấu trúc
`SKILL.md` với `name`/`description`, reference/script theo nhu cầu và `agents/openai.yaml`.
Không cần chuyển sang `.codex/skills` hay tạo một skill riêng chỉ cho reference multi-account.

Những thay đổi skill **đã làm ngày 2026-09-06** (trạng thái train mới nhất xem mục 14):

- Giữ `.agents/skills` làm nguồn chính. Hai `SKILL.md` trong `.claude/skills` trở thành
  entrypoint ngắn trỏ tới nguồn chính; các reference/helper đã sửa được đồng bộ ở cả hai cây.
- Viết lại [multi-account.md](.agents/skills/kaggle-training-notebook/references/multi-account.md):
  tách inspect với switch; đọc inventory/quota hiện tại; không cộng mặc định thành 90 giờ;
  phân biệt CLI OAuth, Codex header helper và static header của Claude/VS Code.
- Bỏ kết luận nhận dạng account từ `has_ever_run`, và bỏ suy luận probe riêng thất bại
  có nghĩa token chắc chắn hỏng. Native MCP vừa kiểm lại vẫn xác thực thành công,
  GPU used/reserved=0, allowed=108000s; quota này không trả username.
- Sửa [kaggle_account.py](.agents/skills/kaggle-training-notebook/scripts/kaggle_account.py):
  quota khả dụng trừ cả `time_reserved`, chặn budget âm/NaN/Inf, không báo success khi refresh
  hoặc sync thất bại; kiểm snapshot đúng username, token đủ hai loại, chuẩn hóa tiền tố Bearer,
  giữ file credential mode 0600 và không in nội dung exception có thể mang secret.
  `use` vẫn có thể để lại thay đổi một phần nếu refresh/sync lỗi; sẽ trả nonzero và báo rõ.
  Chưa có rollback/lock liên tiến trình: không chạy đồng thời hai thao tác đổi account.
- Chuẩn hóa W&B liveness theo đúng session và heartbeat; `time_reserved` là toàn account.
  Giữ bộ 10 metrics và nhắc W&B resume không phải checkpoint resume. Sửa hướng dẫn attach
  output dùng `kernel_sources`, và Internet phải bật khi cần online W&B.
- Thay yêu cầu số học “numerics-exact” bằng dung sai có kiểm chứng cho AMP/compile, giữ
  nguyên thuật toán đã chốt. Không sửa `agents/openai.yaml` vì metadata hiện có hợp lệ.

Kiểm chứng: cả **4/4 SKILL.md pass quick_validate**, metadata UI hợp lệ, reference link
thực tế tồn tại (ví dụ template trong code fence không tính là link tới file hiện có),
các support file sửa ở hai cây khớp byte. Helper có **10 test offline** trong
[test_kaggle_account.py](.agents/skills/kaggle-training-notebook/scripts/test_kaggle_account.py),
dùng credential giả và mock network; chạy lại bằng:

```bash
conda run -n nckh python .agents/skills/kaggle-training-notebook/scripts/test_kaggle_account.py
```

**Giới hạn kiểm chứng:** chưa thử switch/refresh/sync thật trong lượt này; không gọi `ensure`
hay `use` với credential thật. `list` xác nhận `minhtran0601` vẫn active, `.mcp.json` có token
tương ứng; ba tài khoản lưu hiện có là minhtran0601, odixe0502, khanhngoc0304, đều đủ hai
file credential. Không sửa các file cấu hình MCP chứa token, không upload/launch notebook.
SHA256 nguồn train/asset vẫn khớp bằng chứng review.

**Câu hỏi đang chờ người dùng:** skill mới trước review cho tự xoay account, nhưng quyết định
ở mục 2 chỉ chọn minhtran0601. Đã hỏi giữ minhtran0601 hay cho tự chọn các account đã lưu.
Chưa có câu trả lời thì giữ quyết định hiện có; skill hỗ trợ cả hai chính sách theo ủy quyền,
không tự coi việc đã lưu token là đồng ý launch trên account đó. Không cần chờ câu này để sửa
guard/resume/verifier hoặc kiểm tra local đã được giao.

**Việc tay, cập nhật 2026-09-07:** đọc/test mục 14; W&B do người dùng tự kiểm tra. Hiện không cần
token mới hay restart toàn bộ window. Codex thường tự nhận thay đổi skill; nếu không thấy thì
restart Codex. Để compact chủ động cuộc trò chuyện, người dùng gửi `/compact` trong composer;
agent không có tool kích hoạt lệnh UI này. Tham chiếu:
[Codex IDE commands](https://learn.chatgpt.com/docs/developer-commands?surface=ide).
Phiên tiếp theo bắt đầu bằng đọc file này, ưu tiên mục 14; không coi các review findings
là đã sửa chỉ vì skill đã chuẩn hóa hoặc fixture thông thường đã pass.

## 14. Re-review notebook sau các bản sửa (2026-09-07)

**Phạm vi:** review notebook hiện tại, đối chiếu output calibration và tiêm lỗi vào bản sao
fixture local. Chỉ cập nhật tài liệu/bằng chứng; **không sửa source, generator, verifier hoặc
notebook, không launch, không đổi tài khoản, không kiểm tra/đổi API key W&B**.
Đề xuất A, batch 512/512/256, 50 round, 1 epoch và AMP vẫn giữ như người dùng đã chốt.

**Kết luận:** luồng train thông thường có bằng chứng tốt hơn lần review trước, đặc biệt
20 client trên T4×2. Chưa nên đóng review chạy dài vì recovery và kiểm chứng output còn
các lỗi đã tái hiện dưới đây. Những lỗi này không có nghĩa calibration gốc đã hỏng.
50/100 client vẫn là fixture + ngoại suy tốc độ; chưa có production run full của hai kịch bản.

> **CẬP NHẬT 2026-09-07 (lượt sửa):** cả **R1–R5 đã sửa và có test tiêm lỗi chứng minh**.
> Chi tiết ở mục 14.1. Mô tả lỗi bên dưới giữ nguyên làm hồ sơ; đừng đọc chúng như trạng
> thái hiện tại của code.

### Bằng chứng đã kiểm tra lại

- MCP trả `COMPLETE` cho `minhtran0601/afpha-dagsnet-calibration`; GPU đã dùng 1489.116s,
  allowed 108000s, refresh 2026-09-12T00:00:00Z. Không tốn thêm GPU quota cho re-review.
- Cả 4 notebook hợp lệ nbformat/kernelspec; kiểm cú pháp Python cell (bỏ qua dòng magic
  shell IPython, không execute notebook). Metadata có T4×2/private/staging đúng và dataset
  đúng scenario. Ba notebook train có 50 round, batch 512/512/256, budget 6.0/6.0/9.5 h.
- **9 file nhúng của mỗi notebook khớp byte với source/asset hiện tại VÀ với `src/` đã
  pull từ calibration Kaggle.** Đây là bằng chứng source đã chạy đúng trên T4 cho 20 client.
- Chạy lại verifier hiện tại: artifact calibration thật **9/9**, artifact fixture
  20/50/100 **9/9 mỗi bộ**. Không chạy lại training, không đọc lại toàn bộ parquet 43M dòng.
- Kiểm bổ sung độc lập artifact calibration cả 3 round: tất cả tensor trọng số hữu hạn;
  client ID là đúng 0..19 không trùng/thiếu; rows và steps từng client đúng; LR đúng lịch
  **3 round**; mu sử dụng nối đúng từ round trước; mu trong resume khớp công thức từ drift.

Bằng chứng có cấu trúc và SHA256 source/generator/verifier:
[notebook-review-2026-09-07.json](papers/khan-2025-afpha/notebook-review-2026-09-07.json).
File `notebook-review-checks.json` cũ là bằng chứng **2026-09-06**, không mô tả code hiện tại.
Artifact calibration gốc đang ở
`/tmp/claude-1000/-home-odixe-nckh-afpha/ea6ce942-8825-43a9-8efc-0b426d5b9394/scratchpad/calout/`.
Harness/bản sao tiêm lỗi ở `/tmp/afpha-review-2026-09-07/`; `/tmp` có thể bị dọn, nên các bước
tái hiện và kết quả quan trọng được lưu ngay trong mục này và JSON nêu trên.

### R1 — P1: crash khi ghi history có thể xóa lịch sử round đã commit

**Vị trí:** `src/fl_train.py:78` (`append_history`), đặc biệt mở `history.csv` mode `w`
tại dòng 86; `Run.commit` gọi trước khi ghi marker round mới.

**Tái hiện:** tạo history round 1 và marker round 1; tiêm OSError sau khi ghi lại CSV header
trong lúc append round 2; retry append round 2. Kết quả history chỉ còn **[2]**, dù marker
round 1 và `metrics/round_001.json` vẫn còn. Retry hiện không dựng lại dòng cũ từ JSON.

**Ảnh hưởng:** cơ chế marker cuối bảo vệ trọng số từng round nhưng chưa bảo vệ CSV dùng
cho bảng/đồ thị 10 metrics. Crash trong một lần rewrite có thể mất nhiều dòng lịch sử đã xong.

**Sửa/test đề nghị:** ghi CSV qua file tạm + atomic replace; khi recovery dựng/đối chiếu
history từ JSON của các round đã commit. Tiêm crash ở header/giữa các dòng: sau resume
history phải đúng 1..N, không mất/trùng round, khớp metric JSON với dung sai làm tròn 6 số.

### R2 — P1: import resume chưa atomic, retry có thể bỏ qua bundle chưa copy xong

**Vị trí:** `src/fl_train.py:122` (`import_previous`), nhánh return khi thấy marker tại
dòng 131 và vòng copy theo thứ tự đường dẫn tại dòng 155.

**Tái hiện:** import nguồn có marker round 1; ngắt copy trước file trong `weights/`.
Thư mục `complete/` đã được copy trước đó. Destination báo last round = **1** nhưng chưa
có weights. Gọi import lại trả **0 ngay**, weights vẫn thiếu; driver sau đó lỗi load resume
hoặc weights. Các file đã tồn tại còn bị skip mà không kiểm đầy đủ/nội dung.

**Sửa/test đề nghị:** copy vào staging tạm, kiểm bundle đầy đủ trước khi công bố marker/run;
hoặc bảo đảm marker chỉ xuất hiện sau khi import toàn bộ round hợp lệ. Retry phải sửa hoặc
loại bỏ bản import dở, không trộn file cũ/dở với nguồn mới. Test ngắt ở metrics/resume/weights,
file bị copy một phần và marker có gap; round nguồn đã commit phải phục hồi được.

### R3 — P2: fingerprint và prepack cache chưa nhận ra dữ liệu đổi nhưng giữ nguyên số dòng

**Vị trí:** `src/fl_train.py:310` (fingerprint), `src/fl_train.py:173` (prepack cache).
Fingerprint hiện có rows/client và n_test, chưa có version/content identity của train/test.
Cache thêm root path nhưng cũng không phân biệt nội dung thay đổi tại cùng đường dẫn.

**Tái hiện:** tạo parquet nhỏ, prepack một lần; sửa feature **và đảo label** của test tại
cùng file, giữ số dòng. Fingerprint vẫn y nguyên; prepack báo **cache hit**, trả feature và
label cũ. Trong session mới, không dùng cache cũ, cùng fingerprint đó còn có thể cho resume
trên dữ liệu khác; `y_true.npy` chỉ được ghi ở start_round=0 nên cần kiểm định danh test.

**Sửa/test đề nghị:** dùng dataset/version hoặc manifest digest đáng tin cậy cho train/test
và gắn nó vào cả fingerprint lẫn cache. Chỉ row count hoặc mtime không chứng minh cùng dữ
liệu. Tránh thêm quét giải nén full chỉ để tính fingerprint; có thể dùng manifest có hash file.
Test hai bộ cùng row count nhưng khác feature/label phải invalidate cache hoặc từ chối resume.
Đây là rủi ro khi cập nhật/đổi dataset, không phải bằng chứng bốn dataset version 1 đang sai.

### R4 — P2: nhiều nguồn resume cùng fingerprint được chọn âm thầm theo round lớn nhất

**Vị trí:** `src/fl_train.py:138` và dòng 149. `import_previous` chọn `best`, không yêu cầu
một nguồn duy nhất hoặc chứng minh các nguồn là các version thuộc cùng một lịch sử run.

**Tái hiện:** hai nguồn A/B cùng fingerprint, trọng số khác nhau, A có 1 marker và B có 2.
Importer tự chọn B và báo import round **2**, không báo mơ hồ. Fingerprint cấu hình không
phải run identity; hai lần train độc lập cùng cấu hình vẫn có thể có trọng số khác nhau.

**Sửa/test đề nghị:** cho chọn nguồn resume cụ thể hoặc fail khi nhiều nguồn phù hợp; nếu
tự chọn version mới hơn thì phải xác minh cùng run và lịch sử chung. Test hai nguồn cùng
số round lẫn khác số round. Với fresh run, không tự import chỉ vì gặp output cùng cấu hình;
notebook hiện luôn truyền `--resume-search /kaggle/input`, kể cả `REQUIRE_RESUME=False`.
Metadata fresh hiện `kernel_sources=[]` nên chưa tạo tình huống này trong launch thông thường.

### R5 — P1: verifier vẫn báo 9/9 cho một số artifact sai quan trọng

**Vị trí:** `scripts/verify_run.py:69` (history), `rounds_ok` từ dòng 128 và phần resume.
Kiểm formula bằng dữ liệu random không thay thế đối chiếu các giá trị được lưu trong run.

Trên **bốn bản sao độc lập** của fixture 20 client đã pass, từng thay đổi sau vẫn trả
**exit 0, “9 passed, 0 failed”**:

| Thay đổi tiêm vào bản sao | Điều hiện chưa được kiểm |
|---|---|
| Sửa accuracy round 1 trong CSV thành 0.999999 | CSV chưa đối chiếu metric JSON/confusion |
| Sửa mọi `resume/round_002.pt[mu]` thành 999 | Mu resume chưa đối chiếu `mu_next`/drift và công thức |
| Gán NaN vào một tensor `weights/round_001.pt` | `strict=True` kiểm cấu trúc, không kiểm mọi tensor hữu hạn; flatpack chỉ kiểm round cuối |
| Thay client log ID 1 bằng bản sao ID 0 | Chưa kiểm tập ID duy nhất/đủ và rows từng ID với config |

**Sửa/test đề nghị:** mỗi phép tiêm lỗi ở bảng phải khiến verifier trả nonzero và nêu đúng
round/file. Kiểm thêm LR/mu thực tế từng client, mu nối qua các round và resume, per-class
report với confusion. Dùng `model_source` đã lưu để kiểm reconstruction khi review artifact
của phiên cũ; verifier hiện import model/công thức từ source repo hiện tại.

**Giới hạn cần ghi rõ:** verifier kiểm *các round đã hoàn thành*, chưa bắt buộc hoàn tất
`cfg[rounds]`. Một bộ 3 round với config đổi thành 50 vẫn pass. Điều này hữu ích khi kiểm
checkpoint giữa chừng, nhưng báo cáo kết thúc cần gate riêng **đủ 50/50 round**; không dùng
exit 0 của verifier làm bằng chứng train đã hoàn thành cả kịch bản.

### Đính chính về calibration — đã sửa cách diễn giải trong mục 6

`scripts/build_notebooks.py:406` truyền `--rounds CAL_ROUNDS`, `CAL_ROUNDS=3` tại dòng 532.
LR từ client logs thật xác nhận lịch **0.001, 0.000505, 0.00001**. Ba notebook train vẫn dùng
50 round đúng đặc tả; không phát hiện lỗi lịch LR của notebook train. Chưa cần chạy lại
calibration chỉ để sửa báo cáo. Nếu cần kiểm ổn định trong ba round đầu của production,
phải giữ scheduler horizon=50 và tách riêng giới hạn dừng sau round 3.

## 14.1 Trạng thái sau lượt sửa (2026-09-07)

| # | Sửa gì | Bằng chứng |
|---|---|---|
| R1 | `history.csv` ghi qua file tạm + `os.replace` (atomic); trước mỗi lần ghi, mọi round **có marker** nhưng thiếu dòng được dựng lại từ `metrics/round_NNN.json` | CSV bị cắt còn `[2]` → sau commit round 3 thành `[1,2,3]`, dòng round 1 mang đúng accuracy 0,10; tiêm OSError lúc publish → file cũ **nguyên vẹn** |
| R2 | marker `complete/*.done` copy **cuối cùng**, chỉ sau khi `bundle_gaps()` xác nhận đủ artifact round 1..N; file đã tồn tại chỉ được skip khi **khớp size**; mỗi file copy qua `.part` + `os.replace` | ngắt copy trước `weights/` → **0 marker**, `last_complete=0`; retry import đủ 2 round, `bundle_gaps=[]`; file bị cắt còn 1 byte được copy lại đúng 1591 byte |
| R3 | `fldata._file_identity()` digest từ **footer parquet** (size file, num_rows, num_row_groups, và mỗi row-group × cột: `total_compressed_size` + min/max/null_count) → vào **cả** fingerprint lẫn prepack cache key; cũng ghi vào `config.json` | đổi feature + đảo label giữ nguyên số dòng: row count **không** đổi (8) nhưng digest test đổi `89eff2d9…` → `6f925d04…`, digest train giữ nguyên |
| R4 | nhiều nguồn cùng fingerprint → **SystemExit** liệt kê từng nguồn; thêm `--resume-source` để chỉ định | hai nguồn A(2 round)/B(3 round) → abort nêu đủ hai đường dẫn và gợi ý `--resume-source`; chỉ định A → import đúng A (tag trọng số 1.0), không phải B |
| R5 | verifier thêm: CSV↔JSON từng round; **mọi tensor hữu hạn** mọi round; flatpack round-trip mọi round; per_class↔confusion; ID client đúng `0..n-1` và rows khớp config; LR khớp lịch cosine; mu chuỗi từ drift **và** khớp `resume/*.pt`; cờ `--require-complete` | 4/4 phép tiêm lỗi của bảng R5 giờ **exit nonzero** và nêu đúng round/file (xem 14.2) |

### 14.2 Bốn phép tiêm lỗi R5 — kết quả sau khi sửa

| Tiêm vào bản sao | Verifier giờ báo |
|---|---|
| CSV accuracy round 1 → 0.999999 | `history.csv round 1 accuracy=0.999999 disagrees with metrics/round_001.json 0.452001 by 5.48e-01` |
| `resume/round_002.pt[mu]` → 999 | `round 2: resume/round_002.pt mu[6]=999.0 disagrees with the committed mu_next 0.0148…` |
| NaN vào một tensor `weights/round_001.pt` | `round 1: weights/001.pt key stems.0.0.weight holds non-finite values` |
| Client log ID 1 thay bằng bản sao ID 0 | `round 1: client ids are not exactly 0..19 (duplicates or gaps)` |

### 14.3 Kiểm chứng lại sau khi sửa

- Verifier siết chặt chạy trên **output T4 thật** của calibration: **9/9**, và giờ còn xác nhận
  thêm lịch LR thật (0.001 / 0.000505 / 0.00001) và chuỗi mu khớp `resume/*.pt` từng round.
- Fixture **100 client** (batch 256, 20 cụm) chạy end-to-end với driver đã sửa: exit 0, đỉnh
  RSS **2,16 GB**, verifier **9/9**; `--require-complete` pass khi đủ round và **fail đúng**
  khi config nói 50 mà chỉ có 3.
- **Resume qua session mới** với code mới: import 3 round từ mount **read-only**, verifier
  **10/10** gồm cả gate hoàn thành.
- 4 notebook build lại, `scripts/validate_notebooks.py` pass: kernelspec, id khớp slug,
  T4×2/private/internet, **36 file nhúng khớp byte**, batch/round/MAX_HOURS đúng từng kịch bản.

**Vẫn đúng và cần giữ:** verifier chỉ chứng nhận *các round đã hoàn thành*; muốn kết luận một
kịch bản xong phải chạy `--require-complete`. Và 50 client vẫn chưa có production run.

### Thứ tự người dùng nên kiểm tra trước launch

1. Sửa/test R1, R2, R5 để recovery và output có bằng chứng tin cậy.
2. Chốt/xử lý R3, R4 trước workflow đổi dataset hoặc resume qua nhiều nguồn.
3. Build lại notebook sau sửa source; kiểm source nhúng khớp, chạy fixture đủ ba scenario
   và các phép tiêm lỗi liên quan. Không cần lặp benchmark full 43M dòng nếu chỉ sửa I/O/verifier.
4. Giữ W&B cho người dùng tự kiểm tra như yêu cầu. Chỉ launch khi người dùng ra lệnh;
   lượt re-review này không khởi chạy bất kỳ notebook nào.
