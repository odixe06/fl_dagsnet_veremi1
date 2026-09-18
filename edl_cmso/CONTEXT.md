# CONTEXT — bàn giao phiên làm việc

> **📌 Note 2026-09-13 (chủ dự án, ghi từ phiên `~/nckh/pfedes`): thống nhất MỘT lịch learning rate cho mọi dự án anh em.**
> Lịch: AdamW, **cosine theo round** `lr(t) = lr_min + (lr − lr_min)/2·(1 + cos(π(t−1)/(T−1)))`, **lr = 1e-3, lr_min = 1e-5, T = 50**,
> hằng trong một round (công thức `afpha`; bản tham chiếu `~/nckh/pfedes/papers/pfedes-yi-2025/proj/pfedes.py::lr_at`).
> pFedES đã chạy lại với lịch này (kernel `*-cos`, 13-09). **Dự án này CHƯA sinh/push notebook mới** — chỉ làm khi chủ dự án yêu cầu;
> quota tuần 13→19-09 đã dành cho pFedES (~92 h). Chỗ cần sửa khi làm: cấu hình `LambdaLR` warmup 1 round + cosine của build 4 (centralized) — cần bỏ warmup và đặt đúng `lr_min + (lr − lr_min)/2·(1+cos(π(t−1)/(T−1)))`, t = 1…50, nếu chạy lại.
> Lý do và bằng chứng (đỉnh sớm sau ~2 epoch local, cosine chặn đà trôi nhưng không kéo lại đỉnh): `~/nckh/pfedes/CONTEXT.md` §12.

**Cập nhật:** 2026-09-04 · **Trạng thái:** ✅ **`edl-cmso-v4-dagsnet` XONG SẠCH 46/50 round** — ablation cho kết quả **cao nhất toàn dự án** (`f1_macro` 0,85320) với ít hơn 45,7% tham số (mục 1n). Không còn session nào chạy. Trước đó: không còn session nào chạy (`time_reserved = 0`). fast-50 chạy liên tục đến round 0 step **5.750** rồi dừng lúc 03:23:34 UTC — **không** phải step 2.250 như ghi chép cũ. **Kết luận đã đóng: TPU không phải đường chạy 50 round (mục 1m).**

| Build | Kernel | Phần cứng | Batch | Trạng thái |
|---|---|---|---|---|
| **1** | `edl-cmso-veremi` | 2×T4 fp16 | 4.096 | ✅ **XONG** 50/50 round, 9,74 h — [report.md](papers/build1-cmso-before-extractor/report.md) |
| **2 run 1** | `edl-cmso-veremi-v2` | 2×T4 fp16 | 4.096 | 🔴 **PHÂN KỲ** round 25; round 0–24 dùng được (mục 1d) |
| **2 run 2** | `edl-cmso-veremi-fp32` | 2×T4 fp32 | 4.096 | ⚫ **HUỶ TAY** 2026-09-02, còn trong prep. fp32 sai hướng — xem 1g |
| **2 run 3** | `edl-cmso-veremi-c50` | 2×T4 fp16 + **`torch.compile`** | 4.096 | 🔴 **PHÂN KỲ round 22**, sụp đổ từ round 14. Dùng được **round 0–13**. `torch.compile` cho **565,7 s/round = 1,34×**, đúng như đo local |
| **3 run 1** | `edl-cmso-v3-tpu` | TPU v5e-8 SPMD bf16 | 16.384 | 🔴 **PHÂN KỲ round 1**; batch lớn phá kết quả (mục 1g) |
| **3 run 2** | `edl-cmso-v3-b4096` | TPU v5e-8 SPMD bf16 | 4.096 | 🔴 **DỪNG SẠCH ở round 0** — 72,06% step gradient hỏng, tripwire mới chặn đúng lúc (1h, 1i) |
| **3 run 3** | `edl-cmso-v3-wd` | TPU v5e-8 SPMD bf16 | 4.096 | 🔴 **DỪNG round 0** — 66,10% step hỏng. `weight_decay=5e-2` KHÔNG sửa được (72,06% → 66,10%) |
| **3 diag** | `edl-cmso-v3-diag` | TPU v5e-8 SPMD bf16 | 4.096 | ✅ **TÌM RA NGUYÊN NHÂN** (mục 1l): XLA autocast chạy LayerNorm ở bf16, step hỏng đầu tiên 3469 tại `vit.0.n2` |
| **3 run 4** | `edl-cmso-v3-fp32ln` v1 | TPU v5e-8 SPMD, §4.8 **fp32** + DAGSNet bf16 | 4.096 | ⚫ **HARD-CANCEL ở round 6 step 7.250** vì ~53,1 phút train/round; v2 là notebook CPU dừng đã `COMPLETE`, v1 không commit artifact |
| **4** | `edl-cmso-v4-dagsnet` | 2×T4 fp16 + compile | 4.096 | ✅ **46/50, DỪNG SẠCH** theo budget. Ablation DAGSNet trần: **đỉnh r5 `f1_macro` 0,85320 — CAO NHẤT TOÀN DỰ ÁN**, 395.024 tham số. 330,1 s/round, 4,22 h |
| **2 final** | `edl-cmso-v2-final` | 2×T4 fp16 + compile | 4.096 | ⚪ **ĐÃ DỰNG, CHƯA PUSH** — wd 5e-2 + W&B + abort trong round; chờ quota (cần 7,86 h) |
| **3 run 5** | `edl-cmso-v3-fast50-wandb` / `edl_cmso_v3_tpu_fast50` | TPU v5e-8 SPMD, selective-fp32 + W&B | 4.096 → 8.192 → 16.384 | 🔴 **DỪNG trong r0 ở step 5.750** — grad 3,9×10¹³, loss tăng 0,55→1,68, **skip 0% suốt 23 window**; 0,3101 s/step; 0 artifact, chỉ 23 heartbeat W&B |

**Quota (đo 2026-09-03, `time_reserved = 0`):** GPU **4,64 h** còn (91.294,9/108.000 s) ·
TPU **7,86 h** còn (43.710,5/72.000 s) · refresh **2026-09-05**.
**Đã có tài khoản Kaggle thứ hai `minhtran0601`** để xoay tua quota — xem mục 4b. Ngân sách
tuần thành GPU 30+30 h, TPU 20+20 h, và hai account chạy song song được.

**TPU đã bị loại khỏi vai trò run 50 round — kết luận đóng, đừng thử lại (mục 1m).** Lịch
batch tăng dần cũng không cứu được: model bị dispatch chi phối nên TPU cho **số bước tối ưu cố
định mỗi giờ bất kể batch**, tức tổng số bước bị ngân sách thời gian ấn định và batch chỉ phân
bổ lại số bước đó. Batch nhỏ nhất nhét vừa 50 round trong 7,86 h là **26.389** — mà batch
16.384 đã đo `f1_macro` r0 **0,479 so với 0,831** ở batch 4.096.
**Đường GPU:** c50 đo 565,7 s/round → 50 round = **7,86 h** với đủ **525.450** bước tối ưu ở
batch 4.096, gấp 8× lượng tối ưu TPU mua được cùng thời gian. Chờ quota refresh 2026-09-05.

**Ngoài build 1, chưa có run đúng thứ tự pipeline nào đủ 50 round.** Có hai cơ chế khác
nhau (mục 1i/1l): hai run GPU vỡ do attention collapse muộn; TPU b4096/wd vỡ trong round 0
do LayerNorm bị XLA autocast xuống bf16. Run `fp32ln` đi qua ít nhất 70.304 train step mà
không tái hiện mẫu lỗi hàng loạt/abort (cửa sổ heartbeat cuối có 0 skip), nhưng bị dừng vì quá chậm.
Run `fast50` thêm cơ chế thứ ba: gradient **hữu hạn nhưng nổ tới 3,9×10¹³**, loss tăng đều, và
`skip_pct` bằng 0 suốt — không tripwire nào trong dự án bắt được (mục 1m).
Báo cáo hiện chỉ có thể dựa trên **round 0–13 của c50** và **round 0–24 của run 1**; đỉnh
f1_macro của cả hai đều nằm ở **round 0**. Kế hoạch cũ
"lấy v3 TPU làm chính" **đã bỏ**: batch 16.384 chỉ cho 2.627 bước tối ưu mỗi epoch thay vì
10.509, và f1_macro round 0 rơi từ 0,83081 xuống 0,47856. Batch lớn là lý do duy nhất TPU
thắng về throughput, mà chính nó phá kết quả. Build 1 KHÔNG đặt chung bảng với bài báo (mục 6).

**Báo cáo tổng hợp cả ba build:** [`report.md`](report.md) ở gốc project.

Câu đầu tiên cho session mới:

> Tiếp tục dự án EDL-CMSO. Đọc CONTEXT.md ở gốc project trước.

---

## 1. Việc đang làm

Dựng lại phương pháp của bài báo **Khan et al. 2025, *Scientific Reports*, DOI `10.1038/s41598-025-94445-9`** — "A Secure and Efficient Deep Learning-Based Intrusion Detection Framework for the Internet of Vehicles" — trên dataset VeReMi NextGen của tôi, rồi huấn luyện + đánh giá trên Kaggle 2×T4.

Bài báo gốc: [`s41598-025-94445-9.md`](s41598-025-94445-9.md) · Mô tả dataset: [`dataset.md`](dataset.md)
Dataset Kaggle: `odixe0502/veremi-nextgen2026-centralized`

**Dựng lại stage 3–5 của bài báo:** DWT → ViT → GAT → fusion (Eq. 28) → CMSO feature selection (Eq. 29–37) → DAGSNet (DenseNet + GoogleNet + AlexNet + SqueezeNet, Eq. 38–48).

**Bỏ hẳn, theo yêu cầu của tôi:**
- Stage 1 — IoVCipherGuard (HE / SMPC / AES-256, §4.3–4.6)
- Stage 2 — tiền xử lý (§4.7): dataset đã imputed + standardised sẵn
- Stage 6 — AFPHA federated aggregation: **chạy centralized, một model duy nhất**

## 1b. Mục tiêu huấn luyện — cái tôi thực sự muốn có

**Mục tiêu:** chứng minh phương pháp EDL-CMSO của bài báo chạy được và đo được **trên dữ liệu
V2X của tôi**, chứ không phải đạt lại con số của bài báo. Bài báo làm phân loại nhị phân trên
CIC-IDS 2017 / CAN; tôi làm 16 lớp trên VeReMi NextGen chia theo thời gian. Hai thứ không so
sánh được — cái kế thừa là **phương pháp**, không phải con số.

Sản phẩm cuối cần có:

1. **Một bảng 10 metric** trên **toàn bộ** 10.761.343 dòng test, đo lại sau **mỗi** round trong
   50 round — để thấy đường cong hội tụ, không chỉ điểm cuối.
2. **`f1_macro` là con số chính.** Mất cân bằng 41:1 nên `accuracy` gần như vô nghĩa; 5 trong
   10 cột sẽ ra cùng một số (đẳng thức collapse, không phải lỗi).
3. **Báo cáo per-class + confusion matrix** ở round cuối — để biết lớp nào mô hình thực sự
   học được, đặc biệt `timeDelayAttack` (khó nhất) và `trafficCongestionSybil` (bị rò rỉ).
4. **Toàn bộ deviation được ghi lại minh bạch** trong `rebuild.md` — bài báo bỏ trống 12 chỗ,
   mỗi lựa chọn thay thế phải nói rõ là bắt buộc hay do tôi chọn. Đây là thứ người phản biện
   sẽ hỏi.
5. **Notebook đã chạy kèm output** tải về máy — code, công thức, số đo nằm chung một file,
   tự chứng minh được.

Đây là công việc nghiên cứu khoa học, nên **tính trung thực của số đo quan trọng hơn việc số
đó đẹp**. Nếu một lớp cho F1 gần như hoàn hảo thì phải nói rõ nó đến từ đâu — xem mục 6.

## 1c. Build 2 — dựng lại đúng thứ tự pipeline của bài báo

**Vì sao có build 2.** Build 1 chạy CMSO trên 66 cột đầu vào, **trước** bộ trích xuất
(deviation 18 của nó). Bài báo thì §4.9 đi sau §4.8: CMSO chọn trong số đặc trưng mà bộ
trích xuất **sinh ra**. Build 2 khôi phục thứ tự đó.

```
build 1   66 raw ──CMSO(41)──► DWT ─► ViT ─► GAT ─► fusion ─► DAGSNet
build 2   66 raw ─► DWT ─► ViT ─► GAT ─► fusion(Eq.28) ──CMSO──► DAGSNet
bài báo   giống hệt build 2  (§4.8 → §4.9 → §4.10)
```

### Bốn quyết định đã chốt 2026-09-01 — đừng hỏi lại

| # | Quyết định | Ghi chú |
|---|---|---|
| A | **CMSO chạy trên extractor khởi tạo ngẫu nhiên** | không có warm-up — đúng nghĩa đen thứ tự feed-forward, không bịa thêm pha bài báo không mô tả |
| B | batch 4096, 50 round, LR 1e-3 cosine | **y hệt build 1**, để biến duy nhất đổi là vị trí CMSO |
| C | Eq.28 ghép **hệ số wavelet thô** (6 kênh) | fused = 6+128+128 = **262**, không phải 384 của build 1 |
| D | chỉ một kịch bản: có CMSO | không chạy cặp Table 3; run không-FS để sau khi quota refresh |

### Hợp đồng bắt buộc của quyết định A

Mask chọn trên một phép chiếu ngẫu nhiên rồi áp lên phép chiếu **khác** thì vô nghĩa. Nên:
probe dựng theo `CFG.seed` → đóng băng 33 tensor / 284.164 tham số §4.8 vào
`extractor_init.pt` → CMSO tìm trên chính map đó → `train_worker` load đúng state ấy ở
round 0. Smoke check 6 xác nhận `fuse()` **bit-identical** sau load, trong khi seed khác
lệch 6,0. **Đừng bao giờ gỡ bước này.**

**Giới hạn phải ghi kèm mọi con số của build 2:** lúc chọn, 262 kênh còn là phép chiếu
ngẫu nhiên. Mask **không** phải bằng chứng đặc trưng nào quan trọng — nó là một phép thu
hẹp bề rộng có cấu trúc mà mạng sau đó thích nghi theo. Không được phát biểu kiểu "CMSO
tìm ra đặc trưng tốt" từ run này.

### Hệ quả tốt và hệ quả xấu

- **Tốt:** cả 66 feature luôn vào DWT nên `k = 11` cố định → lớp bug `k=1` làm treo DDP ở
  build 1 không thể xảy ra nữa.
- **Xấu:** CMSO không còn cách nào bỏ nhóm `session` ở đầu vào (build 1 giữ 13/16). Rò rỉ
  Sybil ở mục 6 **nặng hơn** build 1, không nhẹ đi.

### Chi phí dự kiến

k đi từ 7 (build 1: 41 feature → 42 hệ số) lên 11. ViT, GAT và cả 4 backbone đều scale
theo k → round chậm hơn build 1 khoảng **1,4–1,6×**, ~1000–1100 s. Tổng ≈ **15–16 h**,
**không vừa một session 11 h**: run dừng sạch sau khoảng round 31 rồi resume session sau.
Đây là đường đi đã thiết kế, không phải sự cố.

### Resume sang session 2

Khi session 1 dừng (`logs/stopped_early.json` xuất hiện), thêm
`"odixe0502/edl-cmso-veremi-v2"` vào `kernel_sources` trong
`papers/build2-paper-order/notebook-run1-baseline/kernel-metadata.json` rồi push lại. Notebook tự
resume từ `last.pt`; `channel_mask.json` và `extractor_init.pt` được `resolve_resume` kéo
theo — thiếu file thứ hai thì mạng resume **không phải** mạng mà mask được chọn cho.

### Hai lỗi bắt được ở local, đừng để tái diễn

1. **`autocast` dùng ở cell 16 mà không import.** Mọi cell khác lấy symbol từ trong module
   `%%writefile`, top level không có. Sẽ nổ `NameError` **sau** 17 phút prep parquet và
   giết cả session — đúng dạng lỗi đã làm mất 2 session ở build 1. Đã thêm **check 12**
   quét tĩnh toàn notebook vào `smoke.py` để chặn vĩnh viễn.
2. **Một lập luận sai đã được đo lại.** Tôi từng viết mask bỏ hết kênh GAT sẽ treo DDP.
   Đo thật: `cat` + `index_select` giữ `gat.*` trong đồ thị autograd nên gradient là
   **zero chứ không phải None** — DDP an toàn với mọi mask. Hậu quả thật là nhánh GAT đóng
   băng ở khởi tạo suốt 50 round mà vẫn tốn forward pass. Floor 2 kênh/mỗi số hạng Eq.28
   vẫn giữ, nhưng vì lý do đó chứ không phải vì deadlock.

**Pre-flight:** `python papers/build2-paper-order/smoke.py` trong env `nckh` —
**40/40 pass**, trích thẳng 5 module `%%writefile` ra khỏi notebook nên test đúng code sẽ
chạy. Chạy lại trước mọi lần push.

## 1d. ⚠ Run 1 của build 2 phân kỳ — bài học, đừng lặp lại

**Kernel** `odixe0502/edl-cmso-veremi-v2` v1 · 10,89 h GPU · status `COMPLETE` nhưng
**kết quả chỉ dùng được một nửa**. Output đã kéo về `papers/build2-paper-order/run1-wd1e4-diverged-r25/`
(761 MB, đủ artifact).

### Chuyện gì xảy ra

`train_loss` thành **NaN ở round 25** và không hồi phục. Đã xác minh trên checkpoint:

| Checkpoint | Tham số NaN | Buffer BN NaN | Optimizer NaN |
|---|---|---|---|
| round 23 | 0/132 | 0/62 | 0 |
| round 24 | 0/132 | 0/62 | 0 |
| **round 25** | **131/132** | **62/62** | **262** |

Sạch tuyệt đối ở round 24, nổ toàn bộ trong round 25. Dấu hiệu báo trước nằm sẵn trong
loss: giảm đều tới 0,0506 (round 21) rồi **tăng ngược** 0,0514 → 0,0559 → 0,0771 → NaN.
`max|w|` lúc đó đã đạt 17,9.

> ⚠ **Câu "mất ổn định dưới fp16" ở đây là chẩn đoán SAI của tôi, giữ lại để đối chiếu.**
> v3 chạy **bf16** — dải mũ bằng fp32 — mà vẫn NaN ở round 1. Nguyên nhân thật đã đo được
> ở **mục 1g**: Eq. (28) làm 3 kênh lệch scale 208× so với 130 kênh còn lại.

### Lỗi vận hành phải nhớ

**Vòng train không hề kiểm tra `train_loss` có hữu hạn không.** Hậu quả:

- **4,1 h GPU đốt vô ích** — 20 round chạy tiếp trên trọng số NaN
- 20 checkpoint NaN được ghi
- **`last.pt` bị ghi đè bằng NaN** → run không resume được, phải chạy lại từ đầu

**Đã vá:** tripwire trong `train_worker.py` dừng ngay khi `train_loss` không hữu hạn,
**trước** eval và **trước** `save_round`, ghi `logs/diverged.json`. `train_loss` đến từ
`all_reduce(SUM)` nên hai rank có cùng giá trị và break cùng nhau — không cần collective
thêm, không có nguy cơ deadlock. **Đừng bao giờ gỡ tripwire này.**

**Quy tắc chung rút ra:** mọi vòng train dài phải có tripwire cho trạng thái không hồi
phục được. Chi phí một phép `isfinite` là bằng không; chi phí không có nó ở đây là 4,1 h
GPU cộng với việc mất khả năng resume.

### Phần dùng được, và nó nói gì

Round 0–24 hợp lệ. **f1_macro đỉnh 0,83081 tại round 0** (accuracy 0,86545) — cao hơn
đỉnh của build 1 (0,81676 tại round 2). Sau đó trôi xuống chậm tới 0,8114 ở round 21.
Cùng hiện tượng overfitting dưới split theo thời gian như build 1 nhưng lộ ra sớm hơn:
đạt đỉnh ngay sau 1 epoch.

**CMSO giữ 133/262 kênh** — wavelet 3/6, ViT 69/128, GAT 61/128, fitness 0,56789. Cuộc
tìm chỉ nhích 0,55859 → 0,56789 qua 50 iteration: gần như phẳng, đúng như đã cảnh báo ở
quyết định A vì đang chọn trên phép chiếu ngẫu nhiên.

### Một ước lượng của tôi đã sai, đã đo lại

**s/round thật = 759,7** (min 729, max 791), không phải 1000–1100 như tôi dự đoán — chỉ
gấp **1,08×** build 1 chứ không phải 1,4–1,6×. Lập luận sai: tôi tính k đi từ 7 lên 11 làm
mọi thứ đắt hơn, nhưng bỏ qua việc DAGSNet — phần chiếm chi phí chính — lại nhận **hẹp
hơn** (133 kênh so với 384 của build 1), gần như bù trừ hết. Bài học: khi ước lượng chi
phí, phải tính cả chiều rộng chứ không chỉ chiều dài của tensor.

## 1e. TPU — bốn probe đầu, và vì sao batch 4.096 không dùng được TPU

> **Đọc kèm mục 1f.** Kết luận "không dùng TPU" ở đây chỉ đúng cho **batch 4.096**. Mục 1f
> cho thấy ở batch 16.384 với SPMD thì TPU chạy tốt và đó là run chính (v3). Giữ mục này
> vì nó ghi lại các bẫy đã dính và số đo một-chip.

Bốn probe, tổng **842 s trong 20 h quota TPU**. Chi tiết kỹ thuật đầy đủ đã ghi vào skill:
[`references/tpu-pytorch.md`](.claude/skills/kaggle-training-notebook/references/tpu-pytorch.md).

### Số đo quyết định

```
1 core, per_core=  512:  127,8 ms/step →   4.008 samples/s
1 core, per_core= 4096:  128,3 ms/step →  31.934 samples/s
```

Batch gấp 8 lần, **thời gian mỗi step không đổi**. Model bị **chi phí dispatch chi phối**,
không phải compute: k=11 token và ≤262 kênh nằm dưới ô 128×128 của MXU, DAGSNet là một đám
op tí hon. MLP cùng cỡ tham số chạy 6 ms/step — nhanh hơn 20 lần.

| Cấu hình | samples/s |
|---|---:|
| 2×T4 GPU (đo ở run 1) | **56.714** |
| TPU v5e, global batch 4.096 | **31.934** |

**TPU chậm hơn GPU 1,8×.** 50 round cần 18,7 h chỉ riêng train, chưa tính eval/prep/CMSO —
vượt 20 h quota.

**8 core không cứu được:** 8 × 512 dòng, mỗi core vẫn 127,8 ms → 32.000 samples/s, đúng
bằng 1 core gánh 4.096. Ở batch nhỏ như vậy, chia cho 8 core mua được con số không. "Dùng
hết phần cứng TPU" và "giữ batch 4.096" là hai mục tiêu loại trừ nhau.

### Ba cái bẫy đã dính, đừng dính lại

1. **`machine_shape: "TpuV38"` KHÔNG cho v3-8.** Kaggle nhận chuỗi nhưng cấp
   `TPU_ACCELERATOR_TYPE = v5litepod-8` (v5e-8). Đúng vết xe đổ của `machine_shape` GPU ở
   build 1. **Luôn in `os.environ["TPU_ACCELERATOR_TYPE"]` ở cell đầu.**
2. **`torch_xla.launch()` chết nếu kernel đã đụng XLA device** —
   `InitializeComputationClient() can only be called once`, 8 process cùng SIGABRT, không
   có traceback Python. Phải chạy multi-core từ process riêng qua `subprocess`. Trên image
   này kể cả vậy vẫn hỏng: một process đã thấy cả 8 device, muốn dùng hết phải qua SPMD.
3. **`%%writefile` bắt buộc là dòng đầu cell.** Đặt comment lên trước → lỗi cú pháp, chỉ
   phát hiện khi chạy trên Kaggle.

### Môi trường TPU đo được (2026-09-02)

`v5litepod-8`, 8 core × 15,75 GiB HBM, 224 vCPU, 396 GB RAM, `torch 2.8.0+cpu` +
`torch_xla 2.8.0` trên PJRT. **`/kaggle/temp` KHÔNG tồn tại** — notebook GPU cache fp16 ở
đó, trên TPU phải đổi sang `/tmp` (1 TB trống). Chỉ được **1 session TPU batch cùng lúc**.

### Điểm chưa lý giải được

Benchmark matmul chuỗi cho 1,4–1,8 TFLOP/s, thấp hơn hai bậc so với mức danh nghĩa của
v5e. Phép đo end-to-end trên model thì đúng phương pháp và tự nhất quán, nên tin nó; nhưng
ghi nhận đây là bất thường chưa giải thích được.

## 1f. SPMD — cách dùng hết 8 chip TPU, và giới hạn thật của phần cứng

### Vì sao `torch_xla.launch` sai ngay từ mô hình

PJRT trên image Kaggle đặt **cả 8 chip dưới MỘT process**. Fork ra 8 process con là mô
hình sai, và nó chết bằng `InitializeComputationClient() can only be called once` — 8
SIGABRT, không traceback Python. Cách đúng là **SPMD**:

```python
xr.use_spmd()                     # BẮT BUỘC trước khi runtime khởi động
mesh = Mesh(np.arange(8), (8, 1), ("data", "model"))
xs.mark_sharding(xb, mesh, ("data", None))    # shard trục batch
```

XLA tự nhân bản model và tự all-reduce gradient. Đã xác minh: `spmd True`, `chips 8`,
`mesh {'data': 8, 'model': 1}`.

### Số đo SPMD trên 8 chip

| global batch | /chip | ms/step | samples/s | so 2×T4 | 50 round |
|---:|---:|---:|---:|---:|---:|
| 4.096 | 512 | 148,3 | 27.614 | **0,5×** | 21,7 h |
| 8.192 | 1.024 | 148,8 | 55.058 | 1,0× | 10,9 h |
| **16.384** | **2.048** | ~150 | ~109.000 | 1,9× | **~5,5 h** ← đã chọn |
| 32.768 | 4.096 | 150,9 | 217.123 | 3,8× | 2,8 h |
| 65.536 | 8.192 | 154,4 | 424.510 | 7,5× | 1,4 h |
| 262.144 | 32.768 | — | **OOM** 16,04 G / 15,75 G HBM | | |

**Kết luận cốt lõi, đúng cho CẢ TPU LẪN GPU:** model này **bị chi phí dispatch chi phối,
không phải compute**. Step time gần như không đổi từ batch 4.096 đến 65.536. Nguyên nhân:
k=11 token và ≤262 kênh quá nhỏ so với ô 128×128 của MXU, DAGSNet là một đám op tí hon.
MLP cùng cỡ tham số chạy nhanh hơn 20 lần.

Hệ quả: **"dùng hết phần cứng" chỉ đạt được bằng batch lớn.** Ở batch 4.096, cả 8 chip TPU
cho 27.614 samples/s — bằng nửa hai con T4. Không có cách tối ưu nào cứu được điều đó.

### Điều này cũng đúng với GPU — ⚠ mục này SAI, đã đo lại ở 1g

> **Đừng dùng đoạn dưới.** Tôi kết luận "GPU chưa dùng tối đa" từ chỗ VRAM còn dư
> (6,21/14,6 GB) và suy ra nó dispatch-bound như TPU. **Cả hai đều sai.** Dư VRAM không
> phải dư compute, và batch sweep đo ở local cho thấy step time **tuyến tính** theo batch
> → GPU **compute-bound, đã ở trần**. Số đo ở mục 1g.

### Ba bẫy XLA đã chặn trong notebook v3

1. **Shape tĩnh khi eval.** `N_TEST % 16.384 = 13.439`, không chia hết cho mesh 8. Batch
   cuối vừa là shape mới (XLA **compile lại toàn bộ đồ thị eval, 20–30 s mỗi round**) vừa
   không shard sạch. Đã pad mọi batch lên đúng `eval_batch` rồi cắt phần đệm.
2. **`xm.get_memory_info()` chết dưới SPMD** — `MemoryInfo not supported for SPMD virtual
   device`. Đừng gọi.
3. **`/kaggle/temp` không tồn tại trên node TPU** — dùng `/tmp` (1 TB).

Chi tiết đầy đủ đã ghi vào skill:
[`references/tpu-pytorch.md`](.claude/skills/kaggle-training-notebook/references/tpu-pytorch.md).

### v2-fp32 và v3 kế thừa gì

Cả hai copy **`channel_mask.json`** (133/262 kênh) và **`extractor_init.pt`** từ build 2
run 1, **không** chạy lại CMSO. Nhờ vậy cả ba run chọn đúng 133 kênh giống nhau từ đúng
phép chiếu §4.8 giống nhau — hợp đồng quyết định A còn nguyên, và khác biệt giữa chúng chỉ
còn: độ chính xác số, batch, phần cứng. **Checkpoint của run 1 KHÔNG được kế thừa** vì
`last.pt` chứa trọng số NaN.

v3 đổi thêm hai thứ do batch gấp 4 lần, đã ghi thành deviation:
**LR 2e-3** (√4 × 1e-3, square-root scaling vì optimizer là Adam) và **warmup 3 round**
thay vì 1, rồi cosine.

## 1g. Đo trên GPU local — ba giả thuyết của tôi bị bác bỏ

**2026-09-02.** Máy này **có GPU**: `NVIDIA GeForce RTX 3050 Laptop, 4 GB, sm_86 (Ampere)`.
Env `nckh` trước đó cài `torch 2.13.0+cpu` nên tôi tưởng không đo được ở local — sai.
Đã cài đè `torch==2.13.0+cu130` (cùng version, chỉ thêm CUDA) và `gcc_linux-64` cho Triton:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate nckh
pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.13.0+cu130"
conda install -y gcc_linux-64 gxx_linux-64      # Inductor cần C compiler
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
```

**Từ giờ mọi giả thuyết về tốc độ phải đo ở local trước, không tốn quota Kaggle.** Script
đo nằm ở [`papers/build2-paper-order/bench/`](papers/build2-paper-order/bench/) —
chúng nạp thẳng `proj/model.py` thật, không phải model đồ chơi.

| script | trả lời câu hỏi gì |
|---|---|
| `bench.py` | host gather / per-step sync / data resident / compile — biến thể A–F |
| `split.py` | thời gian nằm ở `fuse` hay ở 4 nhánh DAGSNet |
| `verify.py` | compile có đổi kết quả không; padding kênh có giúp không |
| `scale.py` | độ lệch scale giữa các thành phần của Eq. (28) |
| `prec.py` | fp32 vs fp16, có/không compile |

### Giả thuyết 1 — "nút thắt là data loading + sync mỗi step". SAI.

```
A  host fancy-index gather + float(loss) mỗi step   50,96 ms/step
B  bỏ sync, cộng dồn loss trên device               50,44 ms/step   1,01×
C  toàn bộ ma trận train nằm sẵn trên GPU           50,28 ms/step   1,01×
```

Chênh 1%. Đưa dữ liệu lên device và bỏ host sync **không mua được gì**. (Vòng train GPU
vốn đã cộng dồn loss trên device qua `running`, nên nó chưa bao giờ là vấn đề.)

### Giả thuyết 2 — "GPU chưa dùng hết vì còn dư VRAM". SAI.

```
batch  512 →  31,70 ms → 16.154 samples/s
batch 1024 →  50,50 ms → 20.275 samples/s
batch 2048 →  97,10 ms → 21.092 samples/s
batch 4096 → 190,94 ms → 21.452 samples/s
batch 8192 → 382,72 ms → 21.405 samples/s
```

Step time **tuyến tính** theo batch, throughput chạm trần. GPU **compute-bound, đã ở
trần** — ngược hẳn TPU (step time phẳng từ 4.096 đến 65.536). Hệ quả:
**tăng batch trên GPU mua được con số không.** Dư VRAM ≠ dư compute; đừng lặp lại suy
luận đó.

### Giả thuyết 3 — "133 kênh không chia hết 8 nên mất tensor core". SAI.

```
C=128  0,743 ms     C=133  0,835 ms  (hiện tại)     C=136  0,887 ms     C=144  0,881 ms
```

Pad lên 136 còn **chậm hơn**. Bỏ hướng này. Stem dù sao cũng chỉ 0,835/97 ms.

### Cái thật sự hiệu quả: `torch.compile` — 1,33×

Profile (batch 2048, fp16) cho biết vì sao:

```
fuse (DWT+ViT+GAT)   12,68 ms   ← 47% của forward
stem+dense            2,21 ms
stem+google           4,45 ms
stem+alex             2,80 ms
stem+squeeze          3,77 ms
full forward         27,04 ms
forward+backward     94,18 ms
```

`fuse` chạy trên tensor (2048,128,11) — khoảng 200 kernel tí hon **bị chặn bởi băng thông
bộ nhớ**, không phải FLOP. Đó đúng là thứ Inductor gộp được.

```
fp16 AMP             96,49 ms   21.225/s
fp16 AMP + compile   72,47 ms   28.261/s   ← 1,33×
fp32                146,24 ms   14.005/s
fp32     + compile  118,74 ms   17.248/s
```

**Compile không đổi kết quả:** max|diff| logits 2,5e-05, argmax khớp **100%** mọi dòng.
`mode="reduce-overhead"` nhanh thêm 5% nhưng dùng CUDA graph, xung khắc với DDP +
GradScaler → **dùng mode mặc định**, có trial fwd+bwd và fallback về eager.

### fp32 là hướng sai — đề nghị đảo lại quyết định cũ

fp32 chậm hơn fp16 **1,52×** → 50 round ~16 h, không thể vừa 8,29 h. Nó cũng **không**
chạm vào cơ chế NaN thật, lại **vứt mất `GradScaler`** — thứ tự động bỏ qua step khi
gradient không hữu hạn. Đó chính là lý do:

| run | precision | cơ chế bỏ step hỏng | chết ở |
|---|---|---|---|
| build 2 run 1 | fp16 AMP | ✅ GradScaler | round 25 |
| build 3 TPU | bf16 XLA | ❌ không có | **round 1** |

**Nhánh XLA nào về sau cũng phải tự thêm kiểm tra hữu hạn + bỏ step.**

### ⚠ Eq. (28) lệch scale 208× — CHẨN ĐOÁN NÀY CỦA TÔI SAI, đã bác bỏ ở mục 1i

Tôi từng viết rằng NaN đến từ việc Eq. (28) nối patch Haar thô (chưa chuẩn hoá) với hai
khối đã LayerNorm, gây lệch scale **207,8×**. **Con số đó là giả tạo:** `scale.py` tự cắm
một dòng ở mức cực đại vào cả 66 đặc trưng cùng lúc, điều không tồn tại trong dữ liệu thật.

Đo lại trên mẫu trải đều 15 shard của dữ liệu thật:

| | tôi tuyên bố | dữ liệu thật |
|---|---:|---:|
| lệch scale activation | 208× | **6,7×** |
| lệch gradient ở stem | (suy ra là lớn) | **1,5×** |

Chỉ 1,5× vì **BatchNorm ngay sau conv của stem hấp thụ** chênh lệch scale đầu vào. Eq. (28)
không phải thủ phạm. Nguyên nhân thật ở **mục 1i**, và nó nằm **trước** Eq. (28).

**Bài học:** đừng bao giờ dựng probe bằng dữ liệu tự chế khi dữ liệu thật có sẵn. Probe
tổng hợp của tôi đo một tình huống không tồn tại, rồi tôi ghi con số đó vào ba file.

### Vì sao TPU bị loại khỏi vai trò run chính

| | v2 GPU batch 4.096 | v3 TPU batch 16.384 |
|---|---:|---:|
| f1_macro round 0 | **0,83081** | **0,47856** |
| accuracy | 0,86545 | 0,51606 |
| trafficCongestionSybil F1 | 0,9669 | 0,5224 |
| timeDelayAttack F1 | 0,1948 | 0,0162 |

Round 0 của v3 hoàn tất sạch, không dính NaN — nên khoảng cách này **không** do phân kỳ.
Batch 16.384 chỉ cho 2.627 bước tối ưu mỗi epoch thay vì 10.509. TPU chỉ thắng nhờ batch
lớn, mà batch lớn phá kết quả. **Đây là cái giá tôi đã không lường khi chọn batch 16.384.**

---

## 1h. Cơ chế bỏ step hỏng cho XLA — và hai lỗ hổng test bắt được

`GradScaler` trên CUDA **bỏ hẳn** `opt.step()` khi gradient không hữu hạn. XLA không có
thứ đó. Bản TPU mới tự dựng lại, **hoàn toàn trên device, không đọc về host** nên không
làm cạn hàng đợi XLA mỗi step:

```python
gnorm = ||stack(||p.grad||)||
ok    = isfinite(gnorm)
scale = where(ok, clamp(clip/(gnorm+1e-6), max=1), 0)
p.grad = nan_to_num(p.grad) * scale        # nan_to_num TRƯỚC, vì 0*nan = nan
```

Test [`bench/guard.py`](papers/build2-paper-order/bench/guard.py) chạy ở local, và nó
**bắt được hai lỗ hổng mà đọc code không thấy**:

**1. BN buffer bị ghi bởi FORWARD, trước khi biết step hỏng.** Chặn gradient là vô nghĩa
nếu `running_mean`/`running_var` đã nhiễm nan. Khớp chính xác forensics run 1: **62/62 BN
buffer NaN** ở round 25. Đã thêm rollback snapshot:

```python
for b, sn in zip(BN_BUFS, BN_SNAPS):
    b.copy_(torch.where(ok, torch.nan_to_num(b), sn));  sn.copy_(b)
```

Sau khi thêm: BN buffer dịch **0,000e+00** trên step hỏng.

**2. AdamW vẫn co trọng số dù gradient bằng 0** — weight decay tách rời áp dụng vô điều
kiện. Đo được 1,19e-07 với `lr=1e-3, wd=1e-4`, đúng bằng cận `lr*wd`. Một guard không
sync **không thể** chặn cái này (muốn chặn phải đọc `ok` về host mỗi step). Chấp nhận và
ghi lại: chỉ xảy ra ở step hỏng, vốn hiếm.

Guard cũng được đối chiếu với `torch.nn.utils.clip_grad_norm_`: **max|Δgrad| = 0,000e+00**.

**Bài học chung:** "bỏ qua một step" không chỉ là bỏ `opt.step()`. Forward đã kịp ghi
BN buffer, và optimizer vẫn có đường tác động khác. Phải **đo trạng thái trước/sau**, đừng
đọc code rồi tin.

---

## 1i. Nguyên nhân phân kỳ THẬT — attention entropy collapse trong ViT

**2026-09-02, đo trên dữ liệu thật ở `/home/odixe/nckh/dataset`** (giống hệt bộ trên
Kaggle: 15 shard train + 4 shard test). Có dữ liệu rồi thì không cần giả thuyết nữa —
nạp thẳng checkpoint và đo.

### Bước 1 — round 24 đã hỏng, và hỏng ở fp32

Nạp `ckpt_round_024.pt` (bản forensics gọi là "sạch tuyệt đối"), chạy dữ liệu thật qua:

```
round   loss   max|w|  max|logit|   grad fp32        grad fp32 max   fp16 hỏng
    0  0,218    2,296        57,5        0,692                0,874      0/12
    5  0,356    8,895       122,4        2,630                3,268      0/12
   15  0,511   14,972       221,5        3,237                4,835      0/12
   21  0,480   17,618       226,5        4,091                5,254      0/12
   23  0,550   17,872       218,0        6,542               10,203      0/12
   24  0,539   17,814       209,5  1.608.282.820   19.299.225.600     12/12
```

Gradient nhảy **250 triệu lần trong một round, ở fp32**, trong khi loss, `max|w|` và logit
gần như không đổi. **Forward bình thường, backward nổ.** Nên không phải fp16, không phải
dải động, không phải BatchNorm (min running_var = 0,89, hoàn toàn lành).

### Bước 2 — khoanh vùng

| tham số | grad r23 | grad r24 | tỷ lệ |
|---|---:|---:|---:|
| `vit.0.att.in_proj_weight` | 3,04 | **1,265e+04** | 4.168× |
| `patch.proj.bias` | 0,15 | 4,26e+02 | 2.882× |
| `patch.proj.weight` | 0,51 | 4,25e+02 | 828× |
| `patch.pos` | 0,19 | 4,23e+02 | 2.269× |
| `vit.0.n1.weight` | 0,16 | 4,06e+02 | 2.464× |

Khối attention đầu tiên của ViT, và mọi thứ **phía trên** nó. Tất cả nằm **trước Eq. (28)**.

### Bước 3 — cơ chế

```
round   ||Wqkv||     |logit|max   softmax entropy   max prob
    0      21,3            268           0,686       1,0000
   15     112,4          2.393           0,528       1,0000
   21     171,9        374.029           0,249       1,0000
   22     200,2      1.135.014           0,217       1,0000
   23     261,6      5.324.782           0,073       1,0000
   24     526,5    150.130.144           0,001       1,0000
```

(entropy nếu attention đều trên 11 token = 2,398)

**Attention entropy collapse kinh điển.** Không có gì chặn logit: `q·k` lớn theo
`‖W_q‖·‖W_k‖`, Adam đẩy cùng hướng suốt ~260.000 bước, logit leo tới 1,5e8. Softmax bão
hoà tuyệt đối, và backward qua softmax ở thang đó sinh gradient 1e9. ViT pre-norm nên đầu
vào attention đã chuẩn hoá — cái không bị chặn là **trọng số chiếu**, không phải đầu vào.

### Một kết quả khoa học riêng, phải vào báo cáo

**`max prob = 1,0000 ngay từ round 0.`** ViT bão hoà từ đầu: nó chưa bao giờ thực hiện
attention có ý nghĩa, chỉ là argmax cứng trên 11 token. Bất kỳ tuyên bố nào về đóng góp
của khối ViT trong pipeline này đều phải kèm quan sát đó.

### Xác nhận thứ hai: c50 sụp đổ theo đúng kịch bản đó

`edl-cmso-veremi-c50` (fp16 AMP + `torch.compile`, batch 4.096, cùng mask và cùng
`extractor_init.pt`) phân kỳ ở round 22, và đường cong cho thấy nó bắt đầu hỏng từ
**round 14**:

```
round  0-13   f1 0,824 → 0,800    loss 0,224 → 0,058     ổn định
round 14      f1 0,732            loss 0,077   ← điểm gãy
round 15      f1 0,647            loss 0,447
round 17      f1 0,038
round 18-20   f1 0,372 → 0,695 → 0,695     (gượng dậy)
round 21      f1 0,433
round 22      NaN
```

Kiểu dao động vỡ-rồi-gượng-dậy này là chữ ký của attention bão hoà với gradient nổ, không
phải của một lỗi ngẫu nhiên. **Hai run độc lập, cùng một cơ chế, cùng một vùng round.**

Đo bổ sung (`bench/onset.py`): 2.000 bước đầu (0,19 epoch) thì `‖Wqkv‖` chỉ đi 14,2 → 15,0
và **0% gradient hỏng ở cả bf16 lẫn fp16+GradScaler**. Sự tăng trưởng cần ~150.000 bước
mới chạm ngưỡng — khớp với việc vỡ ở epoch 14–25, và loại bỏ nốt giả thuyết "GradScaler
đang âm thầm bỏ step ngay từ đầu".

### Cách sửa, xếp theo mức xâm phạm bài báo

1. **Tăng `weight_decay`** — bài báo **không quy định** giá trị này (1e-4 là tôi tự chọn).
   Ít xâm phạm nhất. Đang A/B ở local từ checkpoint round 21.
2. **QK-norm** (LayerNorm lên q, k trước tích vô hướng) — chặn logit theo thiết kế, chắc
   chắn hơn, nhưng là sửa kiến trúc, lệch Eq. (22)–(24).
3. Hạ LR — lệch Table 1, và chỉ trì hoãn chứ không chặn.

**Quyết định của bạn 2026-09-02:** để `edl-cmso-veremi-c50` chạy tiếp không sửa, chấp nhận
phân kỳ quanh round 24. Bản có fix để dành cho sau khi quota refresh 05-09.

---

## 1j. Máy local — ràng buộc phần cứng, ĐỌC TRƯỚC KHI VIẾT SCRIPT ĐO

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop, **4 GB VRAM**, sm_86 (Ampere: có bf16 + Triton) |
| RAM | **8 GB** — đây là ràng buộc chặt hơn VRAM |
| Disk | 936 GB trống |
| Dataset | `/home/odixe/nckh/dataset` — **giống hệt** bộ trên Kaggle, 15 shard train + 4 shard test |
| torch | `2.13.0+cu130` trong env `nckh`, cần `export CC/CXX` trỏ vào gcc của conda cho Inductor |

### ⚠ Lỗi đã làm CRASH máy, đừng lặp lại

`load_train` bản đầu gọi `pq.read_table(f)` cho từng shard — nạp trọn **2,7 triệu dòng**
vào RAM rồi mới lấy mẫu. Với 8 GB thì OOM killer giết tiến trình (**exit 137**) và làm
treo máy giữa chừng.

**Bản đã sửa** đọc theo **row group** (41 group × 131.072 dòng × 30 MB mỗi shard), không
bao giờ giữ quá một group, và có `max_rows=1_500_000` chặn cứng. Đo được: 300.000 dòng →
**peak RSS 0,74 GB**.

Hai chi tiết của loader, đừng vô tình bỏ:
1. **Phải trải mẫu qua nhiều row group.** Dòng trong một row group là liền kề, mà file
   nhóm theo receiver trong `(attack_type, scenario)` — lấy trọn 1 group chỉ ra **11/16
   lớp**. Lấy 16 group mỗi shard mới đủ 16 lớp.
2. **Loader trả về ÍT hơn số yêu cầu** (làm tròn theo group). Mọi script phải tính
   `N = len(X) // B` chứ đừng cắt theo số đã yêu cầu, nếu không batch cuối rỗng và GAT
   ném `cannot reshape tensor of 0 elements`.

**Quy tắc:** mọi script trong `bench/` phải khai báo rõ số dòng nạp và giữ RSS dưới ~1 GB.

---

## 1k. Cách sửa attention collapse — đã đo, đã áp dụng

Khởi từ `ckpt_round_021.pt` của build 2 run 1 (đã ở giữa quá trình sụp đổ, logit 3,74e5),
4.000 bước dữ liệu thật mỗi nhánh — [`bench/wdtest.py`](papers/build2-paper-order/bench/wdtest.py):

| `weight_decay` | `‖Wqkv‖` | `\|logit\|max` | softmax entropy | loss |
|---|---|---|---|---|
| 1e-4 (mọi run cũ) | 171,9 → 175,1 | 3,74e5 → 4,48e5 | 0,225 | 0,0291 |
| 1e-2 | 171,9 → 168,0 | 3,74e5 → 3,57e5 | 0,258 | 0,0201 |
| **5e-2** | 171,9 → **155,8** | 3,74e5 → **2,47e5** | **0,329** | 0,0299 |

Chỉ 5e-2 **đảo ngược** xu hướng, không mất gì về loss. Bài báo **không quy định** weight
decay — 1e-4 vốn là lựa chọn tự do của bản dựng — nên sửa nó không đụng Eq. (22)–(24).

**Quan trọng về phương pháp:** phải A/B từ checkpoint **đang hỏng**. 2.000 bước từ điểm khởi
tạo cho `‖Wqkv‖` 14,2 → 15,0 và **0% gradient hỏng ở mọi precision** — hoàn toàn vô hại,
trong khi lỗi thật cần ~150.000 bước mới hiện ra.

### Guard đã tối ưu trong `edl-cmso-v3-wd`

- `torch._foreach_norm` thay 132 op bằng **1** — bản cũ tốn ~78 ms/step, 40% thời gian step.
- **Bỏ BN rollback.** Nó tốn 189 op mỗi step để chống lại BN buffer không hữu hạn, mà BN
  buffer chỉ hỏng khi **activation** hỏng — điều chưa probe nào thấy (`repro.py`: "first
  non-finite module outputs: none"). Mọi giá trị hỏng đã quan sát đều sinh ra ở backward,
  nơi gradient guard đã phủ.

---

## 1l. Round 0 hỏng trên TPU — nguyên nhân: XLA autocast chạy LayerNorm ở bf16

**Hiện tượng.** Ba run TPU đều báo phần lớn step của round 0 có gradient không hữu hạn:

| run | weight_decay | % step hỏng | `steps_ok` |
|---|---|---:|---:|
| `edl-cmso-v3-b4096` | 1e-4 | 72,06% | 2.936 |
| `edl-cmso-v3-wd` | **5e-2** | 66,10% | 3.563 |

`steps_ok` ~3.000–3.500 ở cả hai → lỗi bắt đầu quanh **step ~3.000**, và **weight decay
không sửa được nó**.

### Đã loại trừ bằng đo đạc

1. **Không phải attention collapse (mục 1i).** Cái đó cần ~150.000 bước; đây xảy ra ở
   ~3.000. Ở round 0 logit mới ~268 và entropy 0,686 — chưa sụp.
2. **Không phải bf16.** `bench/precnan.py`: tại init, 53 batch thật, **0% hỏng ở cả bốn**
   chế độ fp32 / bf16 / fp16 / fp16+GradScaler (median ‖g‖ 0,53).
3. **Không phải dòng dữ liệu cực trị.** `bench/extremes.py` quét cả **43.045.415 dòng**:
   chỉ **16 dòng** có |x| > 400. Một batch gồm 4.096 dòng cực đoan nhất (|x| tới 570,4)
   vẫn cho gradient **hữu hạn 1,93**.
4. **Không tái hiện được trên CUDA.** `bench/onset2.py`: cùng `extractor_init.pt`, cùng
   batch 4.096, cùng bf16, chạy **9.000 step** — **0 gradient hỏng**, `‖Wqkv‖` chỉ đi
   14,4 → 17,0. Trùm toàn bộ vùng TPU vỡ.

**Kết luận:** nguyên nhân **đặc thù TPU/XLA**, không phải của model, dữ liệu hay precision. Đã tìm ra — xem dưới.

### ✅ ĐÃ TÌM RA: XLA autocast chạy LayerNorm ở bf16, CUDA thì không

`edl-cmso-v3-diag` (notebook chẩn đoán, host-sync mỗi step) bắt đúng step hỏng đầu tiên:

```
first_bad_step: 3469
  logits_finite : false      ← FORWARD đã hỏng, không phải backward
  max_abs_logit : inf
  activation hỏng đầu tiên: vit.0.n2   ← một LayerNorm
  max_abs_weight: 1,197      ← trọng số hoàn toàn lành
  Wqkv_norm     : 29,06      ← không phải attention collapse (round 24 là 526)
```

Mảnh ghép cuối, đo ở local:

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    layer_norm(x).dtype   # torch.float32   ← CUDA BẢO VỆ bằng policy
    softmax(x).dtype      # torch.float32   ← CUDA BẢO VỆ
    linear(x).dtype       # torch.bfloat16
```

**XLA autocast không có chính sách đó.** Mà LayerNorm **bình phương** đầu vào để tính
phương sai, còn bf16 trần 3,39e38:

| `|x|` | LayerNorm bf16 |
|---|---|
| 1e18 | 4,56 — bình thường |
| 1e19 | **0,0000** — phương sai tràn, mất sạch tín hiệu, vẫn "hữu hạn" |
| 1e20 | **NaN** |

Dòng 1e19 đáng sợ nhất: lớp đó **âm thầm trả về toàn số 0**, không exception, không
tripwire nào kêu, model chỉ đơn giản ngừng học qua nhánh đó.

**Cách sửa đã kiểm tra trong `edl-cmso-v3-fp32ln`:** mirror đúng policy của CUDA — chạy toàn bộ
§4.8 ở fp32, DAGSNet giữ bf16.

```python
with torch.autocast(x.device.type, enabled=False):
    Fm = self.fuse(x.float()).index_select(1, self.sel_ch)
```

**Quy tắc rút ra: đừng bao giờ giả định policy của autocast chuyển được giữa các backend.**
Cùng code, cùng số, khác backend → khác dtype → khác kết quả. "Chạy được trên CUDA" là
**bằng chứng bằng không** về XLA.

### Lỗi phụ đã vá: `FileNotFoundError` làm Kaggle báo "failed"

Run dừng trong round 0 không ghi `history.csv`; cell kết quả đọc nó → papermill biến
thành execution error → Kaggle đánh dấu **failed** dù tripwire đã làm đúng việc. **Hai run
bị gán nhãn sai vì lý do này.** Đã bọc cell 9/10: nếu thiếu `history.csv` thì in
`diverged.json` và đuôi các file `.jsonl` thay vì ném lỗi.

## 1m. TPU bị loại vĩnh viễn khỏi vai trò run 50 round — số học, không phải thiếu tối ưu

Run `fast50` (2026-09-03) đóng lại câu hỏi "có tối ưu thêm được không". **Không.**

### Ba số đo đóng vấn đề

| phạm vi fp32 trong §4.8 | ms/step (v5e-8, batch 4.096) | round 0 |
|---|---:|---|
| không có (bf16 thuần, probe) | **148,3** | LayerNorm tràn ở step 3.469 |
| toàn §4.8 (`fp32ln`) | 303,0 | sạch qua 70.304 step |
| chỉ LayerNorm+softmax (`fast50`) | **310,1** | grad 3,9×10¹³, loss tăng, skip 0% |

`fast50` thu hẹp phạm vi fp32 vì probe RTX 3050 dự đoán nhanh hơn **1,182×**. Đo thật:
**chậm hơn 2,3%**, và mất luôn ổn định. Chi phí không tỉ lệ với khối lượng tính ở fp32
(LayerNorm chỉ chạm ~5,8 triệu phần tử) mà nằm ở **ranh giới precision phá vỡ fusion**
trong đồ thị đã bị dispatch chi phối — một ranh giới hay nhiều ranh giới đều phạt ~2×.

> **Đây là mục 1l ở trục tốc độ.** Probe CUDA không phải bằng chứng về XLA, kể cả về
> throughput. Quy tắc "đo local trước khi tiêu quota" chỉ đúng GPU→GPU.

### Số học quota — lịch batch không mua thêm được gì

Step time gần như không đổi từ batch 4.096 đến 65.536 → **TPU cho số bước tối ưu cố định mỗi
giờ, bất kể batch** (~11.600 bước/h ở 305 ms/step). Tổng thời gian = tổng bước × step time,
nên **tổng số bước bị ngân sách thời gian ấn định**; batch chỉ phân bổ lại số bước đó lên 50
epoch. Với 7,86 h → ~91.000 bước.

| batch | bước/round | 50 round | |
|---:|---:|---:|---|
| 4.096 | 10.509 | 45,3 h | vượt 5,8× |
| 16.384 | 2.627 | 11,3 h | vượt |
| **32.768** | 1.313 | **5,7 h** + 0,8 h eval | vừa |

Batch nhỏ nhất vừa quota: **26.389**. Nhưng batch 16.384 đã đo `f1_macro` r0 **0,479 vs
0,831**. **Batch chính là thứ đang bị đánh đổi.** Cùng 7,86 h: **65.650 bước trên TPU** hoặc
**525.450 bước trên 2×T4** ở batch 4.096 (565,7 s/round). Gấp 8×. Chấm hết.

### Cơ chế hỏng thứ ba: gradient hữu hạn nhưng nổ

23 heartbeat, `skip_pct = 0` toàn bộ, trong khi:

| step | loss | grad | \|logit\|max | ‖W_qkv‖ | max\|w\| |
|---:|---:|---:|---:|---:|---:|
| 250 | 0,5516 | 1,01 | 25,23 | 14,12 | 1,118 |
| 500 | 1,2094 | 1,84×10⁵ | 6.195,04 | 22,54 | 1,223 |
| 5.750 | 1,6834 | **3,92×10¹³** | 15.262,15 | 29,12 | 1,144 |

Guard co gradient về norm 1; **Adam bất biến với phép co đồng nhất**, nên hướng đã hỏng được
áp dụng nguyên vẹn. Tripwire kiểm tra `isfinite` **không thấy gì**. Đây là kiểu hỏng đắt nhất:
run vẫn "khỏe" trên mọi chỉ báo có sẵn và đốt hết quota.

**Giả thuyết đã bác bỏ:** "fp16 trần 65.504 là tripwire tình cờ, bf16 gỡ mất nó".
`bench/bf16repro.py` — 900 step, dữ liệu thật, batch 1.024, 4 nhánh (bf16+clip,
fp16+GradScaler, bf16+clip+ngưỡng, fp32+clip): **cả bốn đều khỏe**, loss 2,84 → 0,26,
grad ~0,8, `|logit|max` 2,5 → 33, 0 skip. Trùng `bench/onset2.py` (9.000 step bf16 CUDA).
**Kiểu hỏng này không tái hiện được ngoài XLA.**

### Bài học vận hành — trạng thái session

`kaggle kernels status` báo `COMPLETE` trong suốt lúc TPU **đang train**: nó báo trạng thái
**version batch mới nhất**, không phải session còn sống hay không. Notebook CPU stop đẩy làm
version mới chính là thứ nó đang báo. Bit đáng tin duy nhất là
**`get_accelerator_quota` → `time_reserved > 0`**, xác nhận bằng `_timestamp` W&B còn mới và
`progress/global_step` còn tăng. W&B thì ngược lại: kernel bị giết cứng không chạy
`wandb.finish()` nên run treo ở `running` vĩnh viễn.

**Và: ghi chép "run dừng ở step 2.250" là SAI** — đó là một lần đọc giữa run. 23 heartbeat
cách đều 76–78 s, step 250 → 5.750, không có khoảng trống: một lần thực thi liên tục. Khi đọc
W&B của một run đang chạy, **luôn nói rõ đó là ảnh chụp tại thời điểm nào**, đừng ghi thành
điểm kết thúc.

---

## 1n. Build 4 — DAGSNet trần, ablation còn thiếu của cả dự án (2026-09-04)

Mục 6 và report §6 đều ghi "đóng góp riêng của CMSO và ViT cần ablation chưa chạy". Build 4
**chính là ablation đó**: bỏ hẳn §4.8 (DWT→ViT→GAT→Eq.28) và §4.9 (CMSO), đưa 66 đặc trưng
thô thẳng vào DAGSNet.

### Hai quyết định người dùng đã chốt — đừng hỏi lại

| | |
|---|---|
| **Layout** | `patch (B,6,11)` — 66 đặc trưng thành 11 vị trí × 6 kênh, **đúng lưới `PatchEmbed` đang chia**, đưa vào thô. Chỉ stem đổi 133→6 kênh; mọi nhánh giữ nguyên hình học |
| **Tham số** | **giữ nguyên siêu tham số** của build 1–3, không khớp tổng số tham số. 395.024 vs 727.952 — ViT+GAT bị bỏ thật, không resize gì |

### Đo trước khi chọn — hai layout kia đã bị loại bằng số

`bench/dagsonly.py`, RTX 3050, batch 1.024, dữ liệu thật. Tỉ lệ chuyển được GPU→GPU:

| layout | tốc độ | s/round trên 2×T4 | 50 round | params |
|---|---:|---:|---:|---:|
| **patch (B,6,11)** | **1,89×** | **299,3** | **4,16 h** | 395.024 |
| seq (B,1,66) | 0,76× | 740,9 | 10,29 h | 393.104 |
| chan (B,66,1) | 3,09× | 182,8 | 2,54 h | 418.064 |
| *full pipeline* | 1,00× | 565,7 | 7,86 h | 727.952 |

**`seq` CHẬM HƠN cả pipeline đầy đủ** — 66 vị trí conv thay vì 11. Trực giác "bỏ bớt model
thì nhanh hơn" sai ở đây. `chan` nhanh nhất nhưng conv trên trục dài 1 suy biến thành 1×1 →
DAGSNet biến thành MLP, không còn là DAGSNet.

### Vì sao vừa quota 4,64 h

**Không có CMSO thì prep chỉ còn ~3,2 phút.** Build 2 run 1: meta.json xong ở phút 3,2, nhưng
`prep took 80.2 min` — tức **77 phút là CMSO**. Build 4 không có bước đó.
→ 0,06 h prep + 4,16 h train = **4,22 h** trong 4,64 h. `max_hours = 4.35`.

### Một deadlock TÔI SUÝT ĐẨY LÊN — và cách bắt được

Bản đầu của tôi để **chỉ rank 0** đánh giá điều kiện abort. Nếu rank 0 thoát vòng lặp một
mình, `backward()` kế tiếp của rank 1 **treo vĩnh viễn** trong all-reduce gradient của DDP —
đốt sạch budget bằng một cái treo. Đúng lỗi mục 5 đã cảnh báo, ở dạng mới.

Sửa: cả hai rank cùng đến `all_reduce(MAX)` trên cờ abort tại **cùng một step**, rồi cùng
break. `smoke.py` của build 4 có test chạy **2 process gloo thật** ép rank 0 abort và bắt
buộc cả hai phải thoát trong 180 s — nếu treo thì test fail.

### ⚠ Kết quả round 0 — đảo ngược giả định trung tâm của cả dự án

| run | `f1_macro` r0 | tham số |
|---|---:|---:|
| **Build 4 — DAGSNet trần** | **0,84694** | **395.024** |
| Build 2 run 1 — pipeline đầy đủ | 0,83081 | 727.952 |
| Build 2 run 3 (c50) | 0,82422 | 727.952 |
| Build 1 | 0,81676 (r2) | 823.824 |

**Đủ 46 round.** r0 0,84694 · r2 0,85137 · **r5 0,85320 (đỉnh)** · r10 0,84242 · r20 0,84215 ·
r30 0,84016 · r40 0,83601 · r45 0,83479. Train loss giảm đơn điệu 0,2139 → 0,0499 suốt 46
round trong khi f1 đi xuống từ r5 — overfitting đúng sách vở. Kết luận "đỉnh sớm rồi overfit"
ở report mục 6 **vẫn đúng**, chỉ dời từ r0–2 sang r5. ⚠ Nhận định sớm của tôi (lúc mới 4
round) rằng mục 6 phải viết lại là **quá mạnh** — đã sửa.

**Điểm mạnh nhất: round CUỐI của build 4 (r45, 0,83479) — sau 46 epoch overfit — vẫn cao hơn
mức tốt nhất pipeline đầy đủ từng đạt (0,83081).**

**Bỏ 332.928 tham số (45,7%) thì `f1_macro` TĂNG 0,01613.** Nhất quán với mục 1i: ViT bão hoà
từ round 0 (`max prob = 1,0000`) — 36,4% tham số nằm trong khối gần như không làm gì, và chính
khối đó gây cả hai cơ chế phân kỳ. Ablation nói thêm: nó còn **làm giảm điểm**.

**Per-class giải thích cơ chế (report mục 2.9).** 11/16 lớp tăng, và **bốn mức tăng lớn nhất
rơi đúng vào bốn lớp khó nhất**: `suddenConstantSpeed` +0,1470 · `positionMirroring` +0,0714 ·
`constantSpeedOffset` +0,0571 · `timeDelayAttack` +0,0333. Năm lớp giảm đều đã gần trần
(`dosAttack` 0,995, `randomPositionOffset` 0,975…), mức giảm 0,001–0,017 — nhỏ hơn một bậc.
Giả thuyết: §4.8 nén 66 đặc trưng qua DWT→ViT(bão hoà)→GAT→Eq.28 rồi CMSO cắt còn 133/262
kênh; lớp dễ không hại gì, **lớp khó mất đúng tín hiệu mảnh mà DAGSNet cần**. Đó cũng là lý do
`f1_macro` tăng mạnh hơn `accuracy`.

**Chưa kết luận mạnh được.** Một seed, một run, và ablation này đo đóng góp **gộp** của
§4.8+§4.9 — muốn tách riêng CMSO cần thêm run có §4.8 nhưng không §4.9. Chi tiết + caveat:
report.md mục 2.8 và 2.9.

Sức khoẻ: `GradScaler` bỏ 3/10.508 step ở r0 (0,03%, bình thường), grad norm 0,78 → 0,47,
`max|w|` 1,12 → 1,73. Không có dấu hiệu của hai cơ chế phân kỳ mục 1i/1l — đúng như dự đoán
vì cả hai đều nằm trong attention.

**Đã dừng sạch ở round 45 (46 round), đúng như dự đoán.** `logs/stopped_early.json`:
`reason: session wall-clock budget`, `next_round: 46`, `round_seconds: 333.1`. Artifact đã kéo
về `papers/build4-dagsnet-only/runs/` (612 MB) và **khớp chính xác** với W&B. Đã xác minh
đầy đủ: **46/46 round có đủ bộ checkpoint + metrics + confusion + report**, `last.pt` ở round
45 với trọng số hữu hạn, và confusion round 5 cộng đúng **10.761.343** — tức eval quét trọn
test set, không pad không bỏ đuôi.

**Bốn round còn lại (46–49)** resume được: push lại notebook với output của run này attach làm
data source. Cần ~0,5 h GPU — chờ refresh 2026-09-05 hoặc dùng `minhtran0601`.

**Sức khoẻ:** `GradScaler` bỏ 198/483.368 step = **0,041%** (bình thường); grad norm 0,2–0,8;
`scaler_scale` tăng 65.536 → 262.144. Không dấu vết nào của hai cơ chế phân kỳ mục 1i/1l —
đúng như dự đoán vì cả hai đều nằm trong attention.

### `papers/build4-dagsnet-only/`

```
notebook-dagsnet-r0-45/edl_cmso_v4_dagsnet.ipynb   18 cell, đã push
notebook-dagsnet-r0-45/kernel-metadata.json        is_private: true, KHÔNG kernel_sources
smoke.py                                      ★ 56 check, chạy trước mọi push
```

**Build 4 KHÔNG kế thừa gì.** `channel_mask.json` và `extractor_init.pt` mô tả một phép chiếu
§4.8 mà model này không có. Launch cell có assert **cấm** file mask tồn tại.

### Ba cơ chế NaN đã đưa vào notebook

1. **CE ở fp32 ngoài autocast** — softmax trên 16 logit fp16 làm tròn xác suất về đúng 0,
   `log(0) = -inf`.
2. **Tripwire cắt TRƯỚC eval và TRƯỚC khi ghi checkpoint** — run 1 mất 4,1 h, 20 checkpoint
   NaN, `last.pt` bị ghi đè bằng trọng số NaN nên không resume được.
3. **Đếm step bị `GradScaler` bỏ** — nó bỏ step non-finite và **không báo gì**; chỉ có việc
   scale bị chia đôi là quan sát được. c50 chưa có phần này.

## 1o. Build 2 final — bản v2 đã dựng lại, chưa push

`papers/build2-paper-order/notebook-final-wd5e2/` — 22 cell, `smoke.py` 40/40. Khác c50 đúng
bốn chỗ, mọi thứ khác giữ nguyên:

| | c50 | v2-final | vì sao |
|---|---|---|---|
| `weight_decay` | 1e-4 | **5e-2** | cách sửa attention collapse đã đo (1k) |
| quan sát | JSONL | **+ W&B trực tiếp** | Kaggle không công bố gì đến khi kernel dừng |
| abort trong round | không | **skip rate + `\|logit\|max` > 1e6** | check theo round tốn nguyên một round |
| `max_hours` | 8,05 hằng số | **suy từ quota còn lại** | hằng số bị giết trước cả clean stop của chính nó |

`abort_logit_max = 1e6` là tripwire cho **đúng** cơ chế đã giết run 1 và run 3: `|logit|max`
lên 1,5e8 nhưng **vẫn hữu hạn**, nên tripwire `isfinite` không bao giờ thấy.

### ĐÃ PUSH VÀ ĐANG CHẠY (2026-09-04)

`minhtran0601/edl-cmso-v2-final` — version 1, `RUNNING`, quota GPU bắt đầu tiêu.

Ba việc phải làm khi push, đã làm đủ, ghi lại vì lần sau vẫn phải làm:

1. `id` trong `kernel-metadata.json` đổi `odixe0502/` → `minhtran0601/`. Push giữ nguyên id
   trong file, nên bỏ qua bước này là đẩy nhầm sang tài khoản kia (và bị từ chối).
2. `embed_wandb_key.py` — placeholder đã thay bằng credential inline.
3. `is_private: true` xác minh ngay trước khi push.

`kernel_sources = odixe0502/edl-cmso-veremi-v2` **truy cập được** từ `minhtran0601` (đã thử
`kaggle kernels status`, trả về COMPLETE) — nên `channel_mask.json` và `extractor_init.pt`
vẫn là **đúng những file cũ**, curve vẫn so sánh được với run 1 round 0–24.

**`max_hours` 8,05 → 11,0.** Lý do đổi: `minhtran0601` có quota GPU **đầy 30,0 h, đã dùng 0 s**
(đo 2026-09-04), nên ràng buộc không còn là quota mà là **trần session 12 h**. 50 round =
7,86 h huấn luyện + ~0,3 h prep, thừa ~2,8 h — run này **không cần resume**. Đây là lần đầu
trong cả dự án budget không phải là thứ cắt ngang run.

---

---

## 1p. Build 4 ĐÃ ĐỦ 50/50 ROUND, và cây thư mục đã được tổ chức lại (2026-09-04)

4 round cuối chạy trên `minhtran0601/edl-cmso-v4-dagsnet-r46`, resume từ checkpoint round 45
của `odixe0502/edl-cmso-v4-dagsnet` (người dùng đã share notebook đó với group `nckh_minhtriet`
— sau đó `kaggle kernels status` trả COMPLETE thay vì denied).

* best **r5 `f1_macro` 0,85320** — không đổi
* last **r49 0,83648**, hội tụ: r46–r49 nằm gọn trong dải 0,0018, LR cosine đã về 0
* round 0–45 của bản pull sau **trùng khít từng chữ số** với bản pull trước → resume thật
* tổng 4,61 h, 331,8 s/round trung bình, 129.764 mẫu/s

### Hai session, MỘT thư mục run

Yêu cầu của người dùng: dù chạy hai session, kết quả cuối phải trông như một lần chạy. Đã ghép
bằng `scripts/merge_sessions.py` (script mới, có trong cả hai bộ skill):

```
papers/build4-dagsnet-only/runs/edl_cmso_v4_dagsnet/
    checkpoints/ metrics/ preds/ confusion/ reports/   round 0..49 liên tục
    meta.json  config.json
    logs/sessions.json        session nào sinh round nào + 3 trường config khác nhau
    logs/sessions/{1,2}/      log kernel + heartbeat + config RIÊNG của từng session
```

Script từ chối ghép nếu artifact chồng lấn khác byte (nghĩa là session sau huấn luyện lại chứ
không resume), nếu `history.csv` bất đồng ở round chung, nếu `meta.json` khác, hoặc nếu dãy
round bị hổng. `logs/` và `config.json` **không** gộp — mỗi session có `heartbeat.jsonl` riêng
và chỉ session hết budget mới có `stopped_early.json`; ba trường khác nhau giữa hai session là
`max_hours`, `deadline_ts`, `wandb_run_id`, mọi siêu tham số khoa học đều giống hệt.

⚠ **Bản đầu của script bỏ sót file ở gốc run dir**, nên thư mục ghép ra không có `meta.json`.
Đã vá và lấy lại `meta.json` + `config.json` bằng `mcp download_notebook_output` với `filePath`
— tải đúng một file thay vì kéo lại 958 MB. Cách này đáng nhớ.

### ⚠ Lần push đầu HỎNG và đốt 1,9 h quota — đọc kỹ, lỗi này sẽ lặp lại

Nó **không resume**: `attached last.pt : none` → `[resume] no checkpoint found — starting from
round 0` → huấn luyện lại tới round 17 rồi hết `max_hours`.

**Kaggle đã đổi layout `/kaggle/input`.** Đường dẫn thật:

```
/kaggle/input/notebooks/<owner>/<slug>/runs/<run_name>/checkpoints/last.pt
/kaggle/input/datasets/<owner>/<slug>/<upload>/...
```

Ba tầng trước khi tới cây của run. Glob tìm dataset dùng `**` đệ quy nên vẫn chạy; glob tìm
checkpoint viết cứng tối đa hai `*` nên không với tới.

**Lỗi đắt hơn không phải cái glob mà là việc thiếu checkpoint chỉ là một dòng cảnh báo.** Nay
có `REQUIRE_RESUME = True` giết run ngay ở cell 4, trước bước đọc parquet.

### Ba bài học vận hành khác từ lần này

**Hai bản skill đã trôi khỏi nhau.** `.agents/.../notebook-template.md` là bản MỚI HƠN và đã
có sẵn cách đúng (`rglob("last.pt")`, "never guess between candidates"); `.claude/...` là bản
cũ, và notebook của tôi dựng theo bản cũ đó. Đã đồng bộ. **Trước khi dựng notebook, so hai bản
trước đã.** Lưu ý: `kaggle-mcp.md` thì khác nhau *có chủ đích* (bản `.agents` viết cho Codex),
đừng ép đồng bộ file đó.

**`smoke.py` trỏ vào notebook mặc định, không phải notebook sắp push.** Build 4 mặc định
`notebook_dagsonly/`, build 2 mặc định `notebook/` — nên cả hai lần tôi báo "pass" đều là kiểm
nhầm file, kể cả lần trước khi push v2-final. Nay cả hai nhận biến môi trường
(`EDL_V4_NB_DIR`, `EDL_NB`) và build 2 mặc định trỏ `notebook-final-wd5e2/`. **Một notebook đã
vá mà smoke không đọc tới thì coi như chưa kiểm.**

**Thay chuỗi hàng loạt là con dao hai lưỡi.** Khi đổi tên thư mục tôi thay `"notebook"` →
`"notebook-run1-baseline"` và nó đụng vào `"kernel_type": "notebook"` trong **mọi**
`kernel-metadata.json`. Đã phát hiện và sửa ngay, nhưng nếu push trước khi kiểm thì Kaggle từ
chối cả 8 notebook. Sau mỗi lần thay hàng loạt: parse lại JSON và assert bất biến.

---

## 1q. Dọn kho, đổi tên, và BẢN FINAL ĐÃ XONG (2026-09-04)

### ★ `edl-cmso-v2-final` chạy trọn 50/50 round — bài toán ổn định đã đóng

Kernel `minhtran0601/edl-cmso-v2-final`, COMPLETE, 8,26 h, 594,4 s/round.

* best **r4 `f1_macro` 0,82495**, last **r49 0,80288**
* `n_selected = 133` → đúng mask kế thừa từ run 1; `weight_decay = 0.05` đúng như đặt
* `last.pt` ở round 49, toàn trọng số hữu hạn; confusion r4 và r49 đều cộng đúng 10.761.343

**Bằng chứng cơ chế, đọc từ 2.100 heartbeat:**

| | run 1 (chết r25) | final (50/50) |
|---|---|---|
| `‖W_qkv‖` | 21,3 → **526,5** | 14,2 → đỉnh 41,1 → **về 30,8** |
| `\|logit\|max` | 268 → **1,5e8** | 14,9 → đỉnh 92,9 |
| entropy | 0,686 → **0,001** | 1,073 → 0,530 |
| grad norm | **1,6e9** | đỉnh 2,88 |
| skip% | — | đỉnh 1,2%, kết 0 |

Điểm quyết định không phải `‖W_qkv‖` nhỏ mà là **nó lên đỉnh rồi đi xuống** — đúng dự đoán của
`wdtest.py`, nay xác nhận ở quy mô 50 epoch.

**Giá phải trả:** đỉnh 0,82495 < 0,83081 của run 1. Nhưng 0,83081 là số round 0 của một run
về sau tự huỷ; **0,82495 là con số đứng vững duy nhất của pipeline đầy đủ.**

⚠ **Quyết định nâng `max_hours` 8,05 → 11,0 là cần thiết.** Config thực chạy `max_hours 10.65`,
run dùng 8,26 h. Nếu giữ 8,05 (ngân sách 7,70 h) thì nó đã bị cắt quanh round 45.

### ⚠ Per-class ĐỔI KẾT LUẬN khi so với chuẩn đứng vững

So build 4 (r5) với **bản final (r4)** thay vì run 1 (r0):

* **15/16 lớp tăng**, trung bình **+0,0282** (trước đó, so với run 1, chỉ 11/16)
* lớp duy nhất **giảm** là `timeDelayAttack`: 0,2868 → 0,2281 (**−0,0588**)
* so với run 1 thì lớp này lại **tăng** +0,0333

Nghĩa là phần lớn năng lực của run 1 trên lớp khó nhất đến từ **trạng thái chưa hội tụ, chưa
bị weight decay ràng buộc** — đúng trạng thái mà 24 round sau đó tự phá huỷ. **Đừng xây kết
luận trên con số mà run tạo ra nó không giữ nổi.**

Kết luận mới, hẹp hơn nhưng chắc hơn: **§4.8 không vô dụng — nó là thứ duy nhất giúp được
`timeDelayAttack` — nhưng cái nó mua rất hẹp, và giá là 332.928 tham số cộng điểm thấp hơn ở
15 lớp khác.** Đây là hướng nghiên cứu tiếp có cơ sở nhất: DAGSNet trần + một nhánh thời gian
riêng cho lớp đó.

### Tên thư mục theo ĐẶC ĐIỂM, không theo số phiên bản

```
papers/build1-cmso-before-extractor/     CMSO chạy TRƯỚC extractor (ngược thứ tự bài báo)
    notebook-cmso-first/  runs/edl_cmso_r50_b4096/  figures/
papers/build2-paper-order/               đúng thứ tự §4.8 -> §4.9 -> §4.10
    notebook-run1-baseline/              run 1, wd 1e-4
    notebook-run3-compiled/              run 3 (c50), + torch.compile
    notebook-fp32-probe/  notebook-tpu-probe/
    notebook-final-wd5e2/                ★ bản cuối: wd 5e-2 + W&B + abort trong round
    run1-wd1e4-diverged-r25/             run 1 — phân kỳ round 25
    run3-c50-compiled-diverged-r22/      run 3 — phân kỳ round 22
    run4-wd5e2-complete-50round/         ★ run pipeline đầy đủ DUY NHẤT trọn 50 round
    bench/                               18 script đo, GPU local, 0 quota
papers/build4-dagsnet-only/              bỏ hẳn §4.8 và §4.9
    notebook-dagsnet-r0-45/  notebook-dagsnet-resume-r46-49/
    runs/edl_cmso_v4_dagsnet/            MỘT run 50 round, ghép từ hai session
figures/  make_figures.py  make_tables.py
```

### Đã xoá, có chủ đích — 4,9 GB → 1,5 GB (đã cộng cả bản final 512 MB)

* **Toàn bộ `papers/khan-2025-edl-cmso-v3/`** (365 MB): dòng TPU, loại vĩnh viễn ở mục 1m.
* **`checkpoints/` + `preds/`** của build 1 (1,6 GB), build 2 run 1 (761 MB), c50 (565 MB).
* **`prob_round_004.npz` + `prob_round_049.npz`** của bản final (344 MB mỗi file): **tái tạo
  được** từ checkpoint đã giữ bằng một lượt eval. Nguyên tắc đáng nhớ: *thứ gì tái tạo được từ
  một checkpoint đã giữ thì không đáng chiếm hàng trăm MB.*

**Giữ nguyên:** `metrics/`, `confusion/`, `reports/`, `logs/`, `meta.json`, `config.json`,
`channel_mask.json`, `extractor_init.pt`, toàn bộ checkpoint của **bản final** và **build 4**,
và mọi `round_*.npz` argmax của bản final (cần cho ablation Sybil).

### report.md đã tái cấu trúc, và giờ TỰ SINH LẠI ĐƯỢC

* bỏ mục 2.5, 2.6 (kết quả TPU), 3.6, 4.3, 4.6, 4.7, 4.8 → gộp thành **một** mục 4.6
* mục 2.2 giờ là 10 metric của **bản final** (chuẩn đứng vững), c50 chỉ còn một dòng đối chiếu
* mục 2.7 per-class đã đổi chuẩn sang bản final — xem cảnh báo ở trên
* **mục 2.8 — bảng theo từng round của cả 5 run**, nằm giữa hai mốc HTML comment
* 5 hình trong `figures/`

```bash
python make_figures.py            # sinh lại 5 hình
python make_tables.py --apply     # sinh lại mục 2.8 giữa hai mốc
```

⚠ **Quy ước trích dẫn trong report, đừng phá:** `§4.8/§4.9/§4.10` (không có chữ "mục") là mục
của **bài báo**; `mục 2.1`, `mục 4.6`… là mục của **report**. Hai hệ trùng số. Khi sửa report
bằng thay chuỗi, `§4.10` → `mục 4.7` sẽ phá các trích dẫn DAGSNet của bài báo — đã mắc một lần.

---

## 1r. Kaggle OAuth — vì sao token "mất quyền", và cách đã tự động hoá (2026-09-06)

### Nguyên nhân thật: lỗi dấu trong kagglesdk, không phải mất quyền

`~/.kaggle/credentials.json` giữ **`refresh_token` bền** (`KGRT_…`) + **`access_token` ngắn
hạn**. Login qua trình duyệt cho access token ~3 h; làm mới tường minh xin
`DEFAULT_ACCESS_TOKEN_EXPIRATION = 12 h` và được cấp.

Đọc `kagglesdk/kaggle_creds.py`:

```python
def access_token_has_expired(self):
    return not self._access_token_expiration or \
           self._access_token_expiration < datetime.now(timezone.utc) - timedelta(minutes=30)
```

**Biên 30 phút đặt nhầm phía.** Đúng ra phải làm mới *sớm* (`< now + 30min`); như viết thì nó
chỉ làm mới khi token đã quá hạn *thêm* 30 phút. Trong cửa sổ đó CLI vẫn gửi access token đã
chết và server trả `Permission '<x>' was denied` — **đọc như bị thu hồi quyền**, nên phản xạ
tự nhiên (bảo người dùng login lại bằng tay) là sai và tốn công họ vô ích.

**Dấu hiệu nhận biết:** CLI báo permission denied **trong khi MCP vẫn chạy bình thường** →
access token hết hạn, KHÔNG phải mất quyền. Đây đúng là chuyện xảy ra 2026-09-05.

`refresh_access_token()` ghi lại **cùng** refresh token — không xoay vòng — nên làm mới bao
nhiêu lần cũng an toàn, và snapshot cũ không bị "hỏng" vì một lần refresh mới.

### Đã thêm

* `kaggle_account.py refresh` — mint token 12 h ngay, rồi cập nhật snapshot. Không phụ thuộc
  phép kiểm hỏng ở trên. Báo phân biệt rõ trường hợp refresh token **thật sự** bị thu hồi.
* `use <account>` giờ tự gọi `refresh` sau khi swap, vì snapshot mang theo access token cũ
  gần như chắc chắn đã hết hạn.
* **`.claude/settings.json` — hook `SessionStart`** chạy `refresh` mỗi lần mở session. Đây là
  thứ khiến người dùng không phải làm gì bằng tay nữa. Lưu ý hook phải `conda activate nckh`
  trước, vì `kagglesdk` nằm trong env đó chứ không phải interpreter mặc định.

Chỉ còn **một** trường hợp cần login trình duyệt: refresh token bị thu hồi thật (chủ tài khoản
thu hồi, hoặc Kaggle hết hạn sau thời gian dài không dùng).

---

## 1s. ARCHITECTURE.md của DAGSNet (2026-09-06)

`papers/build4-dagsnet-only/ARCHITECTURE.md` — đặc tả mô hình classifier được chọn cho nghiên
cứu. Tự chứa: hợp đồng vào/ra, 66 cột + `mean`/`std_used`, 16 lớp, bảng 69 layer, Eq. 38–48,
thống kê trọng số trước/sau huấn luyện, và **mã nguồn đầy đủ để chạy ở project khác**.

Checkpoint được chọn: **round 5** (`f1_macro` 0,853199) — người dùng đã chốt.

**Kiểm chứng mạnh nhất, làm lại được:** trích khối code ra khỏi file markdown, chạy với
`ckpt_round_005.pt`, logit **trùng khít từng bit** với `proj/model.py` gốc (`max|Δ| = 0,0`).

### Ba điều đã sửa/phát hiện khi viết

* **31 lớp BatchNorm, không phải 62** — con số 62 là của model build 2, tôi đã chép nhầm sang.
* **Pruning và quantization: bài báo nêu, bản dựng KHÔNG làm.** Bằng chứng: `state_dict` có
  161 tensor fp32 + 31 int64, không có int8/fp16 nào. fp16 của AMP là precision *tính toán*,
  không phải quantization trọng số — hai thứ hay bị lẫn.
* **`nn.Softmax` không tồn tại trong module** (đếm được 0). Softmax nằm trong
  `CrossEntropyLoss` lúc train và gọi tường minh lúc inference. Thêm `nn.Softmax` vào `forward`
  sẽ khiến loss áp softmax hai lần.

### Số liệu đã xác minh cho tài liệu

| | |
|---|---|
| optimizer | `AdamW(lr=1e-3, weight_decay=1e-4)`, betas (0,9 / 0,999), eps 1e-8 |
| lịch LR | `LambdaLR`: warmup 1 round rồi cosine, khớp `_last_lr` 0,00097453 ở round 6 |
| module | 31 BatchNorm1d · 2 Dropout(0,1) · 32 ReLU · 1 LayerNorm · 0 Softmax |
| tham số | 395.024 learnable = **1,507 MiB** fp32; buffer 3.295; ckpt 4,70 MiB |
| `argmax(y_prob)` vs `y_pred` | lệch 8/10.761.343 dòng, cả 8 là hoà tuyệt đối ở fp16 |

---

## 2. Các quyết định đã chốt — đừng hỏi lại

| # | Quyết định | Ghi chú |
|---|---|---|
| 1 | **Centralized**, KHÔNG federated learning | tôi đã xác nhận 2 lần |
| 2 | `rounds=50`, `epochs_per_round=1`, batch **2048** global | 1 round = 1 epoch trên toàn bộ 43 M dòng + eval full test + checkpoint |
| 3 | Dùng **toàn bộ 43,045,415 dòng train** mỗi round, không subsample | |
| 4 | Các tham số còn lại giữ nguyên Table 1: Adam, LR 1e-3, cross-entropy, ReLU/softmax | |
| 5 | **Chỉ 10 metric** trong `.claude/skills/.../references/metrics.md` | accuracy + P/R/F1 × {macro, micro, weighted}. KHÔNG dùng 10 metric nhị phân của bài báo |
| 6 | **Không ablation gì cả**, giữ nguyên dataset, một kịch bản có CMSO feature selection | |
| 7 | Tôi đã **xác nhận và chấp nhận** cảnh báo rò rỉ Sybil (mục 6 bên dưới) | vẫn phải ghi cảnh báo kèm mọi con số |
| 8 | Ngân sách ~25 h / ~3 session Kaggle được chấp nhận | ước tính có thể quá thủ cựu, đo thật ở round 1 |
| 9 | Test cục bộ dùng conda env **`nckh`** | `source ~/miniforge3/etc/profile.d/conda.sh && conda activate nckh` |
| 10 | Sau khi train xong, **tải cả notebook đã chạy kèm output** về thư mục làm việc | đã ghi vào skill |

## 3. Kết quả BUILD 1 — đã chạy xong

**Kernel** `odixe0502/edl-cmso-veremi` v3 · **run** `edl_cmso_r50_b4096` · status `COMPLETE`.
50/50 round trong **9,74 h**, 701,5 s/round (lệch dưới 2 s suốt cả run), 61.362 mẫu/s,
peak 4,57 GB / 14,6 GB. Quota dùng 10,13 h / 30 h.

| | round 49 (cuối, bàn giao) | round 2 (đỉnh, **chọn bằng test**) |
|---|---:|---:|
| accuracy | 0.795341 | 0.842970 |
| **f1_macro** | **0.793345** | **0.816758** |
| f1_weighted | 0.788977 | 0.832646 |
| train_loss | 0.0553 | 0.1495 |

**Phát hiện chính:** overfitting dưới split theo thời gian. Train loss 0,2610 → 0,0553 (−79%)
trong khi mọi metric test đạt đỉnh ở **round 2** rồi trôi xuống. LR cosine về 0 **không** kéo lại
được. Khả năng tổng quát hoá tốt nhất đạt sau ~3 epoch.

**Ba con số phải đọc kèm cảnh báo:**
- `timeDelayAttack` F1 **0,2271** — 72,7% bị đoán thành `benign`. Lớp khó nhất đúng như dự báo;
  nguyên nhân là bộ phân loại **theo từng dòng** không thấy quan hệ thời gian.
- `trafficCongestionSybil` F1 0,7696 với **precision 0,9856** — dạng dấu hiệu của rò rỉ đặc trưng.
  CMSO giữ **13/16** nhóm `session`, **gồm cả `f_first_in_session`**. Run này không tách được.
- `dosAttack` F1 **0,9927** trên 1,57 M dòng — kết quả sạch và đáng tin nhất.

### Ba việc tiếp theo, xếp theo giá trị khoa học

1. **Run 50 round với nhóm `session` bị zero** (~10 h GPU) — phép đo duy nhất tách được đóng góp
   thật của model khỏi rò rỉ Sybil.
2. **Run không CMSO** (đủ 66 đặc trưng, ~10 h) — tái lập tuyên bố Table 3 của bài báo.
3. Mô hình theo chuỗi cho `timeDelayAttack`.

### Cách push (đã hoạt động, dùng lại y nguyên)

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate nckh
kaggle kernels push -p papers/build1-cmso-before-extractor/notebook-cmso-first/
```

CLI đã OAuth (`odixe0502`). `kernel-metadata.json` nằm cạnh notebook với
**`machine_shape: "NvidiaTeslaT4"` → 2× T4 sm_75** (giá trị này đã đo, mọi cách viết khác bị
coerce ngầm về 1× P100 — xem mục dưới).

### Bài học vận hành đã ghi vào skill

- `machine_shape: "NvidiaTeslaT4"` là giá trị duy nhất cho T4 ×2. `get_notebook_info` **không**
  xác nhận được (luôn trả `"Gpu"`); chỉ output run in `device_count` mới là bằng chứng.
- Dataset mount ở `/kaggle/input/datasets/<owner>/<slug>/<prefix>` — sâu 4 tầng. Dùng glob đệ quy
  và **in cây `/kaggle/input` trước khi assert**.
- Kéo file JSON sidecar về máy phân tích trước — bắt được 2 lỗi (một lỗi **im lặng**) mà không
  tốn run nào.
- **Notebook đã thực thi KHÔNG nằm trong `kaggle kernels output`.** Lấy qua MCP
  `download_notebook_output` với `filePath: "__notebook__.ipynb"` → trả signed URL → curl.
- `kernels output` có thể dừng sớm **không báo lỗi**. Đếm file, chạy lại lệnh đó (nó bỏ qua file
  đã có).
- Tối đa **2 batch GPU session** chạy song song.

## 4. Cấu trúc thư mục — đầy đủ

```
edl_cmso/
├── CONTEXT.md                       file này — bàn giao phiên làm việc
├── CLAUDE.md / AGENTS.md            hướng dẫn hành vi (giống nhau)
├── dataset.md                       mô tả VeReMi NextGen 16 lớp + mục "Reporting results"
├── s41598-025-94445-9.md            toàn văn bài báo gốc (Khan et al. 2025)
├── .mcp.json                        đăng ký Kaggle MCP server
│
├── papers/build1-cmso-before-extractor/   ══ BUILD 1 — XONG 50/50, CMSO TRƯỚC EXTRACTOR ══
│   ├── paper.md                     phương pháp gốc, Eq. 19–50 LaTeX, Table 1/3/4,
│   │                                  mục "Gaps in the paper" (12 chỗ bài báo bỏ trống)
│   ├── rebuild.md                   24 deviation, caveat dataset, Run history
│   ├── report.md                    báo cáo riêng của build 1
│   ├── make_figures.py / make_report.py    sinh lại figures/ và report.md từ artifact
│   ├── figures/                     5 PNG + summary.json
│   ├── notebook-cmso-first/         edl_cmso_veremi.ipynb + bản .executed
│   └── runs/edl_cmso_r50_b4096/     metrics 50 round, confusion, reports, logs, meta/mask
│                                      ⚠ checkpoints/ và preds/ ĐÃ XOÁ (mục 1q)
│
├── papers/build2-paper-order/       ══ BUILD 2 — ĐÚNG THỨ TỰ BÀI BÁO §4.8→§4.9→§4.10 ══
│   ├── paper.md  rebuild.md         31 deviation, Run history ghi cả hai lần phân kỳ
│   ├── smoke.py                     ★ 40 check LOCAL, 0 GPU — CHẠY TRƯỚC MỌI PUSH
│   │                                  chọn notebook bằng $EDL_NB
│   │                                  MẶC ĐỊNH: notebook-final-wd5e2/ (bản đang chạy)
│   ├── bench/                       ★ 18 script đo + realdata.py, GPU LOCAL, 0 quota
│   │   ├── bf16repro.py             bác bỏ "bf16 gỡ mất tripwire fp16" — 4 nhánh, 900 step
│   │   ├── onset2.py                9.000 step bf16 trên CUDA, 0 gradient hỏng
│   │   ├── dagsonly.py              đo 3 layout DAGSNet trần — sinh ra build 4
│   │   ├── scale.py                 độ lệch scale Eq. (28) — chẩn đoán SAI, đã bác bỏ
│   │   └── bench.py split.py verify.py prec.py guard.py extremes.py …
│   ├── notebook-run1-baseline/      run 1, wd 1e-4 — kernel edl-cmso-veremi-v2
│   ├── notebook-run3-compiled/      run 3 (c50), + torch.compile — kernel …-c50
│   ├── notebook-fp32-probe/         ⚫ đã huỷ tay — fp32 toàn model sai hướng
│   ├── notebook-tpu-probe/          4 probe TPU (probe_tpu{,2,3} + probe_spmd)
│   ├── notebook-final-wd5e2/        ★ XONG 50/50 — wd 5e-2 + W&B + abort trong round
│   │                                  kernel minhtran0601/edl-cmso-v2-final
│   ├── run1-wd1e4-diverged-r25/edl_cmso_v2_r50_b4096/
│   │   ├── channel_mask.json        ★ 133/262 kênh — MỌI run build 2 ĐỀU kế thừa
│   │   ├── extractor_init.pt        ★ phép chiếu §4.8 đóng băng — kế thừa cùng mask
│   │   ├── metrics/history.csv      45 round, CHỈ round 0–24 hợp lệ
│   │   └── confusion/ reports/ logs/    ⚠ checkpoints/ và preds/ ĐÃ XOÁ
│   ├── run3-c50-compiled-diverged-r22/edl_cmso_v2_c50/
│   │   └── metrics 22 round (chỉ 0–13 hợp lệ) + confusion/ reports/ logs/ + mask + init
│   └── run4-wd5e2-complete-50round/edl_cmso_v2_final/   ★ 50/50 ROUND, KHÔNG PHÂN KỲ
│       ├── checkpoints/  51 file — run pipeline đầy đủ duy nhất còn resume được
│       ├── metrics/ confusion/ reports/   50 round đầy đủ
│       ├── preds/        51 file argmax (prob_*.npz đã xoá — tái tạo được)
│       └── logs/heartbeat.jsonl   2.100 dòng: Wqkv_norm, logit max, entropy, skip%
│
├── papers/build4-dagsnet-only/      ══ BUILD 4 — XONG 50/50, KẾT QUẢ CAO NHẤT ══
│   ├── ARCHITECTURE.md              ★★ ĐẶC TẢ MÔ HÌNH CLASSIFIER ĐƯỢC CHỌN
│   │                                  hợp đồng vào/ra, 66 cột + scaler, bảng 69 lớp,
│   │                                  trọng số trước/sau huấn luyện, mã tự chứa để
│   │                                  dùng lại ở project khác (đã chạy kiểm, Δ=0)
│   ├── model/scaler.json            mean + std_used 66 cột, fit trên 43.045.415 dòng train
│   ├── smoke.py                     ★ 56 check — chọn notebook bằng $EDL_V4_NB_DIR
│   │                                  gồm test DDP 2 process bắt deadlock abort
│   ├── notebook-dagsnet-r0-45/      session 1 — 395.024 tham số
│   ├── notebook-dagsnet-resume-r46-49/   session 2 — REQUIRE_RESUME + glob đệ quy
│   └── runs/edl_cmso_v4_dagsnet/    ★ MỘT run 50 round, ghép từ hai session
│       ├── checkpoints/ metrics/ preds/ confusion/ reports/   round 0..49 liên tục
│       ├── meta.json  config.json   (config là của session 2)
│       └── logs/sessions.json + logs/sessions/{1,2}/   provenance từng session
│
├── figures/                         ★ 5 hình dùng trong report.md
├── make_figures.py                  sinh lại figures/ từ history.csv + confusion/*.npy
├── make_tables.py                   sinh mục 2.8; --apply ghi thẳng vào report.md
├── report.md                        ★ BÁO CÁO TỔNG HỢP — 1.102 dòng, 5 run × bảng từng round
│
└── .agents/skills/kaggle-training-notebook/   bản Codex chuẩn, đã validate
    ├── SKILL.md                     workflow GPU/TPU + tripwire + stop-session
    └── references/
        ├── paper-dossier.md         + mục "Equations that render" (lỗi KaTeX multiple \tag)
        ├── tpu-pytorch.md           ★ MỚI, 169 dòng — probe go/no-go, bẫy machine_shape,
        │                              bẫy InitializeComputationClient, bf16, cách benchmark
        ├── kaggle-mcp.md  stop-session.md  dataset-audit.md  metrics.md
        ├── notebook-template.md  perf-2xT4.md  pull-outputs.md  reporting.md
        └── veremi-dataset-layout.md          (chỉ có ở .agents/)

    ├── references/multi-account.md   ★ MỚI — xoay tua 2 tài khoản Kaggle (mục 4b)
    └── scripts/kaggle_account.py     ★ MỚI — list / save / use, đổi cả CLI lẫn MCP

└── skills/wandb-training-monitor/    theo dõi W&B read-only, progress/health/ETA
    ├── SKILL.md                      + mục "Deciding whether a Kaggle run is actually alive"
    └── scripts/inspect_run.py        dùng ~/.netrc; không in/commit key
        (có ở CẢ .agents/skills/ lẫn .claude/skills/ — Claude Code chỉ nạp bản .claude)
```

**Skill W&B mới (2026-09-03):** `references/wandb.md` trong `kaggle-training-notebook`
(cả hai bản) — credential trên Kaggle, fail-fast, hợp đồng metric key, bẫy `scan_history`,
quy tắc liveness ba nguồn, và ngân sách `max_hours` phải suy ra từ quota còn lại.

### W&B và quyền riêng tư (2026-09-03)

- `wandb 0.29.0` đã cài trong conda `nckh`; `wandb.login` local đã xác thực tài khoản
  `minhtriet`, entity mặc định `21522798-uit`, và lưu credential trong `~/.netrc`.
- API đã được gọi lại thành công khi loại `WANDB_API_KEY` khỏi environment; `.env` vì vậy
  không còn cần cho các phiên sau và đã được xóa theo yêu cầu. Không đưa key vào command line,
  notebook, metadata, log hoặc artifact. Xóa key khỏi notebook local sau push **không bảo mật
  được** vì bản source chứa key vẫn còn trong version history trên Kaggle.
- Kaggle CLI/API không mang secret attachment của UI vào batch version. Ngày 2026-09-03, người
  dùng đã **chấp thuận rõ** phương án nhúng W&B key trực tiếp vào source notebook private để mọi
  lần push tự chạy, chấp nhận key tồn tại trong local source và Kaggle private version history.
  Helper phải đọc từ `~/.netrc`, không in key hay đưa key vào command line; metadata phải luôn là
  `"is_private": true`. Helper chuẩn nằm tại
  `.agents/skills/kaggle-training-notebook/scripts/embed_wandb_key.py`. Nếu notebook đổi public
  hoặc thêm collaborator, dừng và rotate key trước.
- Mọi `kernel-metadata.json` hiện có đã audit đều đặt `"is_private": true`; đây là invariant
  bắt buộc phải kiểm tra lại ngay trước mọi lần push sau này.
- Notebook huấn luyện mới sẽ ghi `wandb_run.json` (entity/project/run ID/URL, không có secret),
  heartbeat theo bước, diagnostic ổn định số và đủ 10 metric sau mỗi round. Attempts 1–2 đều
  fail-fast đúng thiết kế ở cell 3; attempt 2 chứng minh việc bật secret trước một lần push CLI
  mới không gắn nó vào batch version. Attempt 3 đã tạo thành công W&B run
  `21522798-uit/edl-cmso/edl-cmso-v3-tpu-fast50`; chưa có heartbeat train tại lần kiểm tra đầu
  vì run vừa khởi tạo. Bốn heartbeat đầu (step 250/500/750/1.000) sau đó cho thấy loss
  0,552→1,209→1,278→1,303; grad norm 1,01→1,84×10^5→8,59×10^11→2,84×10^12;
  |logit|max 25,2→6.195→9.425→9.787. Skip vẫn 0% vì gradient còn hữu hạn và được clip,
  nên tripwire hiện tại không tự dừng. **Capture cuối cùng có 23 heartbeat**, step 250 → 5.750, cách
  đều 76–78 s, không khoảng trống → **0,3101 s/step**, một lần thực thi liên tục 02:55:09 →
  03:23:34 UTC. Ghi chép cũ "dừng ở step 2.250" là một lần đọc giữa run, đã sửa. Heartbeat
  cuối: loss 1,6834, grad norm 3,92×10¹³, |logit|max 15.262, ‖W_qkv‖ 29,12, skip 0%.
  Không có completed round/checkpoint/metric; executed notebook v5 trả 404. ⚠ Artifact W&B
  của run này **đã bị xoá cùng build 3** (mục 1q); các số ngay trên là toàn bộ những gì còn lại.
- **Đọc W&B của run đang chạy phải ghi rõ là ảnh chụp**, không phải điểm kết thúc; và
  `kernels status` không nói gì về việc session còn sống — xem mục 1m.

**Ba file quan trọng nhất khi tiếp tục:** `smoke.py` (chạy trước mọi push),
`run1-wd1e4-diverged-r25/edl_cmso_v2_r50_b4096/channel_mask.json` + `extractor_init.pt` (mọi run sau đều
kế thừa, đừng sinh lại).

### Hai tài khoản Kaggle — xoay tua quota (2026-09-03)

| | |
|---|---|
| account 1 | `odixe0502` — chủ dataset và mọi run từ trước tới nay |
| account 2 | `minhtran0601` — thêm ngày 2026-09-03 để tăng quota |

**Dữ liệu dùng chung được:** dataset `veremi-nextgen2026-centralized` vốn **public**.

⚠ **Chia sẻ notebook là THEO TỪNG NOTEBOOK, không phải cả tài khoản.** Đo thực tế 2026-09-04
khi active là `minhtran0601`:

| notebook | đọc được? |
|---|---|
| `odixe0502/edl-cmso-veremi-v2` | ✅ (đã chia sẻ với group) |
| `odixe0502/edl-cmso-veremi-c50` | ❌ `Permission 'kernels.get' was denied` |
| `odixe0502/edl-cmso-v4-dagsnet` | ❌ như trên |

Notebook **mới tạo không tự vào group**. Hệ quả: mọi run cần `kernel_sources` trỏ sang notebook
của account kia đều phải **chia sẻ notebook đó trước**, hoặc đi đường vòng — `last.pt` của
build 4 chỉ **4,7 MB**, upload thành dataset private nhỏ dưới account 2 là xong, không phải
đụng tới account 1.

**Và `id` trong `kernel-metadata.json` phải đổi owner** trước khi push bằng account khác:
`odixe0502/…` → `minhtran0601/…`. **Nhưng `channel_mask.json` + `extractor_init.pt` phải là ĐÚNG file đó** — nếu
hai account dùng hai bản khác nhau thì các run không còn so sánh được với nhau.

**Hai credential độc lập, phải đổi cùng nhau:**

| kênh | dùng cho | credential |
|---|---|---|
| CLI | `push`, `status`, `output` | `~/.kaggle/credentials.json` — OAuth, **một** account |
| MCP | quota, status, output | bearer `KGAT_…` trong `.mcp.json` của project |

`get_accelerator_quota` báo quota của account mà **MCP** xác thực, nên để lệch là đọc nhầm
quota của account khác.

**Đã xác minh 2026-09-03:** vòng `odixe0502 → minhtran0601 → odixe0502`, mỗi bước kiểm bằng
một lệnh `kaggle kernels status` thật — **refresh token đã lưu sống sót qua việc đổi qua đổi
lại**, nên mỗi account chỉ phải login trình duyệt đúng một lần. Cũng đã xác minh
`minhtran0601` **đọc được** `odixe0502/edl-cmso-veremi-v2` (nguồn `channel_mask.json` +
`extractor_init.pt`) qua group, và dataset truy cập được.

**Đã đo và bác bỏ:** `KGAT_` token **không** dùng được cho CLI. Thử cả `key = "KGAT_a83e…"`
lẫn `key = "a83e…"` (32 hex sau tiền tố, trông y hệt legacy API key) trong `KAGGLE_CONFIG_DIR`
cách ly — CLI 2.2.4 từ chối cả hai. **CLI chỉ có OAuth.**

```bash
S=.claude/skills/kaggle-training-notebook/scripts/kaggle_account.py
python3 $S list              # ai đang active, CLI và MCP có khớp không
python3 $S use minhtran0601  # đổi cả credentials.json lẫn bearer trong .mcp.json
# -> RESTART session, vì MCP server giữ header từ lúc khởi động
```

⚠ **Bẫy thứ tự:** `kaggle auth login --force` **ghi đè** `credentials.json`, mất refresh token
của account đang đăng nhập. **Luôn `save <account-hiện-tại>` trước.** `save` từ chối lưu sai
nhãn nếu tên không khớp login đang active.

Credential nằm ở `~/.kaggle/accounts/` (700, file 600), **ngoài project**. Không in ra, không
truyền qua command-line argument, không commit.

⚠ **`.mcp.json` trong project chứa bearer token sống.** Hiện chưa phải git repo nên chưa rò rỉ;
**kiểm tra lại ngay trước lần `git init` đầu tiên** — thêm vào `.gitignore` hoặc chuyển cấu
hình MCP lên user level.

Chi tiết đầy đủ, gồm cách chạy luồng OAuth không cần trình duyệt trên máy này:
[`references/multi-account.md`](.claude/skills/kaggle-training-notebook/references/multi-account.md).

### Lệnh hay dùng

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate nckh

EDL_NB=notebook-run3-compiled/edl_cmso_c50.ipynb \
  python papers/build2-paper-order/smoke.py         # 40 check, trước mọi push
kaggle kernels push   -p papers/build2-paper-order/notebook-run3-compiled/
kaggle kernels status odixe0502/edl-cmso-veremi-c50

# đo tốc độ ở LOCAL trước, đừng đốt quota Kaggle để thử giả thuyết (mục 1g)
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
python papers/build2-paper-order/bench/prec.py

# ghép output nhiều session thành MỘT run dir (mục 1p)
python .claude/skills/kaggle-training-notebook/scripts/merge_sessions.py \
    --out runs_merged pull_session1/ pull_session2/

# sinh lại toàn bộ hình của report.md từ artifact
python make_figures.py

# build 4 (ablation DAGSNet trần) — smoke 56 check rồi mới push
EDL_V4_NB_DIR=notebook-dagsnet-r0-45 python papers/build4-dagsnet-only/smoke.py
python .agents/skills/kaggle-training-notebook/scripts/embed_wandb_key.py \
  papers/build4-dagsnet-only/notebook-dagsnet-r0-45/edl_cmso_v4_dagsnet.ipynb \
  --metadata papers/build4-dagsnet-only/notebook-dagsnet-r0-45/kernel-metadata.json
kaggle kernels push -p papers/build4-dagsnet-only/notebook-dagsnet-r0-45/

# theo dõi run đang chạy (W&B là cửa sổ DUY NHẤT khi kernel chưa dừng)
python .claude/skills/wandb-training-monitor/scripts/inspect_run.py \
  --run 21522798-uit/edl-cmso/edl-cmso-v4-dagsnet

# đổi tài khoản Kaggle khi hết quota (mục 4b) — nhớ restart session sau đó
python .claude/skills/kaggle-training-notebook/scripts/kaggle_account.py use minhtran0601
```

**Notebook đã thực thi KHÔNG nằm trong `kernels output`.** Lấy qua MCP
`download_notebook_output` với `filePath: "__notebook__.ipynb"` → trả signed URL → `curl`.
Đây cũng là cách duy nhất đọc được output/traceback của một kernel đã chạy xong.

**Sau mỗi lần pull phải cập nhật `rebuild.md`:** một dòng Run history + bảng Results. Mọi
hyperparameter bị đổi để vượt qua crash cũng là deviation — ghi vào bảng, đừng đổi im lặng.

## 5. Notebook — những chỗ cần biết

Mục này viết cho notebook build 1. **Build 2 kế thừa toàn bộ** — nó được chuyển thể trực
tiếp từ file đó, giữ nguyên mọi phần hardening — cộng thêm các điểm riêng ở mục 1c.

`CFG` nằm ở **cell 3**, một chỗ duy nhất. Kiến trúc trong `%%writefile proj/model.py`; `%%writefile` để code vừa hiện trong notebook vừa import được bởi process mà `mp.spawn` tạo ra (hàm định nghĩa trong cell trần thì không).

Vài điểm thiết kế đáng nhớ:

- **Dữ liệu thường trú GPU, không có DataLoader trong vòng train.** Sau CMSO, đặc trưng được chọn ở fp16 chỉ ~2–3 GB; mỗi rank giữ shard riêng nằm luôn trên T4, batch chỉ là index vào VRAM. Với 4 vCPU nuôi 2 GPU thì đây là thứ quyết định, hơn cả AMP. Hai rank cắt shard **bằng nhau** — rank nào xong trước sẽ làm DDP all-reduce deadlock.
- **Cache fp16 đặt ở `/kaggle/temp`** (không bị commit), checkpoint ở `/kaggle/working` (được commit). `/kaggle/working` bị xoá khi mở session MỚI → resume xuyên session phụ thuộc việc attach output của run chết làm data source.
- **`y_prob` chỉ lưu ở round tốt nhất và round cuối**; `y_pred` lưu mọi round. 50 × 10.7 M × 16 float sẽ vượt giới hạn 20 GB output của Kaggle. Vẫn tính lại được mọi metric khác từ `runs/*/preds/` mà không cần train lại.
- **Eval quét tuần tự trên rank 0**, không `DistributedSampler` — sampler sẽ pad hoặc bỏ đuôi, làm sai điểm trên 10.7 M dòng.

### Ba chỗ hardening thêm ở phiên này (deviation 25–26) — đừng gỡ ra

1. **Ngân sách wall-clock `CFG.max_hours = 11.0`.** Kaggle giết batch session ở 12 h. Vòng train
   chỉ bắt đầu một round mới nếu `elapsed + 1.15 × round_vừa_rồi < deadline`; nếu không thì dừng
   sạch, ghi `logs/stopped_early.json`, để các cell cuối chạy và output commit bình thường —
   thay vì phó mặc cho việc bị kill cứng có commit hay không. Quyết định dừng đi qua
   `all_reduce(MAX)`; **nếu để mỗi rank tự xem đồng hồ của mình thì collective kế tiếp sẽ deadlock.**
2. **`init_process_group(timeout=60 phút)`.** Mặc định NCCL của PyTorch là **10 phút**, mà rank 1
   ngồi trong barrier suốt thời gian rank 0 eval 10,7 M dòng + ghi artifact. Vượt timeout là
   watchdog abort cả run, **không có traceback**.
3. **`torch.cuda.set_device(rank)` chuyển lên TRƯỚC `init_process_group`.** Đặt sau thì mọi rank
   dựng NCCL communicator trên `cuda:0`.

### Hai bug đã bắt được ở local — đừng vô tình làm hỏng lại

1. `PatchEmbed` phải được cấp độ dài **đầu ra của DWT** (`n + n%2`), không phải `n_features`. Subset CMSO lẻ sẽ làm vỡ reshape giữa run.
2. Subset < `2 × patch_len` làm trục token còn k=1 → GAT softmax trên 1 node là hằng số → `a_src`/`a_dst` không có gradient → **DDP treo im lặng, không traceback**. Đã chặn bằng `MIN_FEATURES = 12` trong `cmso.binarise` + assert ở cell launch.

Muốn test lại: extract các module `%%writefile` ra rồi chạy trong env `nckh` (đã có `torch 2.13.0+cpu`, `sklearn`, `pyarrow`, `pandas`).

## 6. ⚠ Bắt buộc ghi kèm mọi con số công bố

Lấy từ mục "Reporting results" của [`dataset.md`](dataset.md). Tôi đã xác nhận và chấp nhận điểm 4.

1. Split theo **thời gian mô phỏng**, không theo xe — 64 điểm cắt, mỗi (class × scenario) một điểm.
2. Lớp benign lấy từ luồng **không có tấn công**, khiến nhóm đặc trưng `rate` mạnh bất thường.
3. **3.338.358 dòng nhập nhằng** (bị thao túng nhưng không được gắn cờ) đã bị loại từ nguồn.
4. **Sybil:** 100,00% dòng `trafficCongestionSybil` nằm trong flow mà **mọi** dòng đều là first-in-session (lớp kế tiếp `feignedBraking` chỉ 0,20%; `dosAttack` 0,00%). Nhóm 16 đặc trưng `session` gần như là nhãn cho lớp này. **Build này KHÔNG chạy ablation**, nên F1 Sybil cao phần lớn đến từ nhóm `session` chứ không phải DAGSNet, và run này **không tách được** phần đóng góp đó. Phép đo cần thiết là một run 50 round thứ hai với nhóm `session` bị zero.
5. `timeDelayAttack` là lớp khó nhất.
6. Mất cân bằng **41:1** → `f1_macro` mới là con số đáng đọc, `accuracy` gần như vô nghĩa.
7. **88% flow trong test dài đúng 1 message**, hầu hết là Sybil. Giữ nguyên: DAGSNet phân loại theo từng dòng, không bao giờ ghép flow. Lọc flow ngắn ở đây là xoá gần hết một lớp, không phải làm sạch dữ liệu.
8. CMSO fit trên subsample **chỉ từ train**, holdout cũng từ train. Test không hề được dùng để chọn đặc trưng, tinh chỉnh hay early stopping.

**Không đặt kết quả cạnh số của bài báo trong cùng một bảng.** Bài báo là phân loại **nhị phân** trên CIC-IDS 2017 / CAN với quy ước metric khác (specificity, NPV, MCC, FPR, FNR — đều cần TN mà 16 lớp không định nghĩa được nếu chưa chọn cách trung bình). Cái kế thừa từ bài báo là **phương pháp**, không phải con số.
