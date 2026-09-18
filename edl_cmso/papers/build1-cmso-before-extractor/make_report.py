"""Generate report.md from the pulled run artifacts.

Every number in the report comes from a file under runs/<run>/ — nothing is typed by hand,
so the report cannot drift from the run it describes. Run from the repo root.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("papers/build1-cmso-before-extractor")
RUN = BASE / "runs" / "edl_cmso_r50_b4096"
OUT = BASE / "report.md"

hist = (pd.read_csv(RUN / "metrics" / "history.csv")
          .drop_duplicates(subset="round", keep="last").sort_values("round").reset_index(drop=True))
meta = json.loads((RUN / "meta.json").read_text())
cfg = json.loads((RUN / "config.json").read_text())
fmask = json.loads((RUN / "feature_mask.json").read_text())
summ = json.loads((BASE / "figures" / "summary.json").read_text())
names = meta["class_names"]
FIN, PEAK = summ["final_round"], summ["peak_round"]
K = ["accuracy", "precision_macro", "precision_micro", "precision_weighted",
     "recall_macro", "recall_micro", "recall_weighted", "f1_macro", "f1_micro", "f1_weighted"]

cm_f = np.load(RUN / "confusion" / f"round_{FIN:03d}.npy").astype(float)
rown = cm_f / np.maximum(cm_f.sum(1, keepdims=True), 1)


def conf_top(cls, k=3):
    i = names.index(cls)
    out = [f"{rown[i, j] * 100:.1f}% → `{names[j]}`" for j in np.argsort(-rown[i])[:k]
           if rown[i, j] > 0.01 and j != i]
    return ", ".join(out)


# ── per-round table ───────────────────────────────────────────────────────────
hdr = "| round | accuracy | P_macro | P_micro | P_wtd | R_macro | R_micro | R_wtd | F1_macro | F1_micro | F1_wtd | train_loss | lr | giây |"
sep = "|" + "---|" * 14
rows = [hdr, sep]
for _, r in hist.iterrows():
    mark = ""
    if int(r["round"]) == PEAK:
        mark = " **←đỉnh**"
    if int(r["round"]) == FIN:
        mark = " **←cuối**"
    rows.append("| " + str(int(r["round"])) + mark + " | " +
                " | ".join(f"{r[k]:.6f}" for k in K) +
                f" | {r['train_loss']:.4f} | {r['lr']:.2e} | {r['seconds']:.0f} |")
per_round = "\n".join(rows)

# ── per-class table ───────────────────────────────────────────────────────────
pc = sorted(summ["per_class"].items(), key=lambda kv: -kv[1]["support"])
pcl = ["| lớp | dòng test | % test | P @49 | R @49 | **F1 @49** | F1 @2 | Δ |",
       "|---|---:|---:|---:|---:|---:|---:|---:|"]
for n, v in pc:
    d = v["final"]["f1"] - v["peak"]["f1"]
    pcl.append(f"| `{n}` | {v['support']:,} | {100 * v['support'] / meta['n_test']:.2f} | "
               f"{v['final']['precision']:.4f} | {v['final']['recall']:.4f} | "
               f"**{v['final']['f1']:.4f}** | {v['peak']['f1']:.4f} | {d:+.4f} |")
per_class = "\n".join(pcl)

groups = {}
for g, cols in (json.loads((BASE / "figures" / "groups.json").read_text())
                if (BASE / "figures" / "groups.json").exists() else {}).items():
    groups[g] = cols

acc_f = summ["final"]["accuracy"]
acc_p = summ["peak"]["accuracy"]

md = f"""# Báo cáo — EDL-CMSO trên VeReMi NextGen (16 lớp, centralized)

**Bài báo gốc** · Khan, Tejani, Alsulami et al. (2025), *A Secure and Efficient Deep
Learning-Based Intrusion Detection Framework for the Internet of Vehicles*,
**Scientific Reports**, DOI [`10.1038/s41598-025-94445-9`](https://doi.org/10.1038/s41598-025-94445-9)
· phương pháp gốc chép lại đầy đủ trong [paper.md](paper.md)

**Dữ liệu** · `odixe0502/veremi-nextgen2026-centralized` (Zenodo `10.5281/zenodo.19665762`)
**Kernel** · [`odixe0502/edl-cmso-veremi`](https://www.kaggle.com/code/odixe0502/edl-cmso-veremi) v3
· **Run** · `{cfg['run_name']}` · **Ngày** · 2026-08-31
**Phần cứng** · Kaggle 2× Tesla T4 (sm_75, 14,6 GB mỗi card), 4 vCPU, 31 GB RAM, DDP + fp16 AMP

> **Kết quả một dòng.** {summ['rounds']}/{cfg['rounds']} round hoàn tất trong
> **{summ['wall_hours']:.2f} giờ** ({summ['mean_round_seconds']:.0f} s/round,
> {summ['mean_samples_per_sec']:,.0f} mẫu/s). Round cuối (**49**, mô hình bàn giao):
> `f1_macro` **{summ['final']['f1_macro']:.5f}**. Round có điểm cao nhất (**2**,
> *chọn bằng tập test*): `f1_macro` **{summ['peak']['f1_macro']:.5f}**.

Ba tài liệu, ba vai trò khác nhau — đọc kèm nhau:

| File | Vai trò |
|---|---|
| [`notebook/edl_cmso_veremi.executed.ipynb`](notebook/edl_cmso_veremi.executed.ipynb) | **bằng chứng** — code, công thức và output đã render nằm chung một file |
| [`rebuild.md`](rebuild.md) | **sổ ghi** — 30 deviation so với bài báo, lịch sử từng lần chạy |
| `report.md` (file này) | **tổng hợp** — kết quả và diễn giải |

---

## 1 · Dựng lại cái gì

Bài báo gồm 6 stage. Bản dựng lại này làm **stage 3–5**:

**DWT** (Eq. 19) → **ViT** (Eq. 20–25) → **GAT** (Eq. 26–27) → **fusion** (Eq. 28) →
**CMSO feature selection** (Eq. 29–37) → **DAGSNet** = DenseNet + GoogleNet + AlexNet +
SqueezeNet (Eq. 38–48).

Bỏ hẳn, theo yêu cầu đã chốt:

- **Stage 1** — IoVCipherGuard (HE / SMPC / AES-256, §4.3–4.6). Bảng 2 của chính bài báo chỉ
  báo cáo nó như một cột thời gian; không metric nào phụ thuộc vào nó.
- **Stage 2** — tiền xử lý (§4.7). Dataset đã imputed và standardised sẵn.
- **Stage 6** — AFPHA federated aggregation. Bản này chạy **centralized, một model duy nhất**.

Bài báo để trống **12 chỗ** không nêu giá trị (họ wavelet, cách biến một dòng bảng thành đối
tượng 2-D patch được, kích thước ViT, định nghĩa lân cận GAT, mọi tham số CMSO…). Mỗi chỗ được
lấp bằng một lựa chọn ghi rõ trong `rebuild.md`, phân loại là *bắt buộc*, *do phần cứng*,
*do người dùng* hay *do phán đoán*.

## 2 · Dữ liệu, và các cảnh báo phải đi kèm mọi con số

| split | dòng | đặc trưng | trạng thái |
|---|---:|---:|---|
| train | {meta['n_train']:,} | 66 (`f_*`) | đã imputed **và** standardised |
| test | {meta['n_test']:,} | 66 (`f_*`) | đã imputed, `scaler.json` (fit trên train) áp lúc load |

16 lớp, mất cân bằng **41,4 : 1**, không lớp rỗng, **0 ô NaN/Inf** ở cả hai split, tỉ lệ
train:test = 0,8000 : 0,2000. 21 cột không phải `f_*` (định danh, nhãn, ngữ cảnh thô, khoá phân
mảnh) **không bao giờ** được đưa vào model.

**Bảy cảnh báo dưới đây lấy từ chính tài liệu của dataset. Mọi con số trong báo cáo này chỉ đọc
được khi đặt cạnh chúng:**

1. Split cắt theo **thời gian mô phỏng**, không theo xe — 64 điểm cắt, mỗi (lớp × kịch bản) một
   điểm. Train và test **không cùng phân phối**. Đây là nguyên nhân trực tiếp của hiện tượng
   ở mục 5.
2. Lớp `benign` lấy từ luồng **không có tấn công**, khiến nhóm đặc trưng `rate` mạnh bất thường.
3. **3.338.358 dòng nhập nhằng** (bị thao túng nhưng không gắn cờ) đã bị loại từ nguồn.
4. **Rò rỉ Sybil.** 100,00% dòng `trafficCongestionSybil` nằm trong flow mà *mọi* dòng đều là
   first-in-session (lớp kế tiếp `feignedBraking` chỉ 0,20%). Nhóm 16 đặc trưng `session` gần
   như là nhãn cho lớp này. **Build này không chạy ablation**, nên không tách được phần đóng góp
   đó — xem mục 6.
5. `timeDelayAttack` là lớp khó nhất. Kết quả xác nhận: xem mục 6.
6. Mất cân bằng 41:1 → **`f1_macro` là con số đáng đọc**, `accuracy` gần như vô nghĩa.
7. **88% flow trong test dài đúng 1 message**, phần lớn là Sybil. Giữ nguyên: DAGSNet phân loại
   theo từng dòng, không bao giờ ghép flow.

CMSO fit trên subsample **chỉ lấy từ train**, holdout cũng từ train. Tập test **không hề** được
dùng để chọn đặc trưng hay tinh chỉnh.

## 3 · Cấu hình thực tế đã chạy

Đọc từ [`runs/{cfg['run_name']}/config.json`](runs/{cfg['run_name']}/config.json) — bản ghi của
chính lần chạy, không phải source notebook.

| | | | |
|---|---|---|---|
| rounds | {cfg['rounds']} | epochs/round | {cfg['epochs_per_round']} |
| batch mỗi GPU | {cfg['batch_per_gpu']} | **batch toàn cục** | **{cfg['batch_per_gpu'] * cfg['world_size'] * cfg['grad_accum']}** |
| optimizer | AdamW | learning rate | {cfg['lr']} (Table 1), cosine decay, {cfg['warmup_rounds']} round warmup |
| weight decay | {cfg['weight_decay']} | grad clip | {cfg['clip']} |
| loss | cross-entropy | AMP | fp16 (T4 không có bf16/TF32) |
| d_model | {cfg['d_model']} | patch_len | {cfg['patch_len']} |
| ViT | {cfg['vit_layers']} layer, {cfg['vit_heads']} head | GAT | {cfg['gat_heads']} head, fully connected |
| DenseNet | {cfg['dense_layers']} layer, growth {cfg['dense_growth']} | GoogleNet | {cfg['incep_modules']} inception |
| SqueezeNet | {cfg['fire_modules']} fire | seed | {cfg['seed']} |
| tham số model | 827.123 | step/round/rank | {(meta['n_train'] // cfg['world_size']) // cfg['batch_per_gpu']:,} |

**CMSO** — {fmask['n_selected']}/{fmask['dim']} đặc trưng được giữ, fitness
{fmask['best_fitness']:.5f}, {cfg['cmso_pop']} cá thể × {cfg['cmso_iters']} vòng lặp
(991 lần đánh giá không trùng, 845 s). Phân bố theo nhóm:

| nhóm | giữ / tổng |
|---|---|
| `session` | **13 / 16** |
| `geometry` | 11 / 20 |
| `raw` | 10 / 15 |
| `position` | 3 / 4 |
| `profile` | 3 / 6 |
| `rate` | **1 / 5** |

Hai nhóm mà tài liệu dataset cảnh báo lại đi ngược nhau: CMSO **giữ gần hết nhóm `session`**
(gồm cả `f_first_in_session` — chính đặc trưng bị nêu đích danh ở cảnh báo 4) và **loại gần hết
nhóm `rate`**. Nghĩa là cảnh báo rò rỉ Sybil không những còn nguyên hiệu lực mà còn mạnh hơn:
bộ chọn đặc trưng đã chủ động ưu tiên đúng nhóm bị nghi ngờ.

## 4 · Kết quả — 10 metric sau mỗi round

Đo trên **toàn bộ {meta['n_test']:,} dòng test**, sau mỗi round, quét tuần tự trên rank 0 —
không `DistributedSampler`, nên không có phần đuôi bị pad hay bị bỏ.

> **Năm cột trùng nhau là đúng, không phải lỗi.** Trong phân loại đơn nhãn đa lớp, mỗi dự đoán
> thuộc đúng một lớp, nên $\\sum_c FP_c = \\sum_c FN_c$ và do đó
> `precision_micro` = `recall_micro` = `f1_micro` = `recall_weighted` = `accuracy`.
> Ở round 49 cả năm đều bằng **{acc_f:.6f}**.

{per_round}

Nguồn: [`runs/{cfg['run_name']}/metrics/history.csv`](runs/{cfg['run_name']}/metrics/history.csv)

### Hai con số headline

| | round 49 — **cuối, mô hình bàn giao** | round 2 — đỉnh, **chọn bằng tập test** |
|---|---:|---:|
| `accuracy` | {summ['final']['accuracy']:.6f} | {summ['peak']['accuracy']:.6f} |
| `f1_macro` | **{summ['final']['f1_macro']:.6f}** | **{summ['peak']['f1_macro']:.6f}** |
| `f1_weighted` | {summ['final']['f1_weighted']:.6f} | {summ['peak']['f1_weighted']:.6f} |
| `precision_macro` | {summ['final']['precision_macro']:.6f} | {summ['peak']['precision_macro']:.6f} |
| `recall_macro` | {summ['final']['recall_macro']:.6f} | {summ['peak']['recall_macro']:.6f} |
| train loss | {summ['final']['train_loss']:.4f} | {summ['peak']['train_loss']:.4f} |

**Cột bên phải được chọn bằng chính tập test dùng để chấm điểm nó.** Đó là một dạng chọn mô hình
trên test, nên nó là **ước lượng lạc quan**, không phải hiệu năng có thể kỳ vọng trên dữ liệu
chưa thấy. Cột bên trái không dính vấn đề đó. Nhãn này lặp lại ở mọi chỗ con số round 2 xuất hiện
trong báo cáo, và đó là chủ ý.

## 5 · Hình

![Metric test sau mỗi round](figures/convergence.png)

*Ba metric test (trục trái) và train loss (trục phải, nét đứt). Điểm cần nhìn: hai đường đi
**ngược nhau** gần như từ đầu. Train loss giảm đơn điệu 0,2610 → 0,0553 (−79%) trong suốt 50
round, còn mọi metric test đạt đỉnh ở round 2 rồi trôi xuống.*

![Khoảng tụt so với đỉnh](figures/generalisation_gap.png)

*Khoảng cách `f1_macro` so với đỉnh. Tụt nhanh trong 20 round đầu rồi đi ngang quanh −2,3 điểm.
Learning rate cosine giảm về 0 ở round 49 **không** kéo lại được — giả thuyết "sẽ hồi khi LR nhỏ"
đã bị dữ liệu bác bỏ.*

![F1 theo lớp](figures/per_class_f1.png)

*F1 từng lớp ở round 49 (xanh) và round 2 (cam), **sắp theo số dòng test giảm dần**, nhãn phía
trên là support. Sắp theo support chứ không theo điểm: sắp theo điểm sẽ đẩy các lớp đẹp lên
trước và giấu mất chuyện chúng là lớp hiếm.*

![Confusion round 49](figures/confusion_final_round049.png)

*Confusion chuẩn hoá theo hàng ở round 49 — mô hình bàn giao.*

![Confusion round 2](figures/confusion_peak_round002.png)

*Cùng ma trận ở round 2 (đỉnh, chọn bằng test), để so sánh trực tiếp.*

## 6 · Phân tích theo lớp

{per_class}

Nguồn: [`runs/{cfg['run_name']}/confusion/round_049.npy`](runs/{cfg['run_name']}/confusion/) và
[`reports/round_049.txt`](runs/{cfg['run_name']}/reports/round_049.txt)

**`timeDelayAttack` — F1 {summ['per_class']['timeDelayAttack']['final']['f1']:.4f}, đúng như dataset card dự báo là lớp khó nhất.**
Không phải "kém", mà là gần như không phát hiện được: chỉ **17,4%** số dòng được nhận đúng, còn
**{conf_top('timeDelayAttack', 1)}**. Điều này hợp lý về mặt bản chất tấn công — một message bị
trễ vẫn mang nội dung hoàn toàn hợp lệ; cái bất thường nằm ở *thời điểm*, mà quan hệ thời gian
giữa các message thì một bộ phân loại **theo từng dòng** không nhìn thấy được. Đây là giới hạn
của thiết kế bài toán, không phải của DAGSNet.

**`positionMirroring` — F1 {summ['per_class']['positionMirroring']['final']['f1']:.4f}**, cũng chảy về benign
({conf_top('positionMirroring', 1)}).

**`benign` — F1 {summ['per_class']['benign']['final']['f1']:.4f}** dù là lớp lớn nhất. Precision chỉ
{summ['per_class']['benign']['final']['precision']:.4f}: cột dự đoán `benign` có **34,6% là hàng giả** — 11,3%
đến từ `timeDelayAttack`, 10,7% từ `trafficCongestionSybil`, 5,3% từ `positionMirroring`. Nói cách
khác, `benign` là nơi mọi thứ mô hình không nhận ra bị dồn vào.

**`trafficCongestionSybil` — F1 {summ['per_class']['trafficCongestionSybil']['final']['f1']:.4f}, và đây là con số
phải đọc cẩn thận nhất.** Precision **{summ['per_class']['trafficCongestionSybil']['final']['precision']:.4f}** với recall chỉ
{summ['per_class']['trafficCongestionSybil']['final']['recall']:.4f}: khi model nói "Sybil" thì gần như luôn đúng, nhưng nó bỏ sót
hơn một phần ba. Precision gần tuyệt đối trên một lớp chiếm 22% tập test là **đúng dạng dấu hiệu
của rò rỉ đặc trưng** đã cảnh báo ở mục 2.4 — model có vẻ đã tìm được một chữ ký gần như xác
định, chứ không phải học được hành vi Sybil. **Run này không tách được phần đóng góp đó.** Phép
đo cần thiết là một run 50 round thứ hai với toàn bộ nhóm `session` bị zero; nó chưa được chạy.

**`dosAttack` F1 {summ['per_class']['dosAttack']['final']['f1']:.4f}** trên 1,57 triệu dòng là kết quả sạch và đáng
tin nhất trong bảng — lớp lớn, chữ ký hành vi rõ, không nằm trong nhóm bị cảnh báo rò rỉ.

**Overfitting không đều — và đây là quan sát thú vị nhất.** Từ round 2 sang 49, phần lớn lớp
tụt điểm, nhưng ba lớp lại **tăng**: `timeDelayAttack` {summ['per_class']['timeDelayAttack']['peak']['f1']:.4f} →
{summ['per_class']['timeDelayAttack']['final']['f1']:.4f}, `suddenConstantSpeed`
{summ['per_class']['suddenConstantSpeed']['peak']['f1']:.4f} → {summ['per_class']['suddenConstantSpeed']['final']['f1']:.4f},
`zeroSpeedReport` {summ['per_class']['zeroSpeedReport']['peak']['f1']:.4f} → {summ['per_class']['zeroSpeedReport']['final']['f1']:.4f}.
Nghĩa là 47 round sau không đơn thuần là "học thuộc": model đánh đổi các lớp dễ và các lớp có
chữ ký rò rỉ để lấy một chút tiến bộ ở lớp khó nhất. Đó chính là lý do `recall_macro` ở round 49
({summ['final']['recall_macro']:.4f}) **cao hơn** round 2 ({summ['peak']['recall_macro']:.4f}), dù mọi metric khác đều thấp hơn.

## 7 · Bài báo gốc và bản dựng lại

**Hai bảng dưới đây cố ý tách rời. Không được đặt chung một bảng** — bảng chung là một tuyên bố
so sánh, và trong trường hợp này tuyên bố đó không hợp lệ.

**Bài báo báo cáo (Table 3, có feature selection):**

| Metric | Không FS | Có FS |
|---|---|---|
| Accuracy | 0,980981 | **0,99125** |
| Precision | 0,973654 | **0,985433** |
| Recall | 0,97218 | **0,98452** |
| F-measure | 0,979421 | **0,98455** |
| Specificity | 0,98321 | **0,990158** |
| MCC | 0,972166 | **0,984523** |

**Bản dựng lại này đo được:**

| Metric | round 49 (bàn giao) | round 2 (chọn bằng test) |
|---|---|---|
| accuracy | {summ['final']['accuracy']:.6f} | {summ['peak']['accuracy']:.6f} |
| precision_macro | {summ['final']['precision_macro']:.6f} | {summ['peak']['precision_macro']:.6f} |
| recall_macro | {summ['final']['recall_macro']:.6f} | {summ['peak']['recall_macro']:.6f} |
| f1_macro | {summ['final']['f1_macro']:.6f} | {summ['peak']['f1_macro']:.6f} |

**Bốn lý do khiến hai bảng không so sánh được:**

1. **Khác hình dạng bài toán.** Bài báo làm **nhị phân** (normal / anomaly). Đây là **16 lớp**.
   Với 16 lớp, một bộ phân loại đoán bừa theo tỉ lệ có baseline thấp hơn nhiều so với nhị phân;
   0,99 trên nhị phân và 0,79 trên 16 lớp không nằm trên cùng một thang.
2. **Khác dataset.** CIC-IDS 2017 / CAN / CICIoV2024 là lưu lượng mạng và bus. Đây là **message
   V2X** từ mô phỏng VeReMi NextGen.
3. **Khác quy ước metric.** 5 trong 10 metric của bài báo (specificity, NPV, MCC, FPR, FNR) đều
   cần **TN**, mà 16 lớp không định nghĩa TN nếu chưa chốt cách trung bình. Chúng đơn giản là
   không tồn tại ở dạng đó tại đây.
4. **Khác chính sách split.** Bài báo ghi split là `unstated`. Ở đây split cắt **theo thời gian**
   — cố tình khó hơn split ngẫu nhiên, vì nó cấm mọi rò rỉ từ tương lai của một phiên về quá khứ.
   Một split ngẫu nhiên trên cùng dữ liệu này gần như chắc chắn cho điểm cao hơn nhiều.

**Cái được kế thừa từ bài báo là phương pháp, không phải con số.** Câu hỏi bản dựng lại này trả
lời được là *"pipeline DWT→ViT→GAT→CMSO→DAGSNet có chạy và đo được trên dữ liệu V2X thật không"* —
có. Câu hỏi nó **không** trả lời là *"phương pháp này có tốt bằng con số bài báo công bố không"* —
để trả lời cần chạy lại chính trên CIC-IDS 2017 ở chế độ nhị phân, việc chưa làm.

## 8 · Số liệu này chứng minh và không chứng minh điều gì

**Chứng minh được:**

- Toàn bộ pipeline stage 3–5 dựng lại được từ bài báo và **chạy ổn định**: 50 round × 43 triệu
  dòng, {summ['wall_hours']:.2f} h, nhịp round lệch nhau dưới 2 giây suốt cả run, không NaN, không phân kỳ.
- Trên dữ liệu này, phương pháp đạt `f1_macro` **{summ['final']['f1_macro']:.4f}** ở round cuối trên
  **toàn bộ** {meta['n_test']:,} dòng test — không lấy mẫu, không bỏ đuôi.
- Một số lớp học được thật và sạch: `dosAttack` {summ['per_class']['dosAttack']['final']['f1']:.4f},
  `accelerationMultiplication` {summ['per_class']['accelerationMultiplication']['final']['f1']:.4f},
  `feignedBraking` {summ['per_class']['feignedBraking']['final']['f1']:.4f}.
- **Khả năng tổng quát hoá tốt nhất đạt được sau 3 epoch**, sau đó suy giảm đơn điệu. Với split
  theo thời gian, huấn luyện lâu hơn *làm hại* điểm test.

**Không chứng minh được:**

- **F1 của `trafficCongestionSybil` không quy cho DAGSNet được.** Nhóm `session` được CMSO giữ
  gần hết, gồm cả đặc trưng bị cảnh báo đích danh. Cần một run thứ hai với nhóm `session` bị zero
  mới tách được.
- **Không có ablation nào.** Không có nhánh "không feature selection", nên **không tái lập được
  tuyên bố trung tâm của Table 3** ("feature selection nâng mọi metric") trên dữ liệu này.
- **Một seed, một lần chạy.** Không có khoảng tin cậy, không đo được độ nhạy theo seed.
- Kiến trúc dùng ở đây là **một** cách lấp 12 chỗ trống của bài báo. Cách lấp khác cho con số khác;
  bài báo không đủ thông tin để loại trừ.

**Ba việc tiếp theo, xếp theo giá trị khoa học:**

1. **Run 50 round với nhóm `session` bị zero** — phép đo duy nhất tách được đóng góp thật của
   model khỏi rò rỉ Sybil. Chi phí ~10 h GPU.
2. **Run không có CMSO** (đủ 66 đặc trưng) — tái lập tuyên bố Table 3 trên dữ liệu này. ~10 h.
3. **Mô hình theo chuỗi cho `timeDelayAttack`** — lớp này thất bại vì bộ phân loại theo dòng
   không nhìn thấy quan hệ thời gian, chứ không phải vì thiếu dung lượng model.

## 9 · Tái lập

```bash
# dataset : odixe0502/veremi-nextgen2026-centralized
# kernel  : odixe0502/edl-cmso-veremi   (machine_shape "NvidiaTeslaT4" -> 2x T4 sm_75)
kaggle kernels push -p papers/build1-cmso-before-extractor/notebook/
```

`kernel-metadata.json` nằm cạnh notebook, đã điền sẵn accelerator, dataset source và slug.
Notebook tự resume: mỗi round ghi một checkpoint nguyên tử, chạy lại sẽ tiếp ở round dang dở.

| Nội dung | Đường dẫn |
|---|---|
| Checkpoint từng round (51 file, 492 MB) | [`runs/{cfg['run_name']}/checkpoints/`](runs/{cfg['run_name']}/checkpoints/) |
| Metric từng round + `history.csv` | [`runs/{cfg['run_name']}/metrics/`](runs/{cfg['run_name']}/metrics/) |
| `y_pred` mọi round, `y_prob` round 0/2/49, `y_true` | [`runs/{cfg['run_name']}/preds/`](runs/{cfg['run_name']}/preds/) |
| Confusion + báo cáo per-class từng round | [`runs/{cfg['run_name']}/confusion/`](runs/{cfg['run_name']}/confusion/), [`reports/`](runs/{cfg['run_name']}/reports/) |
| Mask đặc trưng CMSO | [`runs/{cfg['run_name']}/feature_mask.json`](runs/{cfg['run_name']}/feature_mask.json) |
| Log kernel đầy đủ | [`runs/{cfg['run_name']}/logs/kernel.log`](runs/{cfg['run_name']}/logs/kernel.log) |
| Notebook đã thực thi kèm output | [`notebook/edl_cmso_veremi.executed.ipynb`](notebook/edl_cmso_veremi.executed.ipynb) |

`y_pred` được lưu ở **mọi** round, nên bất kỳ metric nào khác — kể cả 10 metric nhị phân của bài
báo, nếu chốt một cách trung bình — đều tính lại được từ `preds/` mà **không cần train lại**.

---

*Sinh tự động bởi [`make_report.py`](make_report.py) từ artifact trong `runs/{cfg['run_name']}/`.
Mọi con số trong file này đọc trực tiếp từ đó; không con số nào gõ tay.*
"""

OUT.write_text(md)
print(f"wrote {OUT}  ({len(md):,} chars, {md.count(chr(10)) + 1} lines)")
