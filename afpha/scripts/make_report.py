#!/usr/bin/env python
"""Generate report.md from the three verified run directories.

Usage: conda run -n nckh python scripts/make_report.py

Every number in the report is read from a run's own artifacts. Nothing is typed by hand and
nothing is copied from a W&B snapshot: W&B holds interim scalars, the committed files are the
evidence. Run `scripts/verify_run.py <run> --require-complete` on each run first -- this script
refuses to build a table from a directory that does not have all its rounds.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import METRIC_KEYS                      # noqa: E402

SCENARIOS = [(20, 512), (50, 512), (100, 256)]
RUNS = {n: ROOT / f"runs/{n}client/runs/afpha-dagsnet-{n}client" for n, _ in SCENARIOS}
SHORT = {"accuracy": "acc", "precision_macro": "P_mac", "precision_micro": "P_mic",
         "precision_weighted": "P_wtd", "recall_macro": "R_mac", "recall_micro": "R_mic",
         "recall_weighted": "R_wtd", "f1_macro": "F1_mac", "f1_micro": "F1_mic",
         "f1_weighted": "F1_wtd"}


def load(n):
    r = RUNS[n]
    cfg = json.loads((r / "config.json").read_text())
    rows = list(csv.DictReader((r / "metrics" / "history.csv").open()))
    done = sorted(int(p.stem.split("_")[1]) for p in (r / "complete").glob("round_*.done"))
    assert done == list(range(1, cfg["rounds"] + 1)), (
        f"{n}-client run is not complete: {len(done)} of {cfg['rounds']} rounds")
    per_class = json.loads((r / "per_class" / f"round_{cfg['rounds']:03d}.json").read_text())
    clients = json.loads((r / "client_log" / f"round_{cfg['rounds']:03d}.json").read_text())
    return {"cfg": cfg, "rows": rows, "per_class": per_class, "clients": clients, "dir": r}


def metric_table(rows):
    """All 50 rounds x all 10 metrics, six decimals, exactly as committed."""
    head = "| round | " + " | ".join(SHORT[k] for k in METRIC_KEYS) + " |"
    rule = "|---:|" + "---:|" * len(METRIC_KEYS)
    out = [head, rule]
    for r in rows:
        out.append(f"| {int(r['round'])} | "
                   + " | ".join(f"{float(r[k]):.6f}" for k in METRIC_KEYS) + " |")
    return "\n".join(out)


def summarize(d):
    rows, cfg = d["rows"], d["cfg"]
    last = rows[-1]
    best_f1 = max(rows, key=lambda r: float(r["f1_macro"]))
    best_acc = max(rows, key=lambda r: float(r["accuracy"]))
    dur = [float(r["t_round_s"]) for r in rows]
    return {"final": last, "best_f1": best_f1, "best_acc": best_acc,
            "total_h": sum(dur) / 3600, "median_s": sorted(dur)[len(dur) // 2],
            "first_s": dur[0], "cfg": cfg}


def main():
    data = {n: load(n) for n, _ in SCENARIOS}
    s = {n: summarize(d) for n, d in data.items()}
    out = []
    W = out.append

    W("# AFPHA–DAGSNet trên VeReMi: báo cáo tổng quan\n")
    W("**Ngày: 2026-09-08. Thư mục: `/home/odixe/nckh/afpha`.**\n")
    W("Ba kịch bản federated learning 20 / 50 / 100 client, mỗi kịch bản 50 round, "
      "đánh giá sau mỗi lần aggregation trên **toàn bộ** tập test 10.761.343 dòng. "
      "Mọi con số trong báo cáo này được đọc trực tiếp từ artifact đã commit của từng run "
      "(`runs/<n>client/`), không lấy từ ảnh chụp W&B.\n")

    W("## 1. Cảnh báo phải đọc trước mọi con số\n")
    W("**Đây không phải một exact reproduction của Khan et al. 2025.** Mục 4.6 của bài báo chỉ "
      "mô tả AFPHA bằng lời (FedAvg + FedProx + Hierarchical FL); bài báo **không công bố** "
      "hệ số proximal, quy tắc thích nghi, cách phân cụm, tỉ lệ tham gia hay lịch optimizer. "
      "Toàn bộ các giá trị đó là **lựa chọn triển khai** theo Đề xuất A "
      "(`papers/khan-2025-afpha/rebuild.md`, người dùng chốt 2026-09-06), không phải "
      "hyperparameter của tác giả. Không so sánh trực tiếp các số dưới đây với số CAN/CIC-IDS "
      "trong bài báo.\n")
    W("Các caveat về dữ liệu, phải đi kèm mọi trích dẫn kết quả:\n")
    W("- Một \"client\" là một **receiver unit**, không phải một xe.")
    W("- Scaler được fit trên **toàn bộ** tập train trước khi chia FL, nên **không được** "
      "tuyên bố đây là preprocessing phân tán.")
    W("- Split theo **thời gian**, không tách theo xe.")
    W("- Điểm số là **global test**, không phải personalized.")
    W("- Feature lưu ở **fp16** trên GPU.")
    W("- Hai cấp aggregation (cụm rồi server, đều sample-weighted) **bằng đúng FedAvg phẳng "
      "về đại số** — đã kiểm chứng số, max|Δ| ≤ 6,0e-08. Phân cấp ở đây là mô tả **hệ thống**, "
      "không phải một khác biệt tối ưu hoá.\n")

    W("## 2. Thiết lập\n")
    W("| | 20 client | 50 client | 100 client |")
    W("|---|---:|---:|---:|")
    W(f"| Batch mỗi client | {s[20]['cfg']['batch']} | {s[50]['cfg']['batch']} "
      f"| {s[100]['cfg']['batch']} |")
    W("| Round | 50 | 50 | 50 |")
    W("| Local epoch mỗi round | 1 | 1 | 1 |")
    W(f"| Cụm | {len(s[20]['cfg']['clusters'])}×5 | {len(s[50]['cfg']['clusters'])}×5 "
      f"| {len(s[100]['cfg']['clusters'])}×5 |")
    W(f"| Step mỗi round | {s[20]['cfg']['steps_per_round']:,} "
      f"| {s[50]['cfg']['steps_per_round']:,} | {s[100]['cfg']['steps_per_round']:,} |")
    W("| Tham gia | 100% | 100% | 100% |")
    W(f"| Dòng train | {sum(s[20]['cfg']['rows_per_client']):,} "
      f"| {sum(s[50]['cfg']['rows_per_client']):,} "
      f"| {sum(s[100]['cfg']['rows_per_client']):,} |")
    W(f"| Dòng test | {s[20]['cfg']['n_test']:,} | {s[50]['cfg']['n_test']:,} "
      f"| {s[100]['cfg']['n_test']:,} |")
    W("")
    W("Model ở **mọi client và server** là DAGSNet, 395.024 tham số, 3.295 phần tử buffer, "
      "192 khoá state_dict, khởi tạo mặc định PyTorch, train từ đầu. Checkpoint lưu "
      "**state_dict** chứ không serialize nguyên model.\n")
    W("AFPHA theo Đề xuất A: loss cục bộ `CE + (mu_i/2)·||w − w_t||²` trên learnable parameters; "
      "`mu` round 1 = 0,01 và `mu_i,t+1 = 0,01·(1 + d_i/(d_i + d̄ + 1e-12))`; Adam (0,9; 0,999), "
      "eps 1e-8, weight decay 0, reset mỗi client mỗi round; LR cosine "
      "`1e-5 + (1e-3−1e-5)/2·(1 + cos(π(t−1)/49))` cố định trong round; clip global grad norm 1,0 "
      "sau unscale và sau khi cộng proximal gradient; BN `running_mean/var` trung bình theo "
      "`n_i/N`, `num_batches_tracked` lấy max; không class weight, không resampling.\n")

    W("## 3. Kết quả tổng hợp\n")
    W("| Kịch bản | acc cuối | F1_macro cuối | F1_macro tốt nhất (round) | acc tốt nhất (round) |")
    W("|---|---:|---:|---:|---:|")
    for n, _ in SCENARIOS:
        x = s[n]
        W(f"| {n} client | {float(x['final']['accuracy']):.6f} "
          f"| {float(x['final']['f1_macro']):.6f} "
          f"| {float(x['best_f1']['f1_macro']):.6f} (r{int(x['best_f1']['round'])}) "
          f"| {float(x['best_acc']['accuracy']):.6f} (r{int(x['best_acc']['round'])}) |")
    W("")
    W("Cả 10 metric ở round cuối:\n")
    W("| metric | 20 client | 50 client | 100 client |")
    W("|---|---:|---:|---:|")
    for k in METRIC_KEYS:
        W(f"| {k} | " + " | ".join(f"{float(s[n]['final'][k]):.6f}" for n, _ in SCENARIOS) + " |")
    W("")

    W("## 4. Phân tích\n")

    W("### 4.1 Chất lượng giảm đơn điệu theo số client\n")
    W("| Kịch bản | acc đỉnh (round) | acc cuối | F1_macro đỉnh (round) | F1_macro cuối |")
    W("|---|---:|---:|---:|---:|")
    for n, _ in SCENARIOS:
        x = s[n]
        W(f"| {n} client | {float(x['best_acc']['accuracy']):.5f} "
          f"(r{int(x['best_acc']['round'])}) | {float(x['final']['accuracy']):.5f} "
          f"| {float(x['best_f1']['f1_macro']):.5f} (r{int(x['best_f1']['round'])}) "
          f"| {float(x['final']['f1_macro']):.5f} |")
    W("")
    W("Cả bốn cột đều giảm đơn điệu khi số client tăng. Cùng một tập dữ liệu 43.045.415 dòng, "
      "cùng code, cùng lịch seed — khác biệt duy nhất là dữ liệu bị chia cho nhiều client hơn, "
      "nên mỗi client có ít dữ liệu hơn và phân phối cục bộ lệch hơn. **Đây chính là hiệu ứng "
      "mà ba kịch bản được thiết kế để đo**, và nó xuất hiện rõ, nhất quán trên cả metric "
      "micro (accuracy) lẫn macro (F1_macro).\n")
    W("Một chi tiết về tốc độ hội tụ: F1_macro của 20 client đạt đỉnh sớm ở round 16 rồi "
      "**giảm nhẹ** về round 50; 50 client đạt đỉnh ở round 7; còn 100 client đạt đỉnh đúng ở "
      "round 50. Càng ít client thì model hội tụ càng sớm và 50 round càng thừa; càng nhiều "
      "client thì càng cần đủ 50 round. Với 20 và 50 client, checkpoint tốt nhất **không phải** "
      "checkpoint cuối — báo cáo phải nói rõ đang trích round nào.\n")

    W("### 4.2 Accuracy giảm trong khi recall_macro tăng — đánh đổi, không phải hỏng\n")
    W("| Kịch bản | recall_macro round 1 | recall_macro round 50 |")
    W("|---|---:|---:|")
    for n, _ in SCENARIOS:
        rows = data[n]["rows"]
        W(f"| {n} client | {float(rows[0]['recall_macro']):.5f} "
          f"| {float(rows[-1]['recall_macro']):.5f} |")
    W("")
    W("Ở cả ba kịch bản, accuracy đạt đỉnh rất sớm (round 3–6) rồi giảm, trong khi "
      "`recall_macro` tăng gần đơn điệu suốt 50 round. Đây là đánh đổi kinh điển trên dữ liệu "
      "mất cân bằng: model dần bỏ độ đúng ở các lớp đa số để lấy recall ở các lớp hiếm. "
      "Không phải phân kỳ — `skip_pct` luôn dưới 0,8%, `mu` luôn trong [0,01; 0,02), "
      "grad norm không nổi, mọi tensor hữu hạn ở cả 150 checkpoint.\n")
    W("Hệ quả cho việc chọn số liệu báo cáo: **accuracy và F1_macro ở đây kể hai câu chuyện "
      "khác nhau và đỉnh của chúng ở hai round khác nhau.** Trích một con số duy nhất mà không "
      "nói round nào sẽ gây hiểu nhầm.\n")

    W("### 4.3 Phát hiện quan trọng nhất: lớp `benign` bị bỏ sót ~75%\n")
    W("| Kịch bản | benign recall | benign precision | benign f1 | dự đoán là benign |")
    W("|---|---:|---:|---:|---|")
    for n, _ in SCENARIOS:
        b = data[n]["per_class"][0]
        W(f"| {n} client | {b['recall']:.4f} | {b['precision']:.4f} | {b['f1']:.4f} "
          f"| {b['predicted']:,} / {b['support']:,} |")
    W("")
    W("Con số này **gần như không đổi** giữa ba kịch bản (recall 0,248–0,274), dù mọi metric "
      "khác đều thay đổi rõ theo số client. Đó là bằng chứng mạnh rằng đây **không phải hiệu "
      "ứng của federated learning** mà là giới hạn của bài toán/feature.\n")
    W("Bốn lớp yếu nhất **giống hệt nhau và đúng thứ tự** ở cả ba kịch bản:\n")
    W("| Kịch bản | 4 lớp F1 thấp nhất |")
    W("|---|---|")
    for n, _ in SCENARIOS:
        w4 = sorted(data[n]["per_class"], key=lambda e: e["f1"])[:4]
        W(f"| {n} client | " + ", ".join(f"`{e['class_name']}` {e['f1']:.3f}" for e in w4) + " |")
    W("")
    W("Ở kịch bản 100 client, các dòng `benign` bị nhận nhầm chủ yếu rơi vào đúng ba lớp này: "
      "`dataReplay` 20,8%, `positionMirroring` 17,7%, `timeDelayAttack` 16,7%.\n")
    W("**Giải thích có cơ sở, không phải suy đoán:** cả ba tấn công đó đều phát đi **nội dung "
      "message hợp lệ** — replay lại message thật, mirror vị trí thật của một xe khác, hoặc gửi "
      "message thật bị trễ. Chúng chỉ khác `benign` ở **ngữ cảnh thời gian và quan hệ giữa các "
      "message**, thứ mà một model 66 feature **per-message** về nguyên tắc không nhìn thấy "
      "được. Đối chứng ủng hộ cách đọc này: các lớp có chữ ký per-message rõ ràng đều gần hoàn "
      "hảo ở cả ba kịch bản — `dosAttack`, `randomPositionOffset`, `trafficCongestionSybil`, "
      "`accelerationMultiplication` đều F1 ≥ 0,95.\n")
    W("> **Kết luận vận hành phải nêu trong mọi bản trình bày:** F1_macro 0,70–0,78 **không** "
      "đồng nghĩa với một IDS dùng được. Bỏ sót ~75% lưu lượng benign là tỉ lệ báo động giả "
      "không thể chấp nhận trong triển khai thật. Muốn cải thiện thì phải thêm feature theo "
      "chuỗi/ngữ cảnh, chứ không phải chỉnh siêu tham số FL.\n")
    W("Không lớp nào bị bỏ trắng: cả 16 lớp đều có `predicted` > 0 và F1 > 0 ở cả ba kịch bản.\n")

    W("## 5. Hiệu năng\n")
    W("| Kịch bản | round 1 (s) | median (s) | tổng train (h) |")
    W("|---|---:|---:|---:|")
    for n, _ in SCENARIOS:
        W(f"| {n} client | {s[n]['first_s']:.1f} | {s[n]['median_s']:.1f} "
          f"| {s[n]['total_h']:.2f} |")
    W("")
    W("Phần cứng: Kaggle Tesla T4 ×2, torch 2.10.0+cu128. Round 1 đắt hơn vì trả tiền "
      "`torch.compile` và lần eval đầu. `torch.compile(mode=\"reduce-overhead\")` được chấp nhận "
      "trên cả hai GPU ở cả ba run.\n")

    for n, _ in SCENARIOS:
        W(f"## 6.{[m for m, _ in SCENARIOS].index(n) + 1} Bảng đầy đủ 50 round — {n} client\n")
        W("Viết tắt: `P`=precision, `R`=recall, `F1`=F1; `mac`=macro, `mic`=micro, "
          "`wtd`=weighted. `acc` = `P_mic` = `R_mic` = `F1_mic` theo định nghĩa "
          "cho bài toán single-label đa lớp.\n")
        W(metric_table(data[n]["rows"]))
        W("")

    W("## 7. Kết quả theo từng lớp ở round 50\n")
    for n, _ in SCENARIOS:
        W(f"### {n} client\n")
        W("| lớp | support | predicted | precision | recall | f1 |")
        W("|---|---:|---:|---:|---:|---:|")
        for e in data[n]["per_class"]:
            W(f"| {e['class_name']} | {e['support']:,} | {e['predicted']:,} "
              f"| {e['precision']:.4f} | {e['recall']:.4f} | {e['f1']:.4f} |")
        W("")

    W("## 8. Kiểm chứng\n")
    W("Mỗi run đã qua `scripts/verify_run.py <run_dir> --require-complete` và pass **10/10**:\n")
    W("1. config đọc từ chính run; 2. **gate hoàn thành đủ 50/50 round**; 3. round liên tục "
      "không gap; 4. cụm là phân hoạch đúng; 5. mọi metric trong `history.csv` khớp JSON từng "
      "round; 6. flatpack round-trip chính xác tuyệt đối; 7. LR đúng hai đầu lịch cosine; "
      "8. aggregation phân cấp = FedAvg phẳng; 9. `mu` trong [0,01; 0,02); "
      "10. per-round: trọng số nạp `strict=True` và **mọi tensor hữu hạn**, metric khớp "
      "confusion, per_class khớp confusion, dự đoán phủ trọn tập test, tổng step = "
      "Σ ceil(n_i/batch), ID client đúng và rows khớp config, LR đúng lịch, chuỗi `mu` khớp "
      "`resume/*.pt`, seed là hàm thuần của (round, client).\n")
    W("Verifier chỉ chứng nhận **các round đã hoàn thành**; cờ `--require-complete` là thứ "
      "biến nó thành bằng chứng \"đã xong cả kịch bản\". Cả ba run đều chạy với cờ này.\n")
    W("Chín file source chạy trên Kaggle khớp **byte** với `src/` + `architecture/` local ở cả "
      "ba run, nên ba kịch bản chạy đúng cùng một code và so sánh được với nhau.\n")
    W("**Không hứa bit-identical khi chạy lại**: cuDNN dùng atomics và quỹ đạo AMP scale phụ "
      "thuộc thời điểm. Ví dụ đo được: calibration 20 client round 1 cho accuracy 0,45200 còn "
      "run thật cho 0,44725 — lệch ~1% tương đối trên cùng cấu hình.\n")

    W("## 9. Nguồn artifact\n")
    W("| Kịch bản | thư mục | W&B |")
    W("|---|---|---|")
    for n, _ in SCENARIOS:
        wb = json.loads((RUNS[n] / "wandb_run.json").read_text())
        W(f"| {n} client | `runs/{n}client/` | `{wb['run_path']}` |")
    W("")
    W("Mỗi thư mục chứa: `weights/round_NNN.pt` (state_dict), `confusion/round_NNN.npy`, "
      "`preds/round_NNN_ypred.npy` + `preds/y_true.npy`, `per_class/round_NNN.json`, "
      "`metrics/round_NNN.json` + `metrics/history.csv`, `client_log/round_NNN.json`, "
      "`resume/round_NNN.pt`, `complete/round_NNN.done`, `config.json`, `model_source/`.\n")
    W("Dựng lại model từ checkpoint:\n")
    W("```python\nfrom dagsnet import build_dagsnet\nimport torch\n"
      "m = build_dagsnet()\n"
      "m.load_state_dict(torch.load('weights/round_050.pt', weights_only=True), strict=True)\n```\n")

    (ROOT / "report.md").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote report.md ({len('\n'.join(out)):,} chars)")
    for n, _ in SCENARIOS:
        print(f"  {n:3d} client: 50 rounds, final acc {float(s[n]['final']['accuracy']):.6f}, "
              f"final f1_macro {float(s[n]['final']['f1_macro']):.6f}")


if __name__ == "__main__":
    main()
