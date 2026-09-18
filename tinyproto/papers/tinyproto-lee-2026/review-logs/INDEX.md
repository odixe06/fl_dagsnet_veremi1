# Chỉ mục log — đọc file nào khi cần gì

Đây là **log thô**. Kết luận đã tổng hợp ở [`../TESTLOG.md`](../TESTLOG.md); đừng suy luận
kết quả trực tiếp từ log ở đây mà không đối chiếu TESTLOG.

Quy ước tên: `-final` = lần chạy cuối được chấp nhận làm bằng chứng của phiên đó.
`-session2` / `-session3` = chạy lại sau các sửa đổi của phiên tương ứng.
Không có hậu tố = lần chạy sớm hơn, **giữ làm lịch sử**, không phải trạng thái hiện tại.

## Kaggle — chạy thật trên 2 × T4 (quan trọng nhất)

| File | Nội dung |
|---|---|
| `kaggle-odixe-v2.log` | **Bounded validation PASSED**, cả 3 kịch bản, 293 s |
| `kaggle-odixe-v1.log` | Lần 1, bị huỷ giữa kịch bản 100; lần đầu xác nhận 2 × T4 |
| `kaggle-odixe-calibration.log` | **Calibration toàn dữ liệu**, 3.430 s, 3 round 20-client |
| `kaggle-mu-sweep-{20,50,100}client-final.log` | **Log kernel đầy đủ của 3 sweep μ** — nguồn của compile report (`amp_*`), timing, và quỹ đạo reg_loss |
| `mu-sweep{,-50client,-100client}-result.json` | `mu_sweep.json` đã pull: người thắng, toàn bộ lưới, chữ ký |
| `calibration.json` | Số đo máy đọc được của lần calibration trên |
| `kaggle-identity.log`, `kaggle-push-*.log`, `kaggle-pull-v1.log` | Thao tác tài khoản/push/pull |
| `reference-gpu-image.json` | (lịch sử) nguồn digest image; nay đã chuyển sang `knowledge/runtime.json` |

## Bộ test local — tổng hợp

| File | Nội dung |
|---|---|
| `suite-session3.log` | Suite sau các sửa R1–R6 (phiên 3) |
| `suite-final-session2.log`, `suite-session2.log` | Suite phiên 2 |
| `suite-final.log` | Suite phiên 1 |

## Bộ test local — từng phần

| File | Nội dung |
|---|---|
| `unit-final.log` | 10 metric đối chiếu sklearn |
| `resume-final.log` | 6 kịch bản crash/resume, có so bitwise |
| `tamper-final.log` | 9 kiểu hỏng artifact cố ý → phải bắt được hết |
| `verify-final.log` | verifier trên run sạch |
| `real-data-smoke.log` | forward/backward trên dòng dữ liệu thật |
| `regressions-final.log`, `regressions-calibration-final.log`, `prelaunch-regressions.log` | Regression theo từng đợt review |
| `fixture-final.log` | Dựng fixture |
| `generate-final.log`, `validate-final.log`, `validate-regenerated-final.log` | Sinh và kiểm 7 notebook |

## Mô phỏng notebook (chạy cell thật trên fixture)

`tinyproto-fp-calibration-final.log`, `tinyproto-fp-mu-sweep{,-50client,-100client}-final.log`,
`tinyproto-fp-train-{20,50,100}client-final.log`, `train20-sim.log`, `mu20-sim*.log`, `verify-sim.log`.

> **Bẫy đặt tên — đọc trước khi trích dẫn.** `tinyproto-fp-mu-sweep-100client-final.log` (~6 KB)
> là **mô phỏng local** trên fixture 4 client × 2048 dòng, 2 round. Log kernel Kaggle thật là
> `kaggle-mu-sweep-100client-final.log` (~119 KB), 100 client × toàn dữ liệu, 4 round.
> Hai file tên gần giống nhau nhưng **không phải cùng một bằng chứng**; số đo thời gian,
> `amp_*` và reg_loss chỉ có giá trị ở bản `kaggle-*`. Phân biệt nhanh bằng kích thước.

## Rà soát trước chạy thật (phiên 3)

| File | Nội dung |
|---|---|
| `prelaunch-review-checks.py` | Script chẩn đoán, chạy lại được, không train |
| `prelaunch-review-checks.log` | Kết quả — các dòng `OBSERVED` là **thiếu sót tại thời điểm đó** (R1–R3), nay đã sửa và có regression |

## Review bộ production chia session — 2026-09-08

| File | Nội dung |
|---|---|
| `production-review-checks.py` | Diagnostic CPU: 7 production notebook thật, μ/config/fingerprint, 15 metric đã pull; tái hiện thiếu sót validator/AMP/W&B. Không khởi chạy Kaggle. |
| `production-review-checks.log` | Pass kiểm bộ hiện tại, peak 645 MiB. Các dòng `GAP` là vấn đề còn mở, không phải đã sửa. |
| `production-review-regressions.log` | 20 regression pass, peak 620 MiB. |

Kết luận và hướng xử lý ở [`../PRODUCTION_REVIEW.md`](../PRODUCTION_REVIEW.md), CONTEXT §8.

## Lịch sử / bối cảnh

`*-after-reboot.log` chạy lại sau sự cố tràn RAM WSL; `unit-before.log`, `regressions.log`,
`fixture.log`, `generate.log`, `remote-build*.log` là các lần sớm hơn.
`mu20-sim.log` và `resume.log` rỗng (0 byte) — **không phải bằng chứng pass**.
