# `knowledge/` — kiến thức dùng chung, KHÔNG phụ thuộc phương pháp

Thư mục này phục vụ **nhiều phiên làm việc và nhiều phương pháp khác nhau** dựng trên cùng bộ
dữ liệu và cùng hạ tầng. Vì thế nó chỉ chứa những gì **không đổi khi đổi phương pháp**.

## Được ghi ở đây

- Thuộc tính của **dữ liệu**: cột, scaler, số dòng, phân bố lớp, thống kê phân mảnh theo client.
- Thuộc tính của **máy local**: RAM/GPU, ngưỡng an toàn, cách chạy test không làm sập WSL.
- Thuộc tính của **hạ tầng dùng chung**: digest image Kaggle đã kiểm, trạng thái public/private
  của các dataset đầu vào, cách đổi tài khoản.
- Kiến trúc mô hình dùng chung cho nhiều phương pháp.

## KHÔNG được ghi ở đây

- Siêu tham số, hằng số, hay giá trị đã chọn của **một** phương pháp (ví dụ μ, s, λ).
- Kết quả, log, manifest, hay artifact của **một** lần chạy.
- Trạng thái đổi theo thời gian: số dư quota, notebook nào đang chạy, việc cần làm tiếp.
- Trỏ tới `CONTEXT.md` hay thư mục `papers/<phương pháp>/`.

Những thứ đó thuộc về `papers/<phương pháp>/` và `CONTEXT.md`.

## Vì sao nghiêm khắc

Một con số của phương pháp A nằm trong `knowledge/` sẽ được phiên sau đọc như **sự thật của dữ
liệu** khi đang dựng phương pháp B, và không ai kiểm lại vì nó nằm ở nơi dành cho sự thật đã
kiểm. Đó là cách một lựa chọn thiết kế bị nhầm thành thuộc tính khách quan.

Ví dụ đã xảy ra (đã sửa 2026-09-08): bảng "ước lượng μ" và bảng chi phí truyền tin theo `s = 50`
từng nằm trong `DATASET.md`. Cả hai phụ thuộc TinyProto; đã chuyển sang
`papers/tinyproto-lee-2026/rebuild.md` §8. Thống kê `n_ij` gốc thì ở lại, vì đó là dữ liệu.
