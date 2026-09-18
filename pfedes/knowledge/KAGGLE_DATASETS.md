# Dataset VeReMi trên Kaggle — bốn bộ, đều PUBLIC

Bốn bộ dữ liệu đầu vào dùng chung cho **mọi phương pháp** xây trên VeReMi NextGen 2026.
Nội dung, cột, scaler, số dòng và phân bố lớp: [`DATASET.md`](DATASET.md).

**Kiểm ngày 2026-09-08 bằng CLI của `odixe0502`.** Cách kiểm: dataset xuất hiện trong tìm kiếm
**không** có cờ `-m` thì là public.

| Dataset | Chủ sở hữu | Trạng thái | Kích thước | Nội dung |
|---|---|---|---|---|
| `veremi-fl-20client` | odixe0502 | **public** | 7,84 GB | train + test, phân mảnh 20 client |
| `veremi-fl-50client` | odixe0502 | **public** | 8,69 GB | train + test, phân mảnh 50 client |
| `veremi-fl-100client` | odixe0502 | **public** | 9,32 GB | train + test, phân mảnh 100 client |
| `veremi-nextgen2026-centralized` | odixe0502 | **public** | 9,30 GB | tập test toàn cục cố định |

## Hệ quả cần nhớ

**Public nghĩa là mọi tài khoản Kaggle mount được, không cần share tay.** Điều này quan trọng
khi một phương pháp phải chạy trên nhiều tài khoản để đủ quota GPU: bốn bộ này không phải là
rào cản, và **không cần** kiểm quyền đọc bằng từng tài khoản nữa.

Ghi chú cũ "group `nckh_minhtriet` không đảm bảo quyền đọc từng dataset, phải kiểm bằng chính
tài khoản sẽ chạy" vẫn đúng **về nguyên tắc chung**, nhưng với riêng bốn bộ này thì câu hỏi
đã khép lại vì chúng public.

**Quyền đọc dataset khác quyền đọc output của notebook.** Dataset public không cấp quyền đọc
kernel output riêng tư của tài khoản khác; bàn giao checkpoint giữa hai tài khoản là vấn đề
riêng, không hưởng lợi gì từ việc bốn dataset này public.

**Public dataset không kéo theo public notebook.** Notebook nào nhúng credential (W&B key,
token) thì phải giữ `is_private: true`; public hoá là lộ khoá.

## Cách kiểm lại

```bash
kaggle datasets list -s veremi-fl-100client --csv   # có trong kết quả => public
kaggle datasets files odixe0502/veremi-fl-100client
```
