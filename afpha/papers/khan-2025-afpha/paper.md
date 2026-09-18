# Nguồn phương pháp AFPHA

Khan et al. (2025), *A secure and efficient deep learning-based intrusion detection
framework for the internet of vehicles*, Scientific Reports 15, 12236.
[Bài gốc](https://www.nature.com/articles/s41598-025-94445-9).
Bản đọc trong repo: [s41598-025-94445-9.md](../../s41598-025-94445-9.md).

## Nội dung có trong bài

Mục 4.6 mô tả AFPHA kết hợp FedAvg, FedProx và Hierarchical Federated Learning:
client train cục bộ, gửi cập nhật, cụm và server tổng hợp mô hình. FedProx thêm
proximal term để hạn chế lệch do dữ liệu không đồng nhất. Phần này còn nói về
mã hóa cập nhật, thuộc phạm vi bảo mật rộng hơn vòng lặp huấn luyện.

Mục 4.10, Eq. (38)–(50), mô tả bốn nhánh DenseNet, GoogleNet, AlexNet, SqueezeNet
và hợp nhất thành DAGSNet. Eq. (50) cho phép đầu ra nhiều lớp.

Table 1 ghi batch 64, 100 epochs, LR 0.001 với adaptive decay, categorical
cross-entropy và optimizer mang tên CMSO. Phần mô tả CMSO cũng nói đến Adam.
Các thông tin này không phải một đặc tả FL có thể chạy trực tiếp.

## Những chỗ chưa đủ để tái lập AFPHA

- Không có công thức adaptive AFPHA hoặc pseudocode của vòng FL.
- Không định lượng proximal coefficient, anchor, tỷ lệ tham gia, số/cách chia cụm.
- Không chốt số lần cập nhật cụm trước khi server tổng hợp, hoặc trọng số hai cấp.
- Không chốt optimizer cục bộ, trạng thái optimizer giữa round, LR decay cụ thể.
- Không mô tả cách tổng hợp BatchNorm, khởi tạo và checkpoint/resume.

Vì vậy bản triển khai cần ghi rõ lựa chọn bổ sung; không được gọi các giá trị tự
chọn là hyperparameter AFPHA gốc. Không suy ra mọi client có cùng trọng số chỉ
vì cùng kiến trúc.

## Nguồn định nghĩa FedProx bổ trợ

[Bài FedProx, MLSys 2020](https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf)
và [code của tác giả](https://github.com/litian96/FedProx).
Local objective có dạng empirical loss cộng mu/2 nhân bình phương khoảng cách
đến trọng số global được phát ở đầu round. Giá trị mu cần lựa chọn cho bài toán.
Nguồn này không cung cấp quy tắc adaptive còn thiếu của AFPHA.

Kết quả CAN/CIC-IDS trong bài không được dùng làm kết quả kỳ vọng hay đối chiếu
trực tiếp với VeReMi 16 lớp, dataset và bộ metrics khác.
