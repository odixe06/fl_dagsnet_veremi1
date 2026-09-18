# TinyProto — tóm tắt nguồn gốc phương pháp

Nguồn trực tiếp: [bản Markdown](../../02846-AAAI26.LeeG-ML.md) và
[PDF được cung cấp](../../02846-AAAI26.LeeG-ML.pdf), Gyuejeong Lee và Daeyoung Choi,
*Communication-Efficient Heterogeneous Federated Learning with Sparse Prototypes in
Resource-Constrained Environments*, submission AAAI-26 số 02846.
Các thay đổi của VeReMi/DAGSNet nằm trong [rebuild.md](rebuild.md), không phải trong bài báo.

Bài báo giảm chi phí truyền prototype bằng CPS: cố định mask nhị phân riêng cho mỗi lớp,
giữ s trong d chiều, chỉ truyền các chiều được giữ. APS phía client nhân prototype đã nén
với số mẫu lớp n_ij; phía server lấy trung bình trên các client có lớp j (Eq.10). Round sau
regularizer dùng prototype toàn cục đã nhân μ (Eq.11), cộng vào classification loss với λ
(Eq.5). Round đầu không có regularizer. Mỗi client giữ mô hình riêng; TinyProto-FP không
trung bình trọng số mô hình. Dự đoán PBFL dùng khoảng cách L2 đến prototype LOCAL dày (Eq.12).
TinyProto-FT là biến thể tích hợp vào FedTGP, ngoài phạm vi bản dựng được chọn.

| Thiết lập | Nguồn §5.1 |
|---|---|
| Dataset chính | CIFAR-10/100, GTSRB-43, Flowers-102, Tiny ImageNet-200 |
| Train/test | 75%/25% |
| Non-IID | Dirichlet α=0,1 |
| Client/participation | 20, tất cả mỗi round |
| Kiến trúc | ResNet-8, EfficientNet, ShuffleNet v2, MobileNet v2 |
| Prototype d/s | 500/50 |
| Round/local epoch/batch | 300/1/32 |
| Learning rate | **0,01** |
| Optimizer | Không nêu rõ tên trong phần chính được cung cấp |
| λ / scheduler | 1 / không dùng scheduler |
| μ | CIFAR-10 1,5e-4; GTSRB 5e-4; Flowers 5e-3; CIFAR-100 và Tiny ImageNet 1,5e-3 |
| Đánh giá | Trung bình accuracy trên các client từng round; báo best global round |
| Lặp lại | Ba seed, báo trung bình |

Table 1 cho TinyProto-FP: accuracy lần lượt 86,20±0,08; 85,84±0,59; 40,39±0,34;
36,25±0,10; 20,06±0,20 (%) theo thứ tự năm cột CIFAR-10, GTSRB, CIFAR-100, Flowers,
Tiny ImageNet. Đây là số nguồn, không phải kết quả VeReMi và không dùng để tính delta so sánh.

Phần chính nói quy trình chọn μ và cấu hình bổ sung nằm ở appendix; bản PDF được cung cấp
không có appendix. Không suy diễn grid search, AdamW, masked per-sample MSE, optimizer
persistence hoặc thuật toán tối ưu mask cụ thể là những chi tiết tác giả đã công bố.
