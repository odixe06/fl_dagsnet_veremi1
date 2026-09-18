# AFPHA–DAGSNet: đặc tả đề xuất để chốt

**Trạng thái: đặc tả đã chốt 2026-09-06, code đã viết và test local, chưa launch.**
Ngày chuẩn bị: 2026-09-06. [Phân biệt với bài gốc](paper.md).

## Yêu cầu đã xác nhận

| Thành phần | Giá trị |
|---|---|
| Kịch bản | 20, 50, 100 client |
| Round | 50, đánh số 1–50 |
| Local epoch | 1 lượt đầy đủ trên mỗi client tham gia mỗi round |
| Batch mỗi client | 512 / 512 / 256 tương ứng |
| Kiến trúc | DAGSNet đúng architecture/ARCHITECTURE.md, 395.024 tham số |
| Khởi tạo | mới, PyTorch mặc định; không nạp checkpoint round 5 |
| Đánh giá | chỉ global sau aggregation, toàn bộ test mỗi round |
| Metrics | accuracy; precision/recall/F1 × macro/micro/weighted |
| Lưu trữ | trọng số state_dict, không serialize nguyên đối tượng model |
| Nơi train dự kiến | Kaggle, Tesla T4 ×2, 16 GB mỗi GPU |

## Dữ liệu

- Train local: /home/odixe/nckh/dataset/fl_client/alpha05/{20,50,100}_client/train/.
- Kaggle: odixe0502/veremi-fl-{20,50,100}client, version 1.
- Test local: /home/odixe/nckh/dataset/centralized/test/.
- Kaggle test: odixe0502/veremi-nextgen2026-centralized, upload/test/, version 1.
- Dữ liệu train đã z-score bằng scaler toàn cục fit trên train; không chuẩn hóa lại.
  Test thô phải dùng scaler đó. Chỉ dùng 66 cột f_* theo đúng thứ tự frozen.
- Ba phân hoạch Dirichlet alpha=0.5 có cùng 43.045.415 dòng train theo sidecar.
  Một client là tập receiver unit, không phải một xe vật lý.
- Scaler dùng thống kê của mọi client trước FL; không tuyên bố quy trình preprocessing
  bảo mật phân tán. Giữ caveat Sybil, split thời gian, lớp benign và mất cân bằng
  trong tài liệu dữ liệu/kiến trúc.
- Audit thực tế chạy bằng ../../scripts/audit_fl_data.py; đầu ra audit.json.
  Không coi sidecar là bằng chứng đã quét lại. Script không đo trùng dòng hoặc
  giao nhau flow train/test; những nội dung đó vẫn là giới hạn của audit này.

Audit full đã chạy thành công ngày 2026-09-06: 446/1.069/1.992 file train,
43.045.415 dòng/kịch bản, 4 file test với 10.761.343 dòng. Đã quét mọi ô của
66 đặc trưng và mọi nhãn; không có non-finite, số mẫu/nhãn khớp manifest/sidecar,
train tổng và test đều đủ 16 lớp. Không có cột đặc trưng hằng.
Mean tuyệt đối lớn nhất của train là 3.05e-8; std trong [0.9999999609, 1.0000000305],
xác nhận train đã chuẩn hóa. Feature order, lớp và scaler khớp tài liệu kiến trúc.

## Đề xuất A — hoàn thiện mô tả AFPHA để có thể triển khai

**Người dùng đã chốt nguyên Đề xuất A ngày 2026-09-06.** Đây là đặc tả thi hành của dự án,
đã được cài trong src/afpha.py và kiểm chứng số (xem CONTEXT.md mục 7).

Các giá trị sau vẫn là **lựa chọn triển khai**, không phải công thức/hyperparameter công bố
của AFPHA. Không gọi bản này là exact reproduction.

### Tham gia và phân cấp

100% client tham gia mỗi round. Chia cố định 5 client/cụm, tương ứng 4/10/20 cụm.
Sinh hoán vị ID client một lần với numpy default_rng(42), chia thành nhóm 5 rồi
lưu clusters.json. Cụm là mô phỏng logic, không suy diễn vị trí địa lý/RSU từ ID.

Đầu round t, mọi client nhận cùng state_dict global w_t, bao gồm BN buffers.
Mỗi client train 1 epoch trên dữ liệu riêng, shuffle theo seed xác định bởi
(42, t, client_id). Không bỏ batch cuối, không trộn dữ liệu giữa client.

Với n_i là số mẫu của client và N_g tổng số mẫu của cụm g:

$$ w_{g,t+1} = \sum_{i\in g}\frac{n_i}{N_g}w_{i,t+1} $$

$$ w_{t+1} = \sum_g\frac{N_g}{N}w_{g,t+1} $$

Một lần tổng hợp cụm và một lần tổng hợp global mỗi round. Do đó hai cấp trung bình
này bằng FedAvg trọng số theo số mẫu về mặt đại số; không tuyên bố phân cấp tự tạo
lợi ích tối ưu hóa. Phần adaptive nằm ở proximal coefficient bên dưới.

### Local objective và adaptive proximal

CrossEntropy không class weights/resampling, cộng proximal trên toàn bộ learnable
parameters (bao gồm bias và BN affine; không áp dụng vào running buffers):

$$ L_{i,t}(w)=\operatorname{CE}_i(w)+\frac{\mu_{i,t}}{2}\|w-w_t\|_2^2 $$

Round 1: mu_i,1 = 0.01 cho mọi client.
Sau round t, đo độ lệch learnable parameters của client với global mới:

$$ d_{i,t}=\|w_{i,t+1}-w_{t+1}\|_2,\qquad \bar d_t=\sum_i\frac{n_i}{N}d_{i,t} $$

Round tiếp theo:

$$ \mu_{i,t+1}=0.01\left(1+\frac{d_{i,t}}{d_{i,t}+\bar d_t+10^{-12}}\right) $$

Hệ số nằm từ 0.01 đến dưới 0.02; client lệch nhiều bị ràng buộc mạnh hơn ở round sau.
Tính d và proximal reduction bằng fp32; không chia proximal theo số tham số.
Lưu d, mu theo client mỗi round để kiểm toán/resume. Nếu mọi độ lệch bằng 0, mu=0.01.

**Đây là heuristic đề xuất, chưa được kiểm chứng hội tụ hoặc hiệu quả trên VeReMi.**
Nó hoàn thiện phần adaptive còn thiếu bằng thông tin cập nhật train, không dùng
test metrics. Nếu cần tái lập chính xác AFPHA tác giả, cần thêm code/đặc tả từ tác giả;
không có cách xác nhận heuristic này trùng phương pháp gốc.

### Optimizer và lịch học

Adam cục bộ, betas=(0.9,0.999), eps=1e-8, weight_decay=0; reset optimizer và AMP scaler
ở mỗi lần bắt đầu train client trong round. LR cosine cố định theo round, không
điều chỉnh theo test:

$$ \eta_t=10^{-5}+\frac{10^{-3}-10^{-5}}{2}\left(1+\cos\frac{\pi(t-1)}{49}\right) $$

LR round 1=0.001; round 50=0.00001, giữ cố định trong từng local epoch.
Clip global gradient norm=1.0 sau unscale; tính/lưu CE, proximal loss, gradient norm,
số optimizer steps và AMP skipped steps. Lỗi non-finite không được làm mất client
một cách im lặng hoặc ghi đè checkpoint hợp lệ.

Adam, cosine, mu và clipping đều là lựa chọn triển khai, không phải tái lập CMSO.
Standalone DAGSNet theo yêu cầu không bao gồm DWT/ViT/GAT/CMSO hay HE/AES/SMPC.
Đây là mô phỏng thuật toán FL trên một máy, không là hệ thống bảo mật phân tán.

### BatchNorm

Tổng hợp running_mean và running_var theo cùng trọng số n_i/N như tham số.
num_batches_tracked lấy max theo cây tổng hợp, giữ dtype integer. Không dùng FedBN,
không recalibrate BN trên test. Trung bình running_var là quy ước hợp nhất buffer,
không được diễn giải thành phương sai pooled chính xác của mọi client.
BN momentum dùng mặc định cố định 0.1 như kiến trúc; counter không được đưa vào proximal.

## Thiết kế tốc độ dự kiến

- Hai worker persistent, mỗi GPU train một client độc lập; gửi client lớn trước
  để giảm thời gian chờ cuối round. Aggregation theo thứ tự ID cố định.
- Giữ model allocation, nạp client vào VRAM nếu đủ khoảng trống đo được. Client lớn
  nhất 5.890.990 dòng tương đương khoảng 1.45 GiB riêng X float32 66 cột; còn phải
  tính labels, permutations, model, optimizer và activations.
- Chuẩn bị cache dạng mmap theo từng client, đọc Parquet có giới hạn RAM; không nhân
  đôi toàn bộ dữ liệu train/test trong RAM. Chỉ cache input cần dùng cho kịch bản đang chạy.
- fp16 AMP + GradScaler trên T4; proximal fp32. Benchmark compile/eager trên model
  thật và batch đã chốt trước khi chọn. Không tăng batch để đẩy VRAM.
- Chia test thành hai shard không giao nhau, cùng global state trên hai GPU,
  ghép dự đoán đúng thứ tự và cộng confusion counts rồi tính metrics một lần.
- Đo throughput, thời gian train/eval/aggregation/I/O, peak VRAM từng GPU. Chưa có
  benchmark nên chưa ước tính 50 round có vừa 30 giờ quota hay không.
- Một notebook riêng cho mỗi kịch bản; lên lịch launch theo quota thực tế. Không
  dự kiến giữ đồng thời cả ba kịch bản trong một session.

## Checkpoint, kết quả và resume

Mỗi round lưu weights/round_NNN.pt chỉ chứa state_dict CPU fp32 cùng BN buffers.
Khoảng 1.52 MiB tensor/round, khoảng 76 MiB/50 round mỗi kịch bản, chưa kể overhead.
Lưu một file resume_state.pt mới nhất, chỉ tensors/primitives, gồm round, fingerprint,
seed/RNG và adaptive state. Vì optimizer reset mỗi client/round, không cần giữ mọi
Adam state ở ranh giới round. Không hỗ trợ tiếp tục giữa một client epoch ở bản đề xuất.

Gói tái dựng gồm model.py, model_config.json, meta.json, scaler.json và weights.
Kiểm tra torch.load(weights_only=True), load_state_dict(strict=True), số tham số
và logits trên batch cố định.

Mỗi round giữ:

- metrics/round_NNN.json và history.csv: đủ 10 metrics.
- confusion/round_NNN.npy và per-class report.
- preds/round_NNN.npz: y_true và y_pred uint8, không lưu probabilities mặc định.
- logs theo client: samples, steps, loss, mu, drift, skips và timing.
- weights, hoàn tất artifact rồi mới viết completion marker/advance last.

Với 10.761.343 test rows, y_true+y_pred uint8 là khoảng 20.53 MiB/round trước nén,
1.00 GiB/50 round mỗi kịch bản. Tổng ngân sách cần cộng cache, logs và trọng số.
Không lưu model riêng mọi client mỗi round vì user yêu cầu đánh giá global và chưa
yêu cầu lưu lịch sử client; chỉ giữ thông tin cần cho thuật toán/resume.

Session mới cần attach output cũ qua kernel_sources và require_resume=true.
Round đang dở có thể chạy lại; round đã commit đầy đủ không được train lại hay ghi
trùng history. Stop budget phải dựa trên quota và thời gian round đã đo, có dự phòng
cho eval/lưu output. Bảng kết quả dùng round cuối, không chọn checkpoint tốt nhất theo test.

## Điều kiện môi trường đã kiểm tra

- Kaggle MCP có callable tools, metadata bốn dataset trả version 1 Ready.
- **Cập nhật 2026-09-06: MCP get_accelerator_quota đã thành công** — 30 h GPU còn nguyên,
  refresh 2026-09-12. Các dòng "Unauthenticated" bên dưới là lịch sử, không còn là trạng thái.
- .codex/config.toml trỏ endpoint chính thức và đọc Authorization từ
  KAGGLE_MCP_AUTHORIZATION; biến này vắng trong process shell hiện tại.
- Probe JSON-RPC trực tiếp bằng header có sẵn trong .mcp.json đã initialize/list tools
  được nhưng account-scoped quota vẫn trả lỗi. Credential hiện có chưa xác thực được;
  không khẳng định chỉ restart sẽ sửa được. Script: ../../scripts/probe_kaggle_mcp.py.
- CLI OAuth hoạt động: kaggle quota trả 30h GPU và 20h TPU còn lại, refresh
  2026-09-12T00:00:00; đây là kết quả CLI, không chứng minh MCP đăng nhập.
- Local GPU RTX 3050 Laptop 4 GB; torch 2.13.0+cu130 báo CUDA available.
- Môi trường nckh có numpy 2.5.2, pyarrow 25.0.0, nbformat 5.11.1.
- Chưa chạy remote notebook, chưa xác nhận allocation T4 ×2 bằng runtime.

Cập nhật sau khi người dùng chỉ định ~/.kaggle làm nguồn token (2026-09-06):
đã đồng bộ .mcp.json, .vscode/mcp.json và .codex/config.toml theo tài khoản CLI
minhtran0601. Codex dùng http_headers_helper đọc accounts/<username>.mcp-token,
không cần export KAGGLE_MCP_AUTHORIZATION. Cấu hình JSON chứa header được đặt mode 0600.

Probe trực tiếp hai token đã lưu (minhtran0601, odixe0502) và OAuth access token
hiện tại đều trả Unauthenticated cho MCP quota, dù header Bearer được gửi đúng.
Không kết luận token hết hạn hay bị thu hồi vì chưa có bằng chứng về nguyên nhân.
Đã cập nhật scripts/probe_kaggle_mcp.py để chọn nguồn kiểm tra rõ ràng và thêm
scripts/sync_kaggle_mcp.py cho lần đổi token/tài khoản tiếp theo.

Thao tác còn cần: cung cấp token MCP hợp lệ trong file của tài khoản đang dùng,
sync lại cấu hình và reload MCP client. Codex IDE hướng dẫn Restart extension;
không mặc định yêu cầu restart toàn bộ VS Code/WSL. Chỉ restart không sửa được lỗi
token bị từ chối trong một direct request mới. Không gửi token trong cuộc trò chuyện.
Kiểm tra lại get_accelerator_quota native sau reload để xác nhận cổng MCP.

## Các cổng kiểm tra trước khi train

1. Người dùng chốt đề xuất A hoặc đưa điều chỉnh.
2. Xác thực MCP account-scoped thành công trong client.
3. Audit input và công bố mức kiểm tra/giới hạn.
4. Dựng và kiểm tra model đúng kiến trúc; gradient, proximal và aggregation tests.
5. Một round nhỏ bằng dữ liệu thật, đủ artifact, mô phỏng crash và next-round resume.
6. Benchmark GPU local; kiểm tra notebook metadata và module được đóng gói.
7. Khi có yêu cầu launch: kiểm tra quota hiện tại và runtime đúng hai T4.

## Kết quả

Chưa có kết quả train thật. Không dùng checkpoint centralized cũ làm kết quả AFPHA.
Các số accuracy ~0,09 trong smoke test local là fixture 60.000 dòng với 6/16 lớp,
chỉ chứng minh đường ống, tuyệt đối không báo cáo như kết quả.
