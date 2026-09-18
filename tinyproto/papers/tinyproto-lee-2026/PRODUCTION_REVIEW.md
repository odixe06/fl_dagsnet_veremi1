# Review bộ production trước khi push — 2026-09-08

**Kết luận: μ hợp lệ theo giao thức đã chốt; chưa nên coi toàn bộ pipeline là đã nghiệm thu.**
Bộ được kiểm là **7 file thật trong `production/`**, sau khi người dùng cung cấp chúng trong
phiên review. Các bản scratchpad `nbtest/s1,s2` không phải bộ sẽ push. Chưa khởi chạy Kaggle,
đổi tài khoản, tạo dataset hoặc sửa `src/`, generator hay notebook trong review này.

## 1. Phần đã ổn

| Kịch bản | k thắng | μ tuyệt đối đọc được bằng cell config production | F1 validation round 4 | Cách biệt ứng viên đứng thứ hai |
|---|---:|---:|---:|---:|
| 20 client | 0.1 | 7.585696202633669e-07 | 0.6331323192 | 0.0021371156 |
| 50 client | 0.3 | 5.646573163018274e-06 | 0.5162 (làm tròn) | 0.0116910595 |
| 100 client | 0.3 | 1.1264610391688424e-05 | 0.4297490979 | 0.0158370591 |

- Đã chạy **cell `CONFIG_BUILD` thật**, thay mount Kaggle bằng đường dẫn local, giữ cổng μ bật;
  fingerprint lấy từ footer parquet thật và scaler thật, không decode toàn bộ dữ liệu.
  Cấu hình hiệu lực là `mu_kind="absolute"`, không phải placeholder k=1 trong cell đầu.
- Cả 15 ứng viên: μ khớp `k/mean_nij`, đủ bốn mục history; điểm cuối trong summary khớp
  `metrics/round_004.json` của artifact đã pull. Cả 15 config đã pull chấp nhận đúng summary.
  Đây không phải chạy lại verifier trên toàn bộ tensor/checksum; bằng chứng đó ở TESTLOG.
- Ba dataset μ local có **nội dung JSON giống** bản kết quả đã lưu; khác thụt đầu dòng nên
  không giống từng byte. Không có lỗi giá trị μ do sự khác biệt định dạng này.
- Cả 7 production notebook pass validator; đủ 10 module nhúng khớp `src/`, không có output cũ,
  image/GPU metadata hiện đúng. Cell config và launch khớp generator, có `wandb_run=wb`.
- Đúng owner, đúng dataset μ, đúng một notebook session trước, `require_resume=True` ở s2+.
  `rounds=50`, `save_preds_rounds=[50]`, batch 512/512/256 giữ nguyên.
- Fingerprint hiệu lực giữa các session khớp, sau khi giải μ và thêm scaler/artifact version:
  20c `c87c9a7c2c3ee03d`; 50c `8407e5651a74f9f3`; 100c `02446616190d10f6`.
  Đổi `max_hours`/`require_resume` không đổi fingerprint; đổi μ thì có.

| Chuỗi | Owner | max_hours hiện tại | Resume |
|---|---|---|---|
| 20c s1 | minhtran0601 | 11 | không |
| 50c s1 → s2 | odixe0502 | 11 → 7.5 | s2 gắn s1 |
| 100c s1 → s2 → s3 | khanhmay0304 | 11 → 11 → 8 | mỗi sN gắn sN−1 |
| 100c s4 | minhtrit06 | 11 | gắn khanhmay0304/…100client-s3 |

## 2. Mức chắc chắn của μ: giữ giá trị, không khẳng định tối ưu dài hạn

20c: k=0.1 hơn k=0.3 **0.214 điểm phần trăm F1**. Nhưng ngay round 1, khi reg=0 và μ chưa
tham gia loss, năm ứng viên có F1 từ 0.5863877861 tới 0.5902983763, biên độ **0.391 điểm
phần trăm**. Tương ứng 50c/100c là 0.187/0.187 điểm phần trăm, nhỏ hơn khoảng thắng cuối
1.169/1.584 điểm phần trăm. Biên độ round 1 **không phải** khoảng tin cậy cho round 4 và
không chứng minh k=0.1 sai; nó cho thấy không nên diễn giải thứ hạng sát nhau là chắc chắn.

`Worker.__init__` bật `cudnn.benchmark=True`; seed theo client/round được dùng cho `randperm`,
còn dropout dùng RNG của worker. CUDA prototype accumulation dùng `index_add_`. Đây là các
đường cần kiểm tra khi truy nguyên sai khác, **chưa xác định nguyên nhân chính**. PyTorch
không bảo đảm tái lập tuyệt đối chỉ bằng seed; cuDNN benchmark và một số CUDA reduction có
thể ảnh hưởng tính tái lập. [Tài liệu PyTorch 2.10](https://docs.pytorch.org/docs/2.10/notes/randomness.html).

Với k thắng ở 100c, F1 round 2 → 4 giảm 0.432618 → 0.429749 dù reg giảm. Reg nhỏ không đồng
nghĩa F1 đã hội tụ. Quan sát reg bùng lên ở **hai điểm đã thử k=1 và k=3** là rõ, nhưng không
chứng minh mọi k≥1 đều phân kỳ, mọi k≤0.3 đều ổn định 50 round, hay ngưỡng chính xác ở k=1.

**Đề xuất:** giữ nguyên ba μ đã chọn theo giao thức v1. Không chọn lại theo fixed test và
không tự đổi μ giữa các session. Chuyển train từ 98% lên 100% làm số đếm APS tăng khoảng 2%,
nhưng giữ μ tuyệt đối chính là lựa chọn đã được người dùng chốt; không tự chia lại μ.

Nếu muốn tăng độ tin cậy trước khi chi ngân sách 50 round, ưu tiên **kiểm tra lặp lại 20c ở
shortlist 0.03/0.1/0.3 trên cùng validation**, hoặc kéo dài shortlist theo số round ấn định
trước. Đây là thí nghiệm tùy chọn, cần chốt giao thức/ngân sách mới trước khi dùng nó thay
winner v1; không bắt buộc làm lại toàn bộ 15 ứng viên chỉ vì μ có giá trị rất nhỏ.

## 3. Các điểm cần sửa/kiểm tra trước chạy dài

### P1 — W&B vẫn có thể dừng training: nên sửa trước push

`src/driver.py` gọi `wandb_run.log(flat, step=rnd)` ngoài `try/except`, sau commit nhưng trước
ghi `timing_NNN.json`. Chỉ login/init được bảo vệ trong notebook. Diagnostic đã tiêm logger
ném exception vào **chính block này** và exception thoát ra ngoài: một lỗi telemetry có thể
dừng run, dù checkpoint round vừa xong vẫn an toàn. Thêm `wandb_run=wb` không sửa vấn đề này.

**Sửa:** bảo vệ `.log()`, báo lỗi ngắn, vô hiệu telemetry còn lại khi lỗi và vẫn ghi timing;
`finish()` production trong `finally` cũng phải được bảo vệ. **Test:** logger giả lỗi ở round
đầu, pipeline vẫn commit round kế tiếp và có timing/history; nhánh W&B=None vẫn pass. Không
cần W&B thật hoặc GPU cho regression này. Regenerate cả hai bộ notebook sau khi sửa nguồn.

### P2 — Quyền đọc input/checkpoint chéo tài khoản chưa được nghiệm thu

Cả bảy file production đều dùng **dataset μ**, kể cả owner odixe0502. Người dùng xác nhận
ba dataset này **chưa tạo trên Kaggle**: hiện chưa thể push production thành công theo bộ
input dự kiến. S4 còn phụ thuộc output riêng tư của `khanhmay0304/…100client-s3`.
`kernel_sources` đúng chuỗi chỉ chứng minh cấu hình, không cấp quyền đọc.

**Làm trước:** tạo ba dataset μ private khi được phép, share đúng tài khoản; kiểm quyền đọc
VeReMi và μ bằng tài khoản nhận. Với s3→s4, thiết lập quyền xem notebook nguồn cho minhtrit06
và thử gắn/đọc output bằng tài khoản nhận, hoặc bàn giao một dataset checkpoint private có
đủ bundle. Phương án dataset checkpoint còn cần bổ sung generator/validator: hiện
`--resume-from` chỉ khai báo kernel source. Không làm cả hai đường cùng lúc.

Quyền chia sẻ notebook và dataset là riêng biệt; Kaggle có mục Share/Sharing cho notebook.
[Tài liệu Kaggle](https://www.kaggle.com/docs/notebooks).
Chỉ chia sẻ trong các tài khoản của chủ sở hữu đã chỉ định, giữ private; notebook đang nhúng
W&B key theo chính sách hiện tại của dự án.

**Test cần bổ sung:** hai notebook validation nhỏ, s1 ghi checkpoint → publish output → s2
ở tài khoản nhận import và chạy round kế tiếp. `tests/remote_smoke.py` hiện chỉ gọi `run()`
hai lần trong **cùng working directory của một notebook** cho mẫu 20c; nó chưa thử publish,
mount output hay bàn giao tài khoản. Nó cũng chưa gắn/đọc dataset μ. Không gọi test đó là
đã nghiệm thu cả hai đường chỉ vì verifier pass.

### P3 — Validator chưa chặn hết lỗi khi regenerate session

Đã tái hiện bốn notebook sai vẫn được báo `0 problem`:

1. Slug s2 nhưng `require_resume=False`, không gắn session trước → có thể train mới.
2. S2 gắn chính nó làm resume source.
3. 100c gắn notebook 50c làm resume source.
4. Có ba dataset nhưng dataset thứ ba không chứa μ (validator chỉ kiểm `len(ds)>=3`).

**Bộ production hiện tại không mắc bốn lỗi này**, đã đối chiếu riêng bằng diagnostic.
Với `require_resume=True` như các s2+ thật, lệch fingerprint sẽ **dừng trước decode**, không
âm thầm train lại round 1. Rủi ro bắt đầu mới xảy ra khi quên cả cờ resume và attachment.

**Sửa/test:** generator + validator kiểm sN, quan hệ sN−1/cùng scenario/khác slug, resume bắt
buộc ở s2+; kiểm đúng VeReMi + centralized + μ route đã khai báo và tránh gắn hai nguồn μ.
Thêm bốn negative test trên, cùng positive test cho đúng bảy file và các thay đổi max_hours.
Không hardcode chỉ một owner cho resume vì s3→s4 đổi owner là hợp lệ.

### P4 — Kế hoạch round/session cũ quá chắc; bộ max_hours mới cần điều chỉnh theo số đo

`16+16+12 → 6` không phải lịch được code cưỡng chế: cả bảy CFG đều đặt
`rounds_this_session=None`. Driver xét `elapsed + 1.15*worst_round < budget`, và giữ cả timing
round chậm nhất của các session trước. Kết quả cuối mỗi phiên phải đọc marker, không suy từ slug.

Mô phỏng đúng điều kiện với round 2.443 s, setup 217 s, worst khởi đầu 2.500 s: phiên 11 h
chỉ nhận **15 round**, và lịch max_hours **11/11/8/11** cho khoảng **15+15+11+9**. Đây chỉ là
minh họa bằng số đo suy ra, không phải benchmark production. Riêng 44 round + ba setup đã
~30.04 h, chưa warm-up lại, import/export, lưu dự đoán round cuối hay biến động tốc độ.

**Đề xuất:** giữ bốn session như khung dự kiến; điều chỉnh max_hours từ quota thật trước mỗi
push như người dùng đã nêu, chừa thời gian ngoài loop và chấp nhận thêm session nếu thiếu.
Không tự nâng `max_hours` để ép đúng 16 round. 50c max_hours 7.5 của s2 cũng phải xét số round
còn lại thực tế. Ba tổng 10.5+17.6+33.9 ≈ **62 h là phần dự báo**, không phải tổng thời gian đã đo.
Lưu dự đoán 100c round 50 thêm ~2.15 GB; cần đo commit/I/O và RAM của nhánh này trong test nhỏ
và dự trù ngoài timing sweep (sweep không lưu dự đoán round 50).

### P5 — Kiểm chứng AMP/resume số học còn bị diễn giải quá rộng

`try_compile()` đã so sánh trong autocast, nhưng **chỉ logits eval trên batch probe lúc setup**,
không so gradient/update trong train hay toàn bộ các round. Điều kiện AMP lấy
`margin > max(10*delta, 1e-3)` từ chính sai số vừa đo, không chặn delta lớn và không yêu cầu
có hàng decisive. Diagnostic đưa logits tham chiếu `[2,0]`, kết quả sai `[0,1000]`: delta=1000,
decisive=0, disagreements=0 — vẫn qua phần quyết định này. Đây là thiếu sót của cổng kiểm,
**không phải** bằng chứng các sweep thật có delta=1000; log thật cho sai số nhỏ ~0.0005–0.0007.

**Sửa/test:** chốt ngưỡng sai số tuyệt đối/tương đối trước probe, yêu cầu đủ coverage; test cổng
bằng kết quả cố ý sai. Trên T4, probe ngắn với checkpoint đã train: eager/compiled AMP,
gradient/optimizer step có kiểm soát dropout/RNG, folded/unfolded cho cả `proto` và `clf`.
Giữ chính xác weights/BN/optimizer/RNG sau probe. Không áp một ngưỡng fp32 tùy tiện cho fp16.

Một run liên tục và một run resume trên T4 cần được so trực tiếp theo tolerance đã chốt;
verifier pass và việc checkpoint load được không chứng minh quỹ đạo bitwise giống nhau.
Không tự bật deterministic hoặc đổi AMP/compile khi đang chạy production, vì có thể đổi
tốc độ và quỹ đạo. Nếu cần thay đổi, chốt và kiểm trước round 1.

### P6 — Khóa nguồn trước nhiều session

Fingerprint hiện có cấu hình khoa học nhưng **không có digest mã nguồn hoặc image runtime**.
Validator so notebook với `src/` hiện tại; sửa đồng thời src và regenerate giữa run vẫn có
thể pass validator và giữ fingerprint dù thuật toán đã đổi.

**Trước chạy:** giữ một manifest SHA256 của 10 module, generator, runtime digest và μ JSON
cho run production; đối chiếu ở mỗi session cùng với fingerprint. Có thể thêm preflight từ
manifest hoặc kiểm trong quy trình build; không cần thiết kế lại checkpoint để hoàn thành
review này. Chỉ đổi ngân sách/session attachment như đã chốt. Thay thuật toán phải được
đánh giá như một thay đổi run, không tự nối lịch sử cũ.

## 4. Thứ tự thực hiện đề xuất

1. Sửa P1 và các guard P3/P5; thêm regression có lỗi tiêm, regenerate và kiểm đúng bộ production.
2. Lưu manifest nguồn P6. Chốt ngân sách còn lại P4; giữ μ hiện tại trừ khi người dùng chọn
   làm thí nghiệm validation bổ sung ở §2.
3. Khi được phép thao tác Kaggle: tạo/share input, kiểm P2 bằng đúng tài khoản và probe T4
   nhỏ. Kết hợp kiểm số học P5 vào probe, không cần lặp calibration toàn dữ liệu.
4. Chỉ sau các bước trên mới bắt đầu production 50 round; theo dõi finite/applied/skipped,
   reg, timing, compile và commit trong những round đầu. Fixed test dùng để báo cáo, không
   dùng đổi μ/chọn checkpoint tốt nhất. Xác nhận hoàn tất bằng `verify_run.py --require-complete`.

## 5. Bằng chứng review local

`review-logs/production-review-checks.py` tái hiện các kiểm tra và thiếu sót trên; log cùng tên
đuôi `.log`. `review-logs/production-review-regressions.log`: **20 test pass**, peak **620 MiB**.
Diagnostic hoàn tất trên 7 production notebook thật: peak **645 MiB**, exit 0.
Mọi lệnh qua `scripts/run_local_checked.py`, tuần tự, CPU, không Inductor/local CUDA.
Không chạy lại cả suite huấn luyện vì review không sửa mã thực thi; các test nguồn cũ không
thay cho negative test của những lỗi mới. Diagnostic từng cần sửa harness (so byte JSON và
đổi import `proj` sang `src` cho local); đó không phải lỗi sản phẩm.
