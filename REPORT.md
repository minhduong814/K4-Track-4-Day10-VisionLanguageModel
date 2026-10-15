Họ tên: Nguyễn Minh Dương

`dataset_version`: `d10-21e4070301`

## Bảng tổng hợp

Các kết quả dưới đây lấy từ `report/evidence/*.json`. E4 là ablation rank trên
base E3; alpha = 2 x rank.

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | zero-shot | 0 | N/A | N/A | 0.0458 | 0.0312-0.0628 | 2.2379 | 0.0984 | 0.0028 | 0.0985 |
| E1 | LoRA LLM | 33,030,144 | 5.0458 | 47.01 | 0.2799 | 0.2434-0.3170 | 1.1093 | 0.2073 | 0.2082 | 0.5159 |
| E2 | E1 + projector | 142,135,296 | 6.7057 | 48.43 | 0.3533 | 0.3110-0.3934 | 1.0032 | 0.1914 | 0.2998 | 0.5978 |
| E3 | E2 + LoRA ViT | 148,426,752 | 6.8406 | 58.22 | 0.4111 | 0.3684-0.4542 | 0.7846 | 0.2365 | 0.3502 | 0.6549 |
| E4 rank 4 | E3, rank=4, alpha=8 | 118,935,552 | 6.4011 | 58.02 | 0.4152 | 0.3735-0.4582 | 0.8746 | 0.2755 | 0.3266 | 0.6920 |
| E4 rank 64 | E3, rank=64, alpha=128 | 266,391,552 | 8.7926 | 60.03 | 0.4269 | 0.3848-0.4680 | 0.8746 | 0.2949 | 0.3515 | 0.6831 |

## 10 câu hỏi

### Câu 1 — Kiến trúc

Qwen3-VL-4B có LLM khoảng 4.0B tham số, ViT khoảng 415,347,712 tham số
và projector khoảng 109,105,152 tham số. E1 chỉ cập nhật LoRA trên các
projection attention và MLP của LLM (33,030,144 tham số), còn ViT và
projector đóng băng. E2 giữ LoRA LLM và mở toàn bộ projector ở FP32, thành
142,135,296 tham số trainable. E3 thêm LoRA vào attention/MLP của ViT
(6,291,456 tham số LoRA), thành 148,426,752 tham số trainable.

### Câu 2 — E0

`parse_valid_rate` của E0 là `0.5627` (175/311 ảnh hợp lệ theo tỉ lệ đã
chấm). F1 micro chỉ `0.0458`; precision `0.0413` và recall `0.0515`.
Hai lỗi chính là output không parse được thành JSON list đúng schema và
output có box/lớp sai, thường kèm box bịa hoặc sai thang tọa độ. Ví dụ dạng
output cần loại bỏ khi decode là phần giải thích hoặc fenced JSON như

```json
[{"label":"crack","bbox_2d":[...]}]
```
thay vì chỉ trả về JSON list. Sau finetune, parse_valid_rate tăng lên
0.9968 ở E1 và 1.0 ở E2/E3.

### Câu 3 — Thang 0-1000

Ground truth và output đều dùng tọa độ nguyên trên thang 0-1000. Nếu một
bên dùng pixel 640x360 mà bên kia vẫn dùng 0-1000, cùng một box sẽ bị lệch
thang (đặc biệt x và y có hệ số khác nhau), làm IoU thấp và box không được
match; F1 vì vậy giảm mạnh dù hình dạng box có thể đúng. Mình kiểm chứng
bằng bước chuyển YOLO chuẩn hóa sang 0-1000, sau đó dùng scorer gốc trên
split test; tất cả evidence đều ghi `coord` theo quy ước 0-1000 và
`score_version=1`.

### Câu 4 — Mask loss

Chỉ token câu trả lời được tính loss vì mục tiêu là học sinh JSON grounding,
không phải học lại prompt, chat template hay token ảnh. Các token prompt và
vision bị đặt nhãn `-100`, nên không đóng góp vào cross-entropy. Nếu tính
loss trên prompt/vision, gradient bị chi phối bởi việc dự đoán lại nội
dung đầu vào và các token cố định, làm tín hiệu cho JSON yếu hơn và dễ
khiến model học thuộc template thay vì dự đoán box. Kiểm tra thực tế dùng
220 image tokens cho ảnh 640x360 và shape `labels=[1,552]`.

### Câu 5 — Projector và lượng tử hóa

Projector nhận đặc trưng ViT rồi đưa về không gian biểu diễn của LLM; E2
muốn cập nhật đầy đủ các lớp `visual.merger` và
`visual.deepstack_merger_list`. Nếu các lớp này bị lượng tử hóa 4-bit,
trọng số trainable không còn là tensor dense phù hợp cho cập nhật đầy đủ,
đồng thời mất độ chính xác ở phép biến đổi multimodal. Vì vậy setup dùng
NF4 cho LLM nhưng giữ vision/projector và `lm_head` không lượng tử hóa.
Projector được đưa về FP32 để gradient và cập nhật ổn định hơn, đặc biệt
với khoảng 109M tham số. Preflight forward/backward LoRA dense của E3 đã
pass trước khi train.

### Câu 6 — Split theo phiên quay

Các frame gần nhau trong cùng video rất giống nhau. Chia ngẫu nhiên có thể
đưa các frame gần như trùng vào cả train và test, làm F1 cao giả vì test
không còn độc lập theo cảnh quay. Split theo session cho kết quả cố định:

| Split | Ảnh | pothole | crack | manhole |
|---|---:|---:|---:|---:|
| train | 1,422 | 1,074 | 1,750 | 691 |
| val | 276 | 66 | 470 | 103 |
| test | 311 | 121 | 299 | 163 |

### Câu 7 — So sánh E1/E2/E3

Mở projector có lợi rõ rệt: F1 tăng từ `0.2799` ở E1 lên `0.3533` ở E2,
trong khi VRAM tăng từ `5.0458` lên `6.7057 GB` và thời gian chỉ tăng từ
`47.01` lên `48.43` phút. Thêm ViT LoRA tiếp tục tăng F1 lên `0.4111`,
giảm FP/ảnh từ `1.0032` xuống `0.7846`, nhưng thời gian tăng lên
`58.22` phút và VRAM lên `6.8406 GB`. Khoảng tin cậy E2 và E3 lần lượt là
`0.3110-0.3934` và `0.3684-0.4542`, chỉ chồng lấn rất ít; trên lần chạy
này, mở projector và thêm ViT LoRA đều đáng giá, đặc biệt E3.

### Câu 8 — Độ rộng khoảng tin cậy

Với 311 ảnh test, CI95 của F1 vẫn rộng khoảng 0.06-0.09:
E1 `0.2434-0.3170`, E2 `0.3110-0.3934`, E3 `0.3684-0.4542`.
Vì các khoảng có phần chồng lấn, không nên khẳng định chênh lệch nhỏ là
có ý nghĩa thống kê. Có thể kết luận thận trọng rằng E3 tốt hơn rõ ràng
về điểm quan sát và FP/ảnh so với E1/E2 trong lần chạy này, nhưng cần
nhiều seed hoặc test lớn hơn để kết luận chắc chắn về các khác biệt nhỏ.

### Câu 9 — Phân tích lỗi theo lớp

Lớp kém nhất là `pothole` trong các setup đã train: F1 là `0.2073`,
`0.1914`, `0.2365` lần lượt ở E1, E2, E3; trong E3 vẫn thấp hơn crack
(`0.3502`) và manhole (`0.6549`). Test chỉ có 121 box pothole, ít hơn
crack (299), và pothole thường có box nhỏ hoặc hình dạng/texture biến
đổi, nên khó định vị hơn. Manhole có F1 cao nhất, có thể vì hình dạng và
ngữ nghĩa ổn định hơn. Hình minh họa ground truth và các dự đoán E3 nằm
trong [`report/figures/`](report/figures/), gồm
[`train_ground_truth.png`](report/figures/train_ground_truth.png) và
[`e3_1.png`](report/figures/e3_1.png).

### Câu 10 — E4

E4 chọn base bằng validation F1: E1 `0.2015`, E2 `0.2860`, E3 `0.3680`,
nên chọn E3. Biến duy nhất được thay là LoRA rank, với alpha bằng 2 lần
rank. Rank 4 đạt test F1 `0.4152` và rank 64 đạt `0.4269`; rank 64 dùng
266,391,552 tham số, 8.7926 GB VRAM và 60.03 phút, so với rank 4 dùng
118,935,552 tham số, 6.4011 GB và 58.02 phút. Rank 64 tốt hơn 0.0117 F1,
nhưng CI hai kết quả chồng lấn đáng kể (`0.3735-0.4582` và
`0.3848-0.4680`), nên kết luận rank 64 tốt hơn chỉ là tín hiệu trong
single run, chưa đủ chắc chắn.

## Nhìn lại

Lỗi vận hành còn gặp là push kết quả từ Kaggle lên GitHub thất bại do
quyền/token, nhưng các stage E0-E4 và evidence vẫn hoàn tất trong Kaggle
Output. Checker cũng cho thấy báo cáo cần được điền riêng sau khi chạy
pipeline; việc pipeline hoàn tất không tự động thay thế các câu trả lời
phân tích. Kết quả cho thấy parse hợp lệ gần như hoàn hảo sau finetune,
nhưng F1 vẫn bị giới hạn bởi box nhỏ, mất cân bằng lớp và kích thước test
chỉ 311 ảnh.
