# Chạy mới toàn bộ E0–E4 trên Kaggle

Notebook đã tích hợp các bản sửa E2/E3. Không cần chạy cell vá lỗi riêng.

## Chuẩn bị một lần

1. Commit và push code hiện tại lên nhánh `main`, gồm `student/`, `KAGGLE.md` và **`lab_day10.ipynb`**. Kaggle clone từ GitHub nên chỉ thấy code đã push.
2. Import lại **`lab_day10.ipynb` mới** vào Kaggle. Bắt đầu session mới để chạy từ đầu.
3. Bật Internet và GPU T4/P100 hoặc GPU lớn hơn. Notebook dùng GPU 0; hai T4 không cộng VRAM.
4. Add Input dataset `lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`.
5. Trong Add-ons → Secrets, cấp notebook quyền dùng `GITHUB_TOKEN` cho repo này, quyền Contents: Read and write. Có thể thêm `HF_TOKEN` quyền đọc để tải model có xác thực; model công khai vẫn tải được khi thiếu token.
6. Giữ **`RESUME_FROM = None`** trong cell setup. Không cần thêm Output của Version cũ.
7. Chọn **Save Version → Save & Run All**.

Notebook được sinh từ `student/make_notebook.py`. Nếu chỉnh generator, chạy `python student/make_notebook.py` rồi commit cả generator và notebook. Preflight kiểm tra hai file khớp nhau.

## Các bước tự động

Setup kiểm tra GitHub write access và dataset, clone repo, cài dependencies, xử lý TorchAO cũ rồi kiểm tra LoRA dense bằng forward/backward thật trong subprocess. Kiểm tra này bắt xung đột backend từng gây lỗi E3 **trước khi chạy E0 hoặc train E1/E2**. Nó kiểm tra đường gắn LoRA vào lớp dense; không thay thế việc train toàn bộ model trên GPU.

Tiếp đó notebook kiểm tra GPU và chạy preflight. VRAM được in thực tế, không còn assertion bắt buộc ≥15 GiB. Log setup nằm trong `report/runs/preflight.log.txt` và `optional_backends.json`.

Các cell thực thi lần lượt:

1. Chuẩn bị split cố định theo recording session và chuyển nhãn YOLO sang JSON tọa độ 0–1000.
2. E0 zero-shot, kiểm tra answer-only loss và cấu hình thí nghiệm.
3. Train E1 (LLM LoRA), E2 (thêm projector), E3 (thêm ViT LoRA).
4. Reload trọng số, inference và chấm val/test cho từng setup bằng scorer gốc.
5. E4 chọn base bằng validation F1, train và chấm hai rank 4/64; alpha = 2 × rank.
6. Tổng hợp bảng kết quả, hình ảnh, log và push `report/` lên nhánh `kaggle-results-<UTC timestamp>`.

Mỗi lượt train có 300 optimizer steps, batch size 1, gradient accumulation 8. E5 không chạy.

LLM dùng NF4; ViT/projector và lm_head không bị lượng tử hóa. Optimizer chỉ nhận tham số trainable FP32 để tránh lỗi GradScaler của E2. Setup audit nằm trong `report/runs/<setup>_setup.json`.

## Kết quả và đưa về local

Cell cuối phải in `Completed. Pull branch ...`. Kiểm tra thêm `report/runs/notebook_stages.json`: mọi stage phải là `completed` hoặc `reused`. Nếu in `PIPELINE FAILED`, pipeline chưa hoàn tất dù Kaggle hiện Completed.

Mặc định `FAIL_VERSION_ON_ERROR=False` để giữ Output nếu một stage lỗi. Các stage sau được bỏ qua; cell cuối vẫn thử push report và log. Lỗi setup, hết thời gian hoặc Kaggle kill session có thể dừng trước cell cuối.

Lấy `report/` từ nhánh kết quả hoặc tải từ Kaggle Output. Dùng `report/RESULTS.md`, `summary.json`, `runs/`, `figures/` để hoàn thiện `REPORT.md` ở local rồi chạy `python scripts/check_submission.py`. Checker vẫn có thể FAIL khi report chưa viết hoặc kết quả chưa đạt rubric.

Trọng số, optimizer checkpoints và predictions nằm trong `/kaggle/working/lab-repo/work/`, không push GitHub. Cache model nằm ở `/tmp/lab-hf-cache`. Giữ Kaggle Output nếu muốn dùng lại trọng số.

## Chỉ khi cần chạy tiếp một lần bị gián đoạn

Phần này không cần cho lượt chạy mới ở trên.

Training lưu checkpoint mỗi 50 optimizer steps, giữ hai checkpoint hoàn chỉnh gần nhất. Checkpoint gồm trainable weights, optimizer, scheduler, GradScaler, RNG và vị trí batch. Khi train xong, giữ trọng số cuối và receipt, dọn checkpoint optimizer tạm.

Trong một Save Version mới, vẫn attach dataset gốc. Thay duy nhất dòng `RESUME_FROM = None` ngay trong cell setup bằng:

```python
import kagglehub
RESUME_FROM = kagglehub.notebook_output_download(
    "minhduong5439/lab-day-10/versions/6"
)
```

Đổi `6` thành Version có Output cần khôi phục. Output phải chứa `lab-repo/work/` và `lab-repo/report/`. Có thể dùng đường dẫn Notebook Output đã Add Input thay cho lệnh tải.

Helper mới `restore_saved_output` khôi phục report theo snapshot, sao lưu report hiện tại khi khác nhau và lưu riêng log cũ trong `work/resume/`. Không ghi đè checkpoint/predictions/receipt xung đột. Receipt chỉ cho phép `REUSE` khi code, config, dependencies, dữ liệu và hash artifact khớp. Chỉ có report trên GitHub thì không đủ khôi phục trọng số.
