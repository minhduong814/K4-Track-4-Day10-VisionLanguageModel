# Chạy E0–E4 trong một Save Version

1. Đưa các thay đổi local này lên `main` của repo trước khi chạy. Notebook clone code từ GitHub, không đọc file đang nằm trên máy bạn.
2. Import bản mới của `lab_day10.ipynb` vào Kaggle. Bật Internet và GPU T4/P100; code dùng GPU 0. Cell setup in tên GPU, VRAM tổng và còn trống theo GiB; không còn assertion bắt buộc ≥15 GiB. GPU được ghi là 16 GB có thể được CUDA báo dưới 15 GiB. Hai T4 vẫn chạy trên GPU 0, không cộng bộ nhớ hai GPU.
3. Add Input dataset `lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`. Notebook tự tìm thư mục có `images/` và `labels-YOLO/` trong `/kaggle/input`.
4. Trong Add-ons → Secrets, tạo và cấp quyền truy cập `GITHUB_TOKEN`: fine-grained token cho repo này, quyền Contents: Read and write. Không ghi token vào cell. Model công khai nên không yêu cầu HF token.
5. Chọn **Save Version → Save & Run All**. Không chọn Quick Save. Mặc định E0–E4, 300 optimizer steps mỗi lượt train; batch 1 × accumulation 8 cho mọi setup. E4 có hai lượt rank 4/64, base chọn bằng F1 trên validation. E5 không chạy.

Notebook kiểm tra GPU, Internet, dataset và GitHub write access trước khi tải model. Các mục 2–8 có cell thực thi riêng: xem nhãn, E0, kiểm tra mask loss, cấu hình, train E1/E2/E3, reload/chấm từng setup, train/chấm E4 rank 4/64, tổng hợp và push. Hàm huấn luyện nằm trong `student/pipeline.py`; notebook gọi từng stage qua subprocess để giải phóng GPU khi stage kết thúc. Mỗi stage có log riêng ở `report/runs/<stage>.log.txt`.

Sau pipeline, cell cuối push `report/` vào nhánh riêng `kaggle-results-<UTC timestamp>`. Khi một stage lỗi, các stage tiếp theo được bỏ qua, cell cuối vẫn thử push log/bằng chứng đã tạo rồi báo lại lỗi gốc; kiểm tra `report/runs/status.json` và `notebook_stages.json` trước khi dùng kết quả. Lỗi ở cell setup hoặc chuẩn bị data sẽ dừng trước các stage/push. Có thể chạy lại cell setup sau lỗi preflight nếu clone vẫn ở main và chưa sửa file tracked; code fetch/fast-forward thay vì clone đè.

Kết quả gồm evidence JSON của scorer gốc, validation scores, cấu hình, loss logs, VRAM/thời gian, summary JSON, bảng `report/RESULTS.md`, ít nhất 6 ảnh có GT/prediction và chú thích. `REPORT.md` còn để bạn tự phân tích số liệu. Submission checker có thể còn FAIL cho báo cáo chưa viết hoặc kết quả model chưa đạt rubric; notebook lưu toàn bộ output checker.

Adapter, projector checkpoints và predictions nằm trong `/kaggle/working/lab-repo/work/`, được giữ trong Kaggle Output và không push GitHub. Repo clone có metadata split trong `data/`; metadata cũng được sao chép sang `report/runs/splits.json`. Cache tải model đặt ở `/tmp` để giảm dung lượng Output. Nếu Kaggle kill session vì hết thời gian/VRAM, cell cleanup có thể không chạy; không có đảm bảo một lần chạy sẽ thành công trước khi kiểm tra trên GPU thật.

Sau khi chạy, lấy tên branch trong output và ở local:

```powershell
git fetch origin
git merge origin/kaggle-results-YYYYMMDD-HHMMSS
```

Tải dataset ở local bằng công cụ gốc nếu cần chạy submission checker. Dùng `report/RESULTS.md`, `report/summary.json`, `report/runs/`, `report/figures/` để hoàn thiện `REPORT.md`, rồi chạy `python scripts/check_submission.py`.

Nếu push thất bại, tải thư mục `report/` trong Kaggle Output về local. Đừng copy `.git/`, token, dataset hoặc checkpoints vào commit. Nếu chỉnh cấu trúc notebook, dùng `python student/make_notebook.py` để tạo lại; không chạy `scripts/make_notebook.py` vì đó là generator khung gốc.
