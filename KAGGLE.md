# Chạy E0–E4 trong một Save Version

1. Đưa các thay đổi local này lên `main` của repo trước khi chạy. Notebook clone code từ GitHub, không đọc file đang nằm trên máy bạn.
2. Import `lab_day10.ipynb` vào Kaggle. Bật Internet và GPU T4/P100; code dùng GPU 0. Chọn GPU có ít nhất 16 GB VRAM.
3. Add Input dataset `lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`. Notebook tự tìm thư mục có `images/` và `labels-YOLO/` trong `/kaggle/input`.
4. Trong Add-ons → Secrets, tạo và cấp quyền truy cập `GITHUB_TOKEN`: fine-grained token cho repo này, quyền Contents: Read and write. Không ghi token vào cell. Model công khai nên không yêu cầu HF token.
5. Chọn **Save Version → Save & Run All**. Không chọn Quick Save. Mặc định E0–E4, 300 optimizer steps mỗi lượt train; batch 1 × accumulation 8 cho mọi setup. E4 có hai lượt rank 4/64, base chọn bằng F1 trên validation. E5 không chạy.

Notebook kiểm tra GPU, Internet, dataset và GitHub write access trước khi tải model. Sau pipeline, nó push `report/` vào nhánh riêng `kaggle-results-<UTC timestamp>`. Khi subprocess pipeline lỗi, nó vẫn thử push log và các bằng chứng đã tạo, sau đó báo lỗi notebook; kiểm tra `report/runs/status.json` trước khi dùng kết quả. Lỗi ở cell setup hoặc chuẩn bị data sẽ dừng trước cell pipeline/push.

Kết quả gồm evidence JSON của scorer gốc, validation scores, cấu hình, loss logs, VRAM/thời gian, summary JSON, bảng `report/RESULTS.md`, ít nhất 6 ảnh có GT/prediction và chú thích. `REPORT.md` còn để bạn tự phân tích số liệu. Submission checker có thể còn FAIL cho báo cáo chưa viết hoặc kết quả model chưa đạt rubric; notebook lưu toàn bộ output checker.

Adapter, projector checkpoints và predictions nằm trong `/kaggle/working/lab-repo/work/`, được giữ trong Kaggle Output và không push GitHub. Repo clone có metadata split trong `data/`; metadata cũng được sao chép sang `report/runs/splits.json`. Cache tải model đặt ở `/tmp` để giảm dung lượng Output. Nếu Kaggle kill session vì hết thời gian/VRAM, cell cleanup có thể không chạy; không có đảm bảo một lần chạy sẽ thành công trước khi kiểm tra trên GPU thật.

Sau khi chạy, lấy tên branch trong output và ở local:

```powershell
git fetch origin
git merge origin/kaggle-results-YYYYMMDD-HHMMSS
```

Tải dataset ở local bằng công cụ gốc nếu cần chạy submission checker. Dùng `report/RESULTS.md`, `report/summary.json`, `report/runs/`, `report/figures/` để hoàn thiện `REPORT.md`, rồi chạy `python scripts/check_submission.py`.

Nếu push thất bại, tải thư mục `report/` trong Kaggle Output về local. Đừng copy `.git/`, token, dataset hoặc checkpoints vào commit. Nếu chỉnh tiếp notebook, không chạy `scripts/make_notebook.py` vì đó là generator khung gốc.
