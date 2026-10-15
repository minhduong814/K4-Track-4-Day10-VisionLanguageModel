# Chạy E0–E4 trong một Save Version

1. Chạy `python student/make_notebook.py` sau khi chỉnh generator, rồi commit/push **cả `student/make_notebook.py` và `lab_day10.ipynb`** cùng code liên quan lên `main` trước khi chạy. Notebook clone code từ GitHub, không đọc file đang nằm trên máy bạn. Nếu chỉ push generator mà quên notebook đã sinh, preflight sẽ FAIL ở `test_notebook_code_compiles_and_every_section_executes`; lỗi này xảy ra trước training. Cell setup mới lưu đầy đủ output pytest tại `report/runs/preflight.log.txt` và nhắc cách đồng bộ.
2. Import bản mới của `lab_day10.ipynb` vào Kaggle. Bật Internet và GPU T4/P100; code dùng GPU 0. Cell setup in tên GPU, VRAM tổng và còn trống theo GiB; không còn assertion bắt buộc ≥15 GiB. GPU được ghi là 16 GB có thể được CUDA báo dưới 15 GiB. Hai T4 vẫn chạy trên GPU 0, không cộng bộ nhớ hai GPU.
3. Add Input dataset `lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`. Notebook tự tìm thư mục có `images/` và `labels-YOLO/` trong `/kaggle/input`.
4. Trong Add-ons → Secrets, tạo và cấp quyền truy cập `GITHUB_TOKEN`: fine-grained token cho repo này, quyền Contents: Read and write. Có thể thêm Secret **`HF_TOKEN`** (đúng tên, token Hugging Face quyền đọc) và bật quyền truy cập cho notebook. Cell setup export token vào environment trước các subprocess tải model, không in hoặc ghi token ra file. Model công khai vẫn tải được khi không có token; warning `unauthenticated requests` nghĩa là request chưa dùng token, không chứng minh key sai.
5. Chọn **Save Version → Save & Run All**. Không chọn Quick Save. Mặc định E0–E4, 300 optimizer steps mỗi lượt train; batch 1 × accumulation 8 cho mọi setup. E4 có hai lượt rank 4/64, base chọn bằng F1 trên validation. E5 không chạy.

Notebook kiểm tra GPU, Internet, dataset và GitHub write access trước khi tải model. Các mục 2–8 có cell thực thi riêng: xem nhãn, E0, kiểm tra mask loss, cấu hình, train E1/E2/E3, reload/chấm từng setup, train/chấm E4 rank 4/64, tổng hợp và push. Hàm huấn luyện nằm trong `student/pipeline.py`; notebook gọi từng stage qua subprocess để giải phóng GPU khi stage kết thúc. Mỗi stage có log riêng ở `report/runs/<stage>.log.txt`.

Sau pipeline, cell cuối push `report/` vào nhánh riêng `kaggle-results-<UTC timestamp>`. Khi một stage lỗi, các stage tiếp theo được bỏ qua, cell cuối vẫn thử push log/bằng chứng đã tạo và in rõ `PIPELINE FAILED`. Mặc định `FAIL_VERSION_ON_ERROR=False`: notebook kết thúc bình thường để Kaggle lưu Output, nhưng `report/runs/status.json` vẫn ghi `failed`. Trạng thái Completed của Kaggle không có nghĩa thí nghiệm đã hoàn tất; luôn kiểm tra `status.json` và `notebook_stages.json`. Đặt `FAIL_VERSION_ON_ERROR=True` nếu muốn ném lại exception ở cell cuối. Lỗi ở setup/chuẩn bị data hoặc Kaggle kill session vẫn có thể dừng trước cell cuối. Có thể chạy lại setup sau lỗi preflight nếu clone vẫn ở main và chưa sửa file tracked; code fetch/fast-forward thay vì clone đè.

## Giữ kết quả và chạy tiếp sau lỗi

- Bước đã hoàn thành có receipt tại `work/resume/stages/`. Code kiểm tra hash artifact, code pipeline, phiên bản thư viện, config và split dữ liệu trước khi in `REUSE` và bỏ qua bước đó. JSON score đơn lẻ trên GitHub không đủ để tiếp tục train/infer: cần adapter/checkpoint trong Kaggle Output.
- Training lưu checkpoint mỗi **50 optimizer steps** tại `work/adapters/<setup>/resume-step-*/`: trọng số trainable (LoRA + projector), AdamW, scheduler, GradScaler, RNG và thứ tự/vị trí batch. Giữ hai checkpoint hoàn chỉnh gần nhất của lượt train đang dở. Lần sau tự in `RESUME ... step 100/300` và tiếp tục từ checkpoint hợp lệ mới nhất; tối đa mất phần đã chạy sau checkpoint cuối.
- Khi train đủ 300 steps, lưu adapter/projector cuối và run metadata, tạo receipt rồi xoá các checkpoint optimizer tạm để giảm dung lượng Output. E1 đã hoàn tất được giữ bằng `final/`, không phải train lại khi E2 lỗi.

Để dùng lại ở **Save Version mới**:

1. Mở Version trước → Output, xác nhận có `lab-repo/work/` và `lab-repo/report/`. Nếu phiên interactive còn sống, có thể Quick Save với tuỳ chọn lưu Output để giữ file trước khi restart; Quick Save không chạy lại pipeline.
2. Trong notebook mới, **Add Input → Notebook Output** cho phép thêm notebook nhưng giao diện có thể không cho chọn Version cụ thể. Nếu cần đúng Version hôm trước, dùng `kagglehub.notebook_output_download` với handle có `/versions/N` như bên dưới. Vẫn attach road-damage dataset gốc.
3. Sửa biến ở cell setup, ví dụ:

   ```python
   RESUME_FROM = "/kaggle/input/ten-notebook-output/lab-repo"
   ```

   Để lấy **chính xác một Version**, thay dòng `RESUME_FROM = None` trong cell setup bằng đoạn sau (không cần thêm Notebook Output qua giao diện):

   ```python
   import kagglehub
   RESUME_FROM = kagglehub.notebook_output_download(
       "KAGGLE_USERNAME/NOTEBOOK_SLUG/versions/3"
   )
   ```

   Thay username/slug theo URL notebook Kaggle và `3` bằng số Version đã chạy xong E2. Dùng nguyên đường dẫn trả về: hàm restore chấp nhận cả thư mục Output chứa `lab-repo/`. API ghim version được mô tả trong [tài liệu chính thức kagglehub](https://github.com/Kaggle/kagglehub#download-notebook-outputs). Đặt đoạn này ngay tại vị trí cấu hình `RESUME_FROM`, để Save & Run All mới cũng thực thi được; nếu chỉ chạy ở một cell tạm rồi để setup gán lại `None`, resume sẽ không được bật.

   Đường dẫn có thể là thư mục cha chứa `lab-repo/`; xem cây Input để lấy tên chính xác. Code chỉ copy `work/` cần cho resume và `report/`, không lấy code hoặc `.git/` của Output cũ. Nếu log preflight/stage, status hoặc thông tin backend khác phiên hiện tại, bản cũ được lưu riêng ở `work/resume/imported-logs/`, giữ log hiện tại và tiếp tục restore. Nếu trọng số, predictions hoặc evidence xung đột, code vẫn dừng để tránh ghi đè kết quả. Xử lý log nằm ở runtime helper, không đổi fingerprint của E1/E2 đã train.
4. Save & Run All. Sau khi dựng lại split, E0/train/evaluate đã hoàn tất và tương thích sẽ được bỏ qua; bước đang dở sẽ tiếp tục từ checkpoint. Bảng tổng hợp được tạo lại sau khi đủ tất cả evidence.

Nếu còn **cùng phiên interactive**, sau khi sửa lỗi có thể chạy lại cell định nghĩa `run_stage` (đặt lại `pipeline_error`), rồi chạy từ cell train bị lỗi trở đi; files trong `/kaggle/working` vẫn được kiểm tra/reuse. Nếu chỉ có `report/` trên GitHub và checkpoint/Output đã mất thì không thể khôi phục trọng số.

Với lỗi skip list `visual` trước bản sửa `model.visual`, E0/E1 cũ dùng cấu hình lượng tử hoá sai nên không được reuse cho so sánh chuẩn. Cần chạy lại lần đó; cơ chế receipt mới sẽ tránh phải lặp các bước đúng trong những lần lỗi tiếp theo.

Kết quả gồm evidence JSON của scorer gốc, validation scores, cấu hình, loss logs, VRAM/thời gian, summary JSON, bảng `report/RESULTS.md`, ít nhất 6 ảnh có GT/prediction và chú thích. `REPORT.md` còn để bạn tự phân tích số liệu. Submission checker có thể còn FAIL cho báo cáo chưa viết hoặc kết quả model chưa đạt rubric; notebook lưu toàn bộ output checker.

Nếu gặp `Attempting to unscale FP16 gradients`, cập nhật `student/pipeline.py` từ bản mới trước khi chạy lại stage train bị lỗi. Setup mới áp dụng danh sách tham số được train sau PEFT/gradient checkpointing: E1 chỉ LoRA LLM, E2 thêm projector, E3 thêm LoRA ViT; tất cả tham số của optimizer được đưa về FP32, forward vẫn autocast FP16. Log `Trainable groups (all fp32)` và `report/runs/<setup>_setup.json` ghi rõ tên, dtype và số tham số từng nhóm. Base model/ViT ngoài setup được đóng băng; `use_cache=False` được truyền trực tiếp khi train. File audit có trước bước train đầu tiên để phát hiện sai setup sớm.

Nếu GradScaler gặp overflow khi hiệu chỉnh scale ban đầu, nó bỏ qua update và giảm scale; scheduler và bộ đếm 300 bước chỉ tăng khi optimizer thực sự update. Log riêng `report/runs/<setup>_amp_overflows.json` ghi lại các lần này; code dừng nếu có 10 overflow liên tiếp. Loss NaN/Inf vẫn dừng ngay để tránh tiếp tục train sai.

Nếu E3 báo `Found an incompatible version of torchao ... 0.10.0 ... 0.16.0`, đây là backend tuỳ chọn có sẵn trên Kaggle. Lab dùng bitsandbytes, nhưng PEFT vẫn kiểm tra TorchAO khi gắn LoRA vào các lớp dense của ViT. Setup mới tự gỡ TorchAO nếu phiên bản dưới 0.16.0, không nâng cấp torch/PEFT/bitsandbytes. Thông tin được ghi ở `report/runs/optional_backends.json`. Thay đổi này chỉ ở runtime/notebook, giữ nguyên fingerprint pipeline để reuse E0/E1/E2 đúng đã chạy.

Nếu **phiên hiện tại vẫn còn `/kaggle/working/lab-repo`**, chạy ngay cell sau, không cần restart kernel hoặc train lại E1/E2:

```python
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
pipeline_error = None
run_stage("train", "e3")
```

E3 bị lỗi trước khi gắn xong LoRA nên chưa có optimizer step để tiếp tục; chỉ riêng E3 bắt đầu train từ đầu. Sau khi E3 xong, chạy các cell từ mục 7 (evaluate E1/E2/E3), rồi mục 8 (E4/summary/push). Những bước đã có receipt hợp lệ sẽ tự `REUSE`. Nếu đã sang session mới, attach Output cũ và đặt `RESUME_FROM` như phần trên, dùng notebook mới có bước xử lý TorchAO. Giữ cùng các phiên bản torch/transformers/PEFT/bitsandbytes/accelerate và cấu hình batch để cache E1/E2 còn hợp lệ.

Nếu E2 báo `only Tensors of floating point and complex dtype can require gradients`, projector có thể vẫn là trọng số đóng gói 4-bit. Với Transformers 5.19, skip list phải dùng đường dẫn đầy đủ `['model.visual', 'lm_head']`; tên ngắn `visual` không loại được các lớp `model.visual.*`. `load_base()` mới kiểm tra ngay rằng toàn bộ ViT/projector và lm_head không bị lượng tử hóa, trong khi LLM có Params4bit. Kiểm tra này chạy cả ở E0, trước khi mất thời gian train. Audit setup ghi cả cấu hình lượng tử hóa. Không ép `.float()` trực tiếp lên packed 4-bit weights để thay thế việc load đúng. Vì sửa này thay cấu hình load model, chạy lại toàn bộ E0–E4 trong một Save & Run All mới để kết quả so sánh nhất quán; không ghép evidence E0/E1 từ cấu hình load cũ với E2/E3 từ cấu hình mới.

Adapter, projector checkpoints và predictions nằm trong `/kaggle/working/lab-repo/work/`, được giữ trong Kaggle Output và không push GitHub. Repo clone có metadata split trong `data/`; metadata cũng được sao chép sang `report/runs/splits.json`. Cache tải model đặt ở `/tmp` để giảm dung lượng Output. Nếu Kaggle kill session vì hết thời gian/VRAM, cell cleanup có thể không chạy; không có đảm bảo một lần chạy sẽ thành công trước khi kiểm 

``

Tải dataset ở local bằng công cụ gốc nếu cần chạy submission checker. Dùng `report/RESULTS.md`, `report/summary.json`, `report/runs/`, `report/figures/` để hoàn thiện `REPORT.md`, rồi chạy `python scripts/check_submission.py`.

Nếu push thất bại, tải thư mục `report/` trong Kaggle Output về local. Đừng copy `.git/`, token, dataset hoặc checkpoints vào commit. Nếu chỉnh cấu trúc notebook, dùng `python student/make_notebook.py` để tạo lại; không chạy `scripts/make_notebook.py` vì đó là generator khung gốc.
