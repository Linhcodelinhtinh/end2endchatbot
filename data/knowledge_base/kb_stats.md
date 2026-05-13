# Báo cáo thống kê Knowledge Base (KB)

## 1) Phạm vi và đầu ra

Tài liệu này tổng hợp chất lượng KB sau pipeline:

1. Thu thập dữ liệu thô: `scripts/collect_kb.py`
2. Làm sạch dữ liệu: `scripts/clean_kb.py`
3. Chunk phục vụ retrieval: `scripts/build_chunks.py`

Các file đầu ra chính:

- `data/knowledge_base/source_manifest.csv`
- `data/knowledge_base/processed/docs.jsonl`
- `data/knowledge_base/processed/chunks.jsonl`

## 2) Thống kê thu thập dữ liệu (manifest)

- Tổng URL đã xử lý: **1006**
- Thành công (`ok`): **948**
- Lỗi (`exception` + `http_error`): **58**
- Tỉ lệ thành công: **94.23%**

Phân rã trạng thái:

- `ok`: 948
- `exception`: 24
- `http_error`: 34

### Theo source_id

- `vnu-main-history`: 317/329 (**96.35%**)
- `vnu-wikipedia`: 1/1 (**100%**)
- `vnu-admission-portal`: 0/2 (**0%**)
- `uet-admission`: 214/216 (**99.07%**)
- `ussh-admission`: 85/109 (**77.98%**)
- `ulis-admission`: 100/110 (**90.91%**)
- `hus-admission`: 0/1 (**0%**)
- `vnu-academic-regulations`: 119/122 (**97.54%**)
- `vnu-general-programs`: 6/6 (**100%**)
- `vnu-international-programs`: 106/110 (**96.36%**)

### Domain xuất hiện nhiều nhất (top 10)

- `vnu.edu.vn`: 303
- `uet.vnu.edu.vn`: 201
- `ussh.vnu.edu.vn`: 111
- `is.vnu.edu.vn`: 108
- `vieclam.uet.vnu.edu.vn`: 90
- `ulis.vnu.edu.vn`: 69
- `tuyensinh.uet.vnu.edu.vn`: 15
- `student.ulis.vnu.edu.vn`: 10
- `vju.ac.vn`: 7
- `lic.vnu.edu.vn`: 3

### Loại content thu thập

- `text/html`: 979
- `application/pdf`: 3

### Lỗi phổ biến nhất (top)

- `http_429`: 23 (rate limit)
- `http_404`: 8
- `http_500`: 2
- timeout/SSL/brotli decode: rải rác

Nhận định: lỗi lớn nhất là giới hạn truy cập (429) và lỗi hạ tầng từng domain, không phải lỗi pipeline.

## 3) Thống kê dữ liệu sạch (`docs.jsonl`)

- Tổng documents sạch giữ lại: **734**
- Ngôn ngữ:
  - `vi`: 716
  - `en`: 18
- Loại tài liệu:
  - `html`: 731
  - `pdf`: 3
- Có trường `published_at` (heuristic): 377 docs

### Phân bố theo category

- `history`: 232
- `admission`: 296
- `regulation`: 107
- `program`: 99

### Phân bố theo source_id

- `vnu-main-history`: 231
- `vnu-wikipedia`: 1
- `uet-admission`: 175
- `ussh-admission`: 43
- `ulis-admission`: 78
- `vnu-academic-regulations`: 107
- `vnu-international-programs`: 99

### Đặc trưng độ dài document

- Số từ trung bình: **1205.45**
- Median số từ: **548.5**
- Min/Max số từ: **26 / 32981**
- Số ký tự trung bình: **5860.01**
- Median số ký tự: **2728.5**
- Min/Max số ký tự: **124 / 181195**

## 4) Thống kê chunk (`chunks.jsonl`)

- Tổng chunks: **5840**
- Trung bình chunks/doc: **7.96**

### Phân bố theo category

- `history`: 2161
- `admission`: 1814
- `regulation`: 487
- `program`: 1378

### Top source đóng góp chunks

- `vnu-main-history`: 1949
- `vnu-international-programs`: 1378
- `uet-admission`: 930
- `ulis-admission`: 630
- `vnu-academic-regulations`: 487
- `ussh-admission`: 254
- `vnu-wikipedia`: 212

### Chất lượng chunk

- `section_label`:
  - `body`: 5733
  - `heading`: 107
- Số từ/chunk:
  - Trung bình: **197.28**
  - Median: **209**
  - Min/Max: **32 / 220**
- Số câu/chunk:
  - Trung bình: **6.75**
  - Median: **5**
  - Min/Max: **1 / 67**

Nhận định: phân bố chunk ổn, độ dài tập trung gần cấu hình mục tiêu (220 từ), phù hợp cho retrieval.

## 5) Đánh giá chất lượng tổng quan

### Điểm tốt

- Tỉ lệ crawl thành công cao (**94.23%**).
- KB có quy mô đủ lớn cho RAG:
  - 734 docs sạch
  - 5840 chunks
- Phân bố category tương đối cân bằng, không bị lệch hoàn toàn về 1 nhóm.
- Pipeline có khả năng tái lập rõ ràng (collect -> clean -> chunk).

### Rủi ro và hạn chế còn lại

- Dữ liệu PDF còn ít (3 file), nên độ phủ văn bản quy chế dạng PDF chưa cao.
- Một số source có tỉ lệ thành công thấp (`vnu-admission-portal`, `hus-admission`).
- Domain phụ phát sinh (VD: `vieclam.uet.vnu.edu.vn`) có thể tạo nhiễu nếu không lọc theo mục tiêu QA.

## 6) Kiến nghị nâng cấp vòng tiếp theo

1. Bổ sung seed URL PDF trực tiếp cho nhóm `regulation` và `program`.
2. Thêm `include_url_keywords` cho từng source để giảm crawl lan sang domain phụ không cần thiết.
3. Tạo vòng crawl incremental với `--resume-from-manifest` để mở rộng KB mà không crawl lại toàn bộ.
4. Làm manual QA:
   - random 30 docs (đúng domain, đúng category, text sạch)
   - random 30 chunks (đúng ngữ nghĩa, không mất ngữ cảnh nặng)

## 7) Kết luận

Phần **“Chuẩn bị dữ liệu thô / Biên soạn tài nguyên tri thức”** đã đạt mức tốt để chuyển sang annotation và RAG training/inference.  
KB hiện tại đã sẵn sàng sử dụng cho các bước tiếp theo của assignment.
