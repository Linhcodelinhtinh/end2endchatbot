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

- Tổng URL đã xử lý: **3083**
- Thành công (`ok`): **2514**
- Không thành công (`exception` + `http_error` + `ignored_non_content`): **569**
- Tỉ lệ thành công: **81.54%**

Phân rã trạng thái:

- `ok`: 2514
- `exception`: 24
- `http_error`: 392
- `ignored_non_content`: 153

### Theo source_id

- `vnu-main-history`: 1229/1481 (**82.98%**)
- `vnu-wikipedia`: 1/1 (**100%**)
- `vnu-admission-portal`: 175/188 (**93.09%**)
- `uet-admission`: 184/185 (**99.46%**)
- `ussh-admission`: 157/370 (**42.43%**)
- `ulis-admission`: 253/275 (**92.00%**)
- `hus-admission`: 0/1 (**0%**)
- `vnu-academic-regulations`: 354/392 (**90.31%**)
- `vnu-general-programs`: 161/189 (**85.19%**)
- `vnu-international-programs`: 0/1 (**0%**)

### Domain xuất hiện nhiều nhất (top 10)

- `ussh.vnu.edu.vn`: 585
- `vnu.edu.vn`: 435
- `cdnportal.vnu.edu.vn`: 426
- `uet.vnu.edu.vn`: 194
- `is.vnu.edu.vn`: 170
- `education.vnu.edu.vn`: 140
- `ulis.vnu.edu.vn`: 134
- `tuyensinh.vnu.edu.vn`: 128
- `vieclam.uet.vnu.edu.vn`: 118
- `ivides.vnu.edu.vn`: 106

### Loại content thu thập thành công (`ok`)

- `html`: **1895**
- `pdf`: **619**

Nhận định: độ phủ PDF đã tăng mạnh so với vòng crawl cũ, cải thiện tốt cho nhóm tài liệu quy định/chương trình.

## 3) Thống kê dữ liệu sạch (`docs.jsonl`)

- Tổng documents sạch giữ lại: **1705**
- Ngôn ngữ:
  - `vi`: 1667
  - `en`: 38
- Loại tài liệu:
  - `html`: 1295
  - `pdf`: 410

### Phân bố theo category

- `history`: 794
- `admission`: 520
- `regulation`: 272
- `program`: 119

### Đặc trưng độ dài document

- Tổng số từ: **2,990,544**
- Số từ trung bình: **1753.98**
- Tổng số ký tự: **14,382,974**
- Số ký tự trung bình: **8435.76**

## 4) Thống kê chunk (`chunks.jsonl`)

- Tổng chunks: **19,092**
- Trung bình chunks/doc: **11.20**

### Phân bố theo category

- `history`: 11,789
- `admission`: 4,419
- `regulation`: 2,050
- `program`: 834

### Phân bố theo ngôn ngữ

- `vi`: 18,794
- `en`: 298

### Độ dài chunk

- Tổng số từ chunks: **3,841,877**
- Số từ/chunk trung bình: **201.23**

Nhận định: độ dài chunk bám sát mục tiêu cấu hình (khoảng 220 từ), phù hợp cho retrieval.

## 5) Đánh giá chất lượng tổng quan

### Điểm tốt

- Quy mô KB đã tăng rõ rệt:
  - 1705 docs sạch
  - 19,092 chunks
- PDF đã có độ phủ tốt hơn nhiều (410 docs PDF sạch).
- 4 nhóm category (`history`, `admission`, `regulation`, `program`) đều đã có dữ liệu.
- Pipeline có thể chạy incremental với `--resume-from-manifest`.

### Rủi ro và hạn chế còn lại

- Tỉ lệ lỗi HTTP còn cao ở một số domain/source (đặc biệt `ussh-admission`).
- Có phát sinh crawl sang domain phụ, cần tiếp tục siết bộ lọc theo mục tiêu QA.
- Một phần PDF từ nguồn redirect có thể là trang lỗi HTML giả PDF (đã có xử lý lọc, nhưng nên tiếp tục giám sát).

## 6) Kiến nghị nâng cấp vòng tiếp theo

1. Siết `include_url_keywords` cho `ussh-admission` và các source có nhiều URL nhiễu.
2. Ưu tiên thêm seed trực tiếp cho nhóm `program` để cân bằng với `history`.
3. Duy trì crawl incremental theo đợt nhỏ (`--resume-from-manifest`) và theo dõi tỷ lệ `http_error`.
4. Làm manual QA:
   - random 30 docs/category
   - random 50 chunks ở các nhóm `regulation` và `program`

## 7) Kết luận

KB hiện tại đã vượt mức "đủ dùng" cho RAG và có độ phủ tài liệu tốt hơn đáng kể so với phiên bản trước.  
Bước tiếp theo nên tập trung vào cân bằng category và kiểm soát nhiễu để tăng chất lượng retrieval/answering.
