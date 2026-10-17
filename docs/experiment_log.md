# Nhật ký thí nghiệm

## Việc cần chạy sau (đã bỏ qua khi viết code, chạy khi có máy phù hợp)

- [ ] Kiểm tra E0 ở kích thước thật (lưới 256/512, 512 kênh) trên server: dựng ma trận `fan_a120` và
      `fan_full` rồi chạy kiểm tra, ghi lại thời gian dựng, bộ nhớ, ||A||, PSNR của FBP 360°:
      `lact ops build -g fan_a120 fan_full`
      `lact ops check -g fan_full fan_a120`
      (ma trận 360° lưới 512 khoảng 320 triệu phần tử khác 0, ~2,5 GB; cần RAM >= 16 GB).
- [ ] Trên GPU: xác nhận nhân ma trận thưa CSR chỉ số int32 chạy trên CUDA (code tự chuyển sang int64 nếu lỗi),
      đo thời gian một phép chiếu xuôi/ngược ở 120°.
- [ ] Chạy `make pilot DEV=cuda` (Cổng 1) để ước lượng GPU-giờ và quyết định K = 14 hay 7.
- [ ] Chạy `make test` và `make smoke` một lần trên máy Mac (conda-forge astra-toolbox bản CPU) để chắc môi trường cài đúng.
- [ ] **Sau refactor v0.2 (kiến trúc phân tầng, CLI `lact`)**: chạy lại toàn bộ `pytest` và `make smoke`. Khi refactor
      chỉ mới kiểm tra tĩnh (ruff, mypy, pyflakes, lint-imports, `lact --help`, dựng 6 mô hình khớp số tham số,
      khai triển lưới E2/E3); chưa chạy test số học và pipeline đầy đủ. Nếu có sai khác, so với bản v0.1
      (commit trước refactor) theo bảng ánh xạ trong `docs/ARCHITECTURE.md`.

- [ ] **Server**: chạy `lact pipeline run --device cuda --until prepare`, ghi thời gian dựng ma trận và RAM đỉnh.
      Nếu ma trận `fan_full` lưới 512 thiếu RAM thì cần cấp thêm RAM trước khi chạy lại.
- [ ] Sau khi có dữ liệu: `lact train pilot --device cuda` trên GPU định dùng để ước lượng GPU-giờ.

Kiểm tra trên dữ liệu Mayo thật (2026-09-28, máy CPU): tải `full_3mm.zip` 716 MB, 10 bệnh nhân, 2.378 lát;
`data inspect` 22 s, `data preprocess` 70 s, `images.h5` 1,6 GB; μ_ref = 0,0379 (~974 HU).

| Ngày | Thí nghiệm | Máy / GPU | Commit | Việc đã chạy | Kết quả / vấn đề |
| --- | --- | --- | --- | --- | --- |
| | E0 | | | `make check` | |
| | E1 | | | | |
| | Cổng 1 | | | `make pilot DEV=cuda` | giây/bước, GPU-giờ ước tính, quyết định K |
| | E2 | | | `make e2 select-lr` | lr từng kiến trúc |
| | E3 | | | `make e3` | |
