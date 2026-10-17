# Kế hoạch phân tích (đóng băng trước khi mở tập kiểm thử)

Ngày đóng băng: ____ (trùng thời điểm tạo `outputs/configs_frozen.lock`)

## Giả thuyết

- H1: ResNet và ConvNeXt có PSNR cao hơn LEARN–CNN ở cả 90° và 120°.
- H2: Khi khoảng quét giảm từ 120° xuống 90°, ViT giảm chất lượng nhiều hơn các mô hình tích chập.
- H3: Ở khoảng 2M tham số, thứ tự PSNR là ConvNeXt ≥ ResNet > ViT.
- H4: Phần lớn sai số còn lại nằm ở vùng tần số không được đo.

## Chỉ số

- Chính: PSNR toàn ảnh trên tập kiểm thử (L310, L506), data_range = 1.
- Phụ: SSIM, RMSE, RMSE (HU), PSNR vùng cơ thể, sai số chiếu tương đối.
- Tổng hợp: lát → trung bình bệnh nhân → trung bình hai bệnh nhân; trung bình ± độ lệch chuẩn qua 3 seed.

## Họ so sánh đã đăng ký (Holm trên cả họ)

| Mã | So sánh | Thí nghiệm |
| --- | --- | --- |
| RQ1-a/b | ResNet vs CNN, ConvNeXt vs CNN ở 120° | E3 |
| RQ1-c/d | ResNet vs CNN, ConvNeXt vs CNN ở 90° | E3 |
| RQ2-a/b/c | (CNN − ViT), (ResNet − ViT), (ConvNeXt − ViT): 120° − 90° | E3 (tương tác, CI bootstrap) |
| RQ3-a/b/c | ConvNeXt vs ResNet, ConvNeXt vs ViT, ResNet vs ViT ở ~2M, 120° | E4 + E3 |

## Quy tắc kết luận

"A tốt hơn B" khi: Δ PSNR > 0, CI 95% bootstrap phân tầng (seed, khối 10 lát) không chứa 0,
Δ cùng dấu ở cả hai bệnh nhân, và Δ lớn hơn độ lệch chuẩn giữa các seed. Báo cáo kèm p Wilcoxon và p Holm.
Kết quả âm và lượt phân kỳ được báo cáo nguyên trạng.

## Phụ lục bổ sung sau khi đóng băng (2026-10-04)

Phần trên giữ nguyên như lúc đóng băng. Các phân tích dưới đây được thêm SAU khi tập kiểm thử của bốn kiến trúc đầu
đã mở, nên chỉ có giá trị thăm dò/kiểm tra độ bền và phải được ghi rõ như vậy trong khóa luận.

- **LEARN–RegNet** (kiến trúc thứ năm): cùng E2 (chọn lr trên tập xác thực), E3, E4 (~2M), E5, E6. So sánh với từng
  kiến trúc còn lại ở mỗi khoảng quét và trong nhóm ~2M nằm ở bảng `B10_supplementary`, hiệu chỉnh Holm trong họ riêng;
  không thêm vào họ RQ1–RQ3 của bảng B8.
- **Biến thể dữ liệu** (Mayo + nhiễu Poisson 10⁶ / 10⁵; DeepLesion): lặp lại nguyên các so sánh RQ1–RQ3 trên từng
  biến thể với cùng quy tắc kết luận. Câu hỏi: thứ hạng kiến trúc và dấu của các Δ có giữ nguyên không. Kết quả của
  mỗi biến thể được báo cáo riêng, không gộp lát của các bộ dữ liệu khác nhau vào một kiểm định.
- Trên DeepLesion (15 bệnh nhân kiểm thử), điều kiện "Δ cùng dấu ở mọi bệnh nhân" chặt hơn nhiều so với Mayo
  (2 bệnh nhân); báo cáo kèm số bệnh nhân có Δ > 0.
- **Backbone mới và bố cục RegNet** (2026-10-05): thêm bốn họ (U-Net, ResNeXt, DenseNet, EfficientNet) và bản bố cục
  RegNet cho mọi họ. Bảng `B10_supplementary` gồm BS1 (backbone mới so với LEARN–CNN, từng khoảng quét), BS2 (bản bố
  cục RegNet so với bản bố cục đều của cùng họ, cùng cỡ tham số) và BS3 (LEARN–RegNet trong nhóm ~2M); Holm trong
  riêng bảng này. Giả thuyết thăm dò: dồn kênh/khối về mức phân giải thấp (bố cục RegNet) có lợi hơn ở 90° so với 120°.
- **Một seed cho mỗi cấu hình** (2026-10-05, lệch so với kế hoạch 3 seed vì giới hạn GPU): mọi lượt mới (backbone
  mới trên Mayo, mọi backbone trên DeepLesion) chỉ train seed 0. Bốn kiến trúc đầu trên Mayo đã có 3 seed; báo cáo
  dùng seed có PSNR **xác thực** tốt nhất (bảng `B11_seeds`), không chọn theo tập test. Hệ quả phải nêu khi diễn
  giải: (i) bốn kiến trúc đầu được chọn tốt-nhất-trong-ba còn các cấu hình khác chỉ có một lần thử; (ii) không còn
  ước lượng được độ lệch qua seed cho cấu hình mới, nên dùng độ lệch đo trên bốn kiến trúc đầu (bảng
  `B12_seed_spread`: trung vị ±0,29 dB PSNR, lớn nhất ±0,9 dB ở E3 và ±2,6 dB ở E4) làm mức nhiễu; khác biệt giữa hai
  kiến trúc nhỏ hơn mức này không được diễn giải là do kiến trúc. Điều kiện "Δ lớn hơn độ lệch seed" của quy tắc kết
  luận không áp dụng được cho so sánh một seed. `--seeds all` cho lại bảng trung bình ± độ lệch chuẩn qua seed.
- RegNet trong các so sánh trên là RegNetY-200MF; bản tích hợp RegNet của mỗi họ gồm cả bố cục 200MF lẫn SE, nên BS2
  trả lời "tích hợp RegNetY có lợi không", không tách riêng được phần của bố cục và phần của SE.
