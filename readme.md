# DS201.R11_Lab02_CNN

## Giới thiệu

Bài tập về nhà huấn luyện và đánh giá các mô hình CNN trên VinaFood21:

1. **LeNet:** ảnh 64×64, so sánh AvgPool và MaxPool.
2. **GoogLeNet:** bản rút gọn 3 Inception block, ảnh 64×64.
3. **ResNet-18:** huấn luyện từ đầu, so sánh có/không augmentation.
4. **ResNet50 pretrained:** feature extraction và fine-tuning từ HuggingFace.

Các mô hình dùng Adam; đánh giá bằng accuracy, precision, recall và F1 macro.
Dữ liệu có `train/` và `test/`; validation được tách từ train với seed cố định.
Kết quả, bảng và biểu đồ được hiển thị trực tiếp trong notebook, đồng thời tự động
lưu thành CSV và PNG.

## Kết quả xuất trên Kaggle

Mặc định lưu vào `/kaggle/working/outputs/`, chia theo bài và cấu hình:

```text
outputs/
├── bai1/
│   ├── lenet_avg/{tables,figures}/
│   ├── lenet_max/{tables,figures}/
│   └── comparison.csv
├── bai2/
│   ├── googlenet_compact/{tables,figures}/
│   └── comparison.csv
├── bai3/
│   ├── resnet18_aug/{tables,figures}/
│   ├── resnet18_noaug/{tables,figures}/
│   └── comparison.csv
└── bai4/
    ├── resnet50_features/{tables,figures}/
    ├── resnet50_finetune/{tables,figures}/
    ├── comparison.csv
    └── comparison_transfer.csv
```

- `tables/`: `metrics.csv` (tham số, best epoch, thời gian và các độ đo),
  `history.csv`, `layer_table.csv`, `per_class_scores.csv`, `confusion_matrix.csv`.
  GoogLeNet có thêm `inception_blocks.csv`.
- `figures/`: `loss_curve.png` và `confusion_matrix.png`.
- Các bảng so sánh được lưu ngay tại thư mục của bài; so sánh GoogLeNet/LeNet
  nằm ở bài 2, so sánh transfer learning/ResNet-18 nằm ở bài 4.

Khi chạy local, kết quả mặc định lưu vào `outputs/` trong workspace.
Chạy lại cùng cấu hình sẽ ghi đè các file kết quả của cấu hình đó.
Sau khi chạy trên Kaggle, tải thư mục `outputs/` về và chép nội dung vào
thư mục `outputs/` local, giữ cấu trúc `bai1/` đến `bai4/`.

## Cấu trúc dự án

| Thư mục / file | Nội dung |
|---|---|
| `main.ipynb` | Chạy bốn bài và trình bày kết quả theo đề |
| `src/models/` | LeNet, Inception/GoogLeNet, ResNet-18, ResNet50 pretrained |
| `src/data.py` | Nạp ảnh, chia train/validation và tạo DataLoader |
| `src/engine.py`, `src/train.py` | Huấn luyện và giữ trọng số tốt nhất theo validation trong RAM |
| `src/metrics.py`, `src/visualize.py`, `src/report.py` | Tính chỉ số, vẽ hình và hiển thị kết quả |
| `src/artifacts.py` | Tự động xuất bảng CSV và hình PNG theo bài |
| `src/utils.py` | Seed, đếm tham số và kiểm tra shape |
| `configs/` | Bảy cấu hình cho bốn bài |
| `data/VinaFood21/train/`, `data/VinaFood21/test/` | Vị trí đặt dataset local |
| `outputs/` | Bảng, biểu đồ và so sánh được chia theo bài 1–4 |
| `references/assignments.ipynb` | Đề bài gốc |
| `tests/` | Kiểm tra code |
| `requirements.txt` | Các thư viện sử dụng |
