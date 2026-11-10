# Hina-ASR — PhoWhisper baseline + fine-tune starter

Model: `vinai/PhoWhisper-base` (đối chứng `vinai/PhoWhisper-small`). Chỉ có code; **chưa có dữ liệu, chưa fine-tune và chưa có checkpoint mới**. Bắt đầu bằng baseline, chọn checkpoint dựa trên validation, kiểm tra test một lần sau khi chốt.

## Cài đặt (Python 3.11, GPU training)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu CUDA=False, cài lại PyTorch theo hướng dẫn build CUDA tương thích tại https://pytorch.org/get-started/locally/ hoặc dùng Colab GPU.

## Dữ liệu

Tạo `data/train.csv`, `data/val.csv`, `data/test.csv`, mỗi file gồm `audio_path,text,speaker_id,session_id`. `audio_path` tương đối theo thư mục chứa manifest. Chuẩn hóa dữ liệu WAV mono 16 kHz, mỗi mẫu tối đa 30 giây. Ví dụ chuyển mã:

```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

Tách train/val/test theo **speaker**, không chia ngẫu nhiên từng file của cùng người; cũng không chia các frame/video/phiên ghi âm gần trùng sang tập khác. Dữ liệu mẫu `manifest_example.csv` chỉ minh họa, không chứa audio thật.

```powershell
python src/audit.py --train data/train.csv --val data/val.csv --test data/test.csv
```

## Benchmark trước khi train

```powershell
python src/evaluate.py --model vinai/PhoWhisper-base --manifest data/val.csv --output outputs/base_val_predictions.csv
```

Giữ test độc lập cho đánh giá cuối. Đánh giá trên validation để so sánh các cấu hình trong giai đoạn phát triển.

## Fine-tune trên GPU

```powershell
python src/train.py --train data/train.csv --val data/val.csv --model vinai/PhoWhisper-base --output outputs/run01 --epochs 5
```

Sau khi chọn model trên validation, chạy **cả baseline và fine-tuned** trên test độc lập một lần với cùng protocol:

```powershell
python src/evaluate.py --model vinai/PhoWhisper-base --manifest data/test.csv --output outputs/base_test_predictions.csv
python src/evaluate.py --model outputs/run01/best-model --manifest data/test.csv --output outputs/finetuned_test_predictions.csv
```

Lưu bảng WER/CER, theo vùng giọng / tiếng ồn, lỗi bỏ sót phủ định, WER cho command quan trọng và latency p50/p95 trên máy inference. Không nên kết luận fine-tune cải thiện nếu chưa có kết quả đo độc lập.

## Colab

Upload cả repository và các tệp audio đã được cấp quyền lên Drive rồi mount vào Colab, chạy `!pip install -r requirements.txt`. Đặt notebook working directory thành repo, chạy lệnh Python tương tự. Chú ý lưu checkpoint ngoài phiên runtime (Drive hoặc kho model private) để tránh mất dữ liệu.
