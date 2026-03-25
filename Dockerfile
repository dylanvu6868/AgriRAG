# Sử dụng Python 3.10 bản gọn nhẹ nhất (slim) để tiết kiệm dung lượng RAM/Ổ cứng
FROM python:3.10-slim

# Thiết lập thư mục làm việc trong Container
WORKDIR /app

# Cài đặt biến môi trường chặn Python tạo cache pyc để giảm thiểu bộ nhớ
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy file requirements.txt vào trước để tận dụng Docker Cache cho thư viện
COPY requirements.txt .

# Cài đặt các thư viện Python (Thêm build-essential nếu gặp lỗi biên dịch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy toàn bộ mã nguồn của dự án vào Container
COPY . .

# Mở cổng 8000
EXPOSE 8000

# Lệnh khởi chạy Server bằng Uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
