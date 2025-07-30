FROM python:3.12.6-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" || true

COPY . .

RUN mkdir -p data/backups data/temp

EXPOSE 5000

CMD ["python", "app.py"]
