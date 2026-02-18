FROM python:3.9-slim

WORKDIR /app

# Instala dependências
COPY requirements.txt .

# 1. Instala PyTorch versão CPU (muito mais leve, evita erro de espaço em disco)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Instala o restante das dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código fonte
COPY . .

# Cria diretório de modelos (caso não esteja montado como volume)
RUN mkdir -p models

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]