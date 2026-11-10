FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY router ./router

RUN useradd --create-home routeruser
USER routeruser

EXPOSE 8000
CMD ["uvicorn", "router.main:app", "--host", "0.0.0.0", "--port", "8000"]
