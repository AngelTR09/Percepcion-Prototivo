# Portal (barra lateral con los tres servicios). Contexto: la raiz del repo.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY servicios/portal/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY servicios/__init__.py ./servicios/__init__.py
COPY servicios/portal ./servicios/portal

USER app
EXPOSE 8000
ENV PORT=8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/salud' % os.environ.get('PORT','8000'), timeout=4)"
# Un solo worker: el modelo se carga una vez en memoria.
CMD ["sh", "-c", "exec uvicorn servicios.portal.servidor:app --host 0.0.0.0 --port ${PORT:-8000} --no-proxy-headers"]
