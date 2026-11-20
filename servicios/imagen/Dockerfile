# Servicio IMAGEN (tecnicas del curso). Sin torch: imagen liviana. Contexto: la raiz del repo.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

WORKDIR /app

COPY servicios/imagen/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY integrador/imagen ./integrador/imagen
COPY servicios/__init__.py ./servicios/__init__.py
COPY servicios/imagen ./servicios/imagen
COPY servicios/comun/estilos.css ./servicios/imagen/static/estilos.css

USER app
EXPOSE 8000
ENV PORT=8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/salud' % os.environ.get('PORT','8000'), timeout=4)"
# Un solo worker: el modelo se carga una vez en memoria.
CMD ["sh", "-c", "exec uvicorn servicios.imagen.servidor:app --host 0.0.0.0 --port ${PORT:-8000} --no-proxy-headers"]
