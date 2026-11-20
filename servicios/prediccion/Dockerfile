# Servicio PREDICCION (panel + motor). Contexto: la raiz del repo.
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

# torch CPU (capa compartida con el otro servicio que usa YOLO: se descarga una sola vez)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY servicios/prediccion/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --force-reinstall --no-deps opencv-python-headless==4.11.0.86

ENV YOLO_CONFIG_DIR=/tmp/Ultralytics \
    AULAPREDICT_MODELOS_FASE8=/app/prediccion/pesos/fase8 \
    AULAPREDICT_DETECTOR_OBJETOS=off
COPY prediccion/backend ./prediccion/backend
COPY prediccion/frontend ./prediccion/frontend
COPY prediccion/pesos ./prediccion/pesos
COPY servicios/__init__.py ./servicios/__init__.py
COPY servicios/prediccion ./servicios/prediccion

USER app
EXPOSE 8000
ENV PORT=8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/salud' % os.environ.get('PORT','8000'), timeout=4)"
# Un solo worker: el modelo se carga una vez en memoria.
CMD ["sh", "-c", "exec uvicorn servicios.prediccion.servidor:app --host 0.0.0.0 --port ${PORT:-8000} --no-proxy-headers"]
