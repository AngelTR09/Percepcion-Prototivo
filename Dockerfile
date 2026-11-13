# Sistema integrado VIDEO + IMAGEN + PREDICCION en un solo contenedor (Docker local y Render).
#   docker build -t supervision .
#   docker run --rm -p 8000:8000 supervision        ->  http://localhost:8000  y  /integrado
# CON_VIDEO=0 construye sin torch/ultralytics (imagen liviana, sin el detector de VIDEO).
FROM python:3.12-slim

ARG CON_VIDEO=1
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

WORKDIR /app

# ultralytics trae opencv-python (con GUI, pide libGL): se deja una sola variante, la headless.
# Dependencias primero: la capa se reutiliza mientras no cambien los requirements.
COPY requirements.txt requirements-video.txt ./
RUN if [ "$CON_VIDEO" = "1" ]; then \
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
        && pip install -r requirements.txt -r requirements-video.txt         && pip uninstall -y opencv-python         && pip install --force-reinstall --no-deps opencv-python-headless==4.11.0.86; \
    else \
        pip install -r requirements.txt; \
    fi

COPY ensamble ./ensamble
COPY integrador ./integrador
COPY prediccion ./prediccion
COPY servidor.py ./

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/salud' % os.environ.get('PORT','8000'), timeout=4)"

# Render inyecta $PORT. Un solo worker: el detector de VIDEO se carga una vez en memoria.
CMD ["sh", "-c", "exec uvicorn servidor:app --host 0.0.0.0 --port ${PORT:-8000} --no-proxy-headers"]
