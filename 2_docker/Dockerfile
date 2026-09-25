FROM python:3.12-slim

WORKDIR /app

# PyTorch versión CPU (mucho más liviana que la de CUDA)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt \
    # ultralytics arrastra opencv-python (con interfaz gráfica, y quizá la 5.x): se deja solo la versión headless 4.x
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps "opencv-python-headless<5"

COPY prototipo_smart_classroom.py panel_web.py ./
# Si existe, se copia el modelo de comportamientos entrenado (RF05); el comodín evita fallar si aún no existe
COPY modelo_comportamientos.p[t] ./

# Descarga los modelos (YOLO y detector de rostros) al construir la imagen, para operar sin internet en el aula
RUN PANEL=0 python -c "import prototipo_smart_classroom"

ENV MOSTRAR_VENTANA=0 \
    PANEL_HOST=0.0.0.0 \
    SALIDA_CSV=/datos/indicadores_aula.csv \
    PYTHONUNBUFFERED=1
VOLUME /datos
EXPOSE 8000

CMD ["python", "prototipo_smart_classroom.py"]
