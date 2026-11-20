# Sistema de Supervisión Inteligente (VIDEO + IMAGEN + PREDICCIÓN)

Un solo servicio FastAPI con tres apartados: **Video** (detección de objetos en vivo), **Imagen** (verificación del objeto con las técnicas del curso) y **Predicciones** (evaluación y análisis facial). IMAGEN tiene la última palabra: lo que no verifica se descarta como falso positivo y no llega a PREDICCIÓN.

## Desplegar

**Docker (local)**
```
docker compose up --build        # http://localhost:8000
```
`CON_VIDEO=0` (en `docker-compose.yml`) construye sin el detector de VIDEO (imagen liviana).

**Render (plan gratis):** New + Blueprint, elegir este repositorio (`render.yaml`, health check `/salud`).

**Hugging Face Spaces:** `python deploy/publicar_huggingface.py --space USUARIO/supervision-inteligente`

## Estructura
- `servidor.py`, `ensamble/`: servidor y unión de los tres módulos.
- `integrador/video_base/`: VIDEO (modelo `best.pt`). `integrador/imagen/`: IMAGEN.
- `prediccion/backend/`, `prediccion/frontend/`: PREDICCIÓN y la interfaz de los tres apartados.
- `prediccion/pesos/fase8/facial.joblib`: modelo facial (HOG + SVM, CMU Face Images).

Prototipo académico: la salida es evidencia para revisión humana; usar solo datos ficticios.
