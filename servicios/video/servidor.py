"""Servicio VIDEO: detecta objetos con el modelo del grupo VIDEO y entrega los hallazgos a IMAGEN.

  GET  /salud                 chequeo de salud
  GET  /api/estado            estado del detector
  POST /api/detectar          JPEG crudo -> objetos con su caja (para dibujar en vivo)
  POST /api/hallazgo          el hallazgo (objeto + persona) se entrega al servicio IMAGEN
  GET  /api/sonido-alerta     el mismo sonido de alerta que usa VIDEO
  GET  /                      interfaz del apartado VIDEO

Variables: IMAGEN_URL (a donde se entregan los hallazgos), PORT.
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

RAIZ = Path(__file__).resolve().parents[2]
VIDEO_BASE = RAIZ / "integrador" / "video_base"          # codigo del grupo VIDEO (solo lectura)
STATIC = Path(__file__).resolve().parent / "static"
IMAGEN_URL = os.getenv("IMAGEN_URL", "http://imagen:8000").rstrip("/")
LIMITE_BYTES = 8 * 1024 * 1024

# Lo mismo que dibuja VIDEO (modulos/deteccion.py): azul = permitido, rojo = el resto.
PERMITIDOS = {"persona", "mochila"}
NOMBRES = {"persona": "Persona", "mochila": "Mochila", "telefono": "Telefono", "audifonos": "Audifonos",
           "reloj": "Reloj", "cuaderno": "Cuaderno/Libro", "libro": "Cuaderno/Libro", "laptop": "Laptop"}

app = FastAPI(title="Servicio VIDEO", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

_detector = None
_candado = threading.Lock()


def _cargar():
    """Carga perezosa: arrancar el servicio no paga el costo de YOLO."""
    global _detector
    if _detector is None:
        with _candado:
            if _detector is None:
                if str(VIDEO_BASE) not in sys.path:
                    sys.path.insert(0, str(VIDEO_BASE))
                from modulos.deteccion import detectar_objetos  # codigo de VIDEO (carga best.pt al importar)
                _detector = detectar_objetos
    return _detector


@app.get("/salud")
def salud():
    return {"servicio": "video", "estado": "ok"}


@app.get("/api/estado")
def estado():
    return {"servicio": "video", "detector": "cargado" if _detector else "disponible_sin_cargar", "imagen_url": IMAGEN_URL}


def _detectar(jpeg):
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("No se pudo decodificar la imagen")
    detector = _cargar()
    inicio = time.perf_counter()
    with _candado:
        detecciones = detector(frame)
    alto, ancho = frame.shape[:2]
    return {
        "ancho": ancho, "alto": alto, "latencia_ms": round((time.perf_counter() - inicio) * 1000, 1),
        "objetos": [{
            "clase": d["clase"], "etiqueta": NOMBRES.get(d["clase"], d["clase"]),
            "confianza": round(float(d["confianza"]), 4), "permitido": d["clase"] in PERMITIDOS,
            "caja": {"x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"]},
        } for d in detecciones],
    }


@app.post("/api/detectar")
async def detectar(request: Request):
    crudo = await request.body()
    if not crudo:
        raise HTTPException(422, "Se requiere una imagen JPEG en el cuerpo")
    if len(crudo) > LIMITE_BYTES:
        raise HTTPException(413, "La imagen supera 8 MB")
    try:
        return await run_in_threadpool(_detectar, crudo)
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    except Exception as error:
        return JSONResponse(status_code=503, content={"estado": "detector_no_disponible", "detail": f"{type(error).__name__}: {error}"})


def _entregar(cuerpo):
    pedido = urllib.request.Request(f"{IMAGEN_URL}/api/recibir", cuerpo, {"Content-Type": "application/json"})
    with urllib.request.urlopen(pedido, timeout=60) as respuesta:
        return json.loads(respuesta.read())


@app.post("/api/hallazgo")
async def hallazgo(request: Request):
    """Entrega al servicio IMAGEN el objeto prohibido (y la persona) y devuelve su decision."""
    crudo = await request.body()
    if len(crudo) > 4 * LIMITE_BYTES:
        raise HTTPException(413, "El hallazgo es demasiado grande")
    try:
        return await run_in_threadpool(_entregar, crudo)   # no bloquea: IMAGEN vuelve a consultar a /api/detectar
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return JSONResponse(status_code=502, content={"estado": "imagen_no_disponible", "detail": str(error)})


@app.get("/api/sonido-alerta")
def sonido_alerta():
    ruta = VIDEO_BASE / "sonido" / "alerta.mp3"
    if not ruta.is_file():
        raise HTTPException(404, "Sin sonido de alerta")
    return FileResponse(ruta, media_type="audio/mpeg")


app.mount("/", StaticFiles(directory=STATIC, html=True), name="interfaz")
