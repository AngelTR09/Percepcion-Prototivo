"""Servidor unico del sistema integrado (el que corre en Docker/Render).

    uvicorn servidor:app --host 0.0.0.0 --port $PORT

  /salud                       chequeo de salud (Render)
  /api/integrado/*             cadena VIDEO -> IMAGEN -> PREDICCION sobre una imagen
  /integrado                   pagina de demostracion de la cadena
  /api/*  y  /                 el backend y el panel de PREDICCION, sin cambios
"""
from pathlib import Path

import ensamble  # noqa: F401  (prepara sys.path antes de importar a los grupos)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.app import app as app_prediccion
from ensamble import servicio

app = FastAPI(title="Sistema de Supervision Inteligente (VIDEO + IMAGEN + PREDICCION)", version="1.0.0")
LIMITE_BYTES = 8 * 1024 * 1024


@app.get("/salud")
def salud():
    return {"estado": "ok"}


@app.get("/api/integrado/estado")
def estado():
    return {
        "estado": "listo",
        "cadena": ["VIDEO", "IMAGEN", "PREDICCION"],
        "detector_video": servicio.estado_detector(),
    }


@app.post("/api/integrado/analizar")
async def analizar(request: Request):
    """Cuerpo JSON: {imagen_b64, timestamp?, objetos_permitidos?, devolver_imagen?}."""
    crudo = await request.body()
    if len(crudo) > LIMITE_BYTES:
        raise HTTPException(413, "La imagen supera 8 MB")
    try:
        datos = await request.json()
    except Exception:
        raise HTTPException(422, "Se requiere un JSON") from None
    if not isinstance(datos, dict):
        raise HTTPException(422, "Se requiere un objeto JSON")
    permitidos = datos.get("objetos_permitidos") or []
    if not isinstance(permitidos, list) or not all(isinstance(x, str) for x in permitidos):
        raise HTTPException(422, "objetos_permitidos debe ser una lista de textos")
    try:
        timestamp = float(datos.get("timestamp", 0))
        frame = servicio.decodificar(datos.get("imagen_b64"))
    except (TypeError, ValueError) as error:
        raise HTTPException(422, str(error)) from None
    try:
        return await run_in_threadpool(
            servicio.analizar_frame, frame, timestamp=timestamp,
            objetos_permitidos=permitidos, devolver_imagen=bool(datos.get("devolver_imagen")),
        )
    except servicio.DetectorNoDisponible as error:
        return JSONResponse(status_code=503, content={"estado": "detector_no_disponible", "detail": str(error)})


@app.get("/integrado")
def pagina_integrado():
    return FileResponse(Path(__file__).parent / "ensamble" / "integrado.html")


# Al final: el montaje en "/" de PREDICCION atiende todo lo que no sea de arriba.
app.mount("/", app_prediccion)
