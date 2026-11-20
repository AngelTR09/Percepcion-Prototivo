"""Servidor unico del sistema integrado (el que corre en Docker/Render).

    uvicorn servidor:app --host 0.0.0.0 --port $PORT

  /salud                       chequeo de salud (Render)
  /api/integrado/*             cadena VIDEO -> IMAGEN -> PREDICCION sobre una imagen
  /api/integrado/verificar     IMAGEN verifica el recorte de un objeto senalado por VIDEO (pasos como imagenes)
  /api/integrado/detectar      objetos con su caja en vivo (la camara del panel dibuja los recuadros)
  /integrado                   pagina de demostracion de la cadena
  /api/*  y  /                 el backend y el panel de PREDICCION, sin cambios
"""
from pathlib import Path

import ensamble  # noqa: F401  (prepara sys.path antes de importar a los grupos)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.app import app as app_prediccion
from ensamble import servicio, verificacion

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


@app.post("/api/integrado/detectar")
async def detectar(request: Request):
    """Cuerpo: JPEG crudo. Devuelve los objetos con su caja para dibujarlos en vivo sobre la camara."""
    crudo = await request.body()
    if not crudo:
        raise HTTPException(422, "Se requiere una imagen JPEG en el cuerpo")
    if len(crudo) > LIMITE_BYTES:
        raise HTTPException(413, "La imagen supera 8 MB")
    try:
        return await run_in_threadpool(servicio.detectar_en_vivo, crudo)
    except servicio.ImagenInvalida as error:
        raise HTTPException(422, str(error)) from None
    except servicio.DetectorNoDisponible as error:
        return JSONResponse(status_code=503, content={"estado": "detector_no_disponible", "detail": str(error)})


@app.post("/api/integrado/verificar")
async def verificar(request: Request, clase: str, confianza: float = 0.0):
    """Cuerpo: JPEG del recorte que VIDEO senalo. IMAGEN lo verifica y devuelve cada paso como imagen."""
    crudo = await request.body()
    if not crudo:
        raise HTTPException(422, "Se requiere el recorte JPEG en el cuerpo")
    if len(crudo) > LIMITE_BYTES:
        raise HTTPException(413, "La imagen supera 8 MB")
    try:
        return await run_in_threadpool(verificacion.verificar_recorte, crudo, clase.strip().lower(), confianza)
    except servicio.ImagenInvalida as error:
        raise HTTPException(422, str(error)) from None
    except Exception as error:
        return JSONResponse(status_code=503, content={"estado": "no_disponible", "detail": f"{type(error).__name__}: {error}"})


@app.get("/api/integrado/sonido-alerta")
def sonido_alerta():
    """El mismo sonido de alerta que reproduce VIDEO."""
    ruta = Path(__file__).parent / "integrador" / "video_base" / "sonido" / "alerta.mp3"
    if not ruta.is_file():
        raise HTTPException(404, "Sin sonido de alerta")
    return FileResponse(ruta, media_type="audio/mpeg")


@app.get("/integrado")
def pagina_integrado():
    return FileResponse(Path(__file__).parent / "ensamble" / "integrado.html")


# Al final: el montaje en "/" de PREDICCION atiende todo lo que no sea de arriba.
app.mount("/", app_prediccion)
