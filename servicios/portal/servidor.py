"""Portal: reune los tres servicios (VIDEO, IMAGEN, PREDICCION) en una barra lateral. No tiene logica de negocio.

  GET /salud       chequeo de salud
  GET /config.js   direcciones (las que abre el NAVEGADOR) de cada servicio
  GET /            la pagina con la barra lateral; cada apartado es el servicio real, vivo y separado

Variables: URL_VIDEO, URL_IMAGEN, URL_PREDICCION (direcciones publicas de cada servicio), PORT.
"""
import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Portal", version="1.0.0")


@app.middleware("http")
async def sin_cache(request, call_next):
    """La interfaz siempre se revalida: el navegador no se queda con una version vieja."""
    respuesta = await call_next(request)
    if not request.url.path.startswith("/api"):
        respuesta.headers["Cache-Control"] = "no-cache"
    return respuesta


@app.get("/salud")
def salud():
    return {"servicio": "portal", "estado": "ok"}


@app.get("/config.js")
def config():
    servicios = {
        "video": os.getenv("URL_VIDEO", "http://localhost:8001").rstrip("/"),
        "imagen": os.getenv("URL_IMAGEN", "http://localhost:8002").rstrip("/"),
        "prediccion": os.getenv("URL_PREDICCION", "http://localhost:8003").rstrip("/"),
    }
    return Response(f"window.SERVICIOS = {json.dumps(servicios)};", media_type="application/javascript",
                    headers={"Cache-Control": "no-store"})


app.mount("/", StaticFiles(directory=STATIC, html=True), name="portal")
