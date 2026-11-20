"""Servicio PREDICCION: el panel y el motor del grupo PREDICCION (sin cambios), mas la recepcion de lo que IMAGEN verifico.

  GET  /salud         chequeo de salud
  POST /api/recibir   lo que IMAGEN verifico (objeto, evidencia, contrato y persona) -> evaluacion del motor
  GET  /api/ultimo    ultimo hallazgo recibido con su evaluacion (lo consulta la interfaz)
  /  y  /api/*        el panel de PREDICCION y su backend

Solo llega aqui lo que IMAGEN confirma: lo no verificado se descarta como falso positivo en el servicio IMAGEN.
"""
import json
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "prediccion"))           # PREDICCION conserva su propia raiz de imports (`backend`)

from backend.app import app as app_prediccion  # noqa: E402
from backend.modulos.prediccion.integracion import integrar_fuentes  # noqa: E402

LIMITE_BYTES = 32 * 1024 * 1024

app = FastAPI(title="Servicio PREDICCION", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
_ultimo = {"ts": 0}


@app.get("/salud")
def salud():
    return {"servicio": "prediccion", "estado": "ok"}


@app.get("/api/ultimo")
def ultimo():
    return _ultimo


def _evaluar(cuerpo):
    try:
        evaluacion = integrar_fuentes(cuerpo["contrato"])
    except Exception as error:  # el motor no debe impedir mostrar lo recibido
        evaluacion = {"error": f"{type(error).__name__}: {error}"}
    _ultimo.clear()
    _ultimo.update({"ts": time.time(), "hallazgo": cuerpo.get("hallazgo"), "imagen": cuerpo.get("imagen"),
                    "persona": cuerpo.get("persona"), "prediccion": evaluacion})
    return evaluacion


@app.post("/api/recibir")
async def recibir(request: Request):
    crudo = await request.body()
    if len(crudo) > LIMITE_BYTES:
        raise HTTPException(413, "El hallazgo es demasiado grande")
    try:
        cuerpo = json.loads(crudo)
        if not isinstance(cuerpo, dict) or not isinstance(cuerpo.get("contrato"), dict):
            raise ValueError("Se requiere un objeto JSON con el contrato VIDEO + IMAGEN")
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    evaluacion = await run_in_threadpool(_evaluar, cuerpo)
    return {"estado": "recibido", "indice_revision": evaluacion.get("indice_revision"), "nivel_revision": evaluacion.get("nivel_revision")}


# Al final: el montaje en "/" atiende todo lo que no sea de arriba (panel y backend de PREDICCION).
app.mount("/", app_prediccion)
