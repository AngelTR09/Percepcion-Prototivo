"""Servicio IMAGEN: verifica con las tecnicas del curso el objeto que le entrega VIDEO.

IMAGEN tiene la ultima palabra: si no verifica el objeto, es un falso positivo y NO se entrega a PREDICCION.

  GET  /salud         chequeo de salud
  POST /api/recibir   hallazgo de VIDEO (objeto + persona) -> analisis -> (si verifica) entrega a PREDICCION
  GET  /api/ultimo    ultimo analisis completo (lo consulta la interfaz)
  GET  /              interfaz del apartado IMAGEN

Variables: VIDEO_URL (para volver a consultar al detector), PREDICCION_URL (a donde se entrega lo verificado), PORT.
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "integrador"))           # modulo IMAGEN del curso: `import imagen`

from servicios.imagen import verificacion  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
PREDICCION_URL = os.getenv("PREDICCION_URL", "http://prediccion:8000").rstrip("/")
LIMITE_BYTES = 32 * 1024 * 1024
MINIATURAS = ("CLAHE", "Bordes Canny", "Otsu + morfología", "Contornos y forma")   # evidencia que viaja a PREDICCION

app = FastAPI(title="Servicio IMAGEN", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def sin_cache(request, call_next):
    """La interfaz siempre se revalida: el navegador no se queda con una version vieja."""
    respuesta = await call_next(request)
    if not request.url.path.startswith("/api"):
        respuesta.headers["Cache-Control"] = "no-cache"
    return respuesta
_ultimo = {"ts": 0}


@app.get("/salud")
def salud():
    return {"servicio": "imagen", "estado": "ok"}


@app.get("/api/ultimo")
def ultimo():
    return _ultimo


def _decodificar(data_url):
    if not isinstance(data_url, str) or "," not in data_url[:100]:
        raise verificacion.ImagenInvalida("objeto_b64 debe ser una data URL")
    return base64.b64decode(data_url.split(",", 1)[1])


def _entregar_a_prediccion(cuerpo):
    pedido = urllib.request.Request(f"{PREDICCION_URL}/api/recibir", json.dumps(cuerpo).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(pedido, timeout=60) as respuesta:
        return json.loads(respuesta.read())


def _procesar(h):
    inicio = time.perf_counter()
    resultado = verificacion.verificar_recorte(_decodificar(h.get("objeto_b64")), str(h.get("clase", "")).strip().lower(),
                                               float(h.get("confianza", 0)))
    entregado, detalle = None, None
    if resultado["pasa_a_prediccion"]:   # lo no verificado se descarta: no llega a PREDICCION
        todos = [it for s in resultado["secciones"] for it in s["items"]]
        cuerpo = {
            "hallazgo": {"clase": resultado["clase"], "etiqueta": h.get("etiqueta"), "confianza": h.get("confianza"), "umbral": h.get("umbral")},
            "imagen": {
                "veredicto": resultado["veredicto"], "motivos": resultado["motivos"], "decision": resultado["decision"],
                "objeto": resultado["secciones"][0]["items"][0]["imagen"],
                "miniaturas": [{"titulo": t["titulo"], "imagen": t["imagen"]} for t in todos if t["titulo"] in MINIATURAS],
            },
            "contrato": resultado["contrato_prediccion"],
            "persona": h.get("persona"),
        }
        try:
            entregado = _entregar_a_prediccion(cuerpo)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            entregado, detalle = False, str(error)
    _ultimo.clear()
    _ultimo.update({
        "ts": time.time(), "tiempo_ms": round((time.perf_counter() - inicio) * 1000),
        "hallazgo": {"clase": resultado["clase"], "etiqueta": h.get("etiqueta"), "confianza": h.get("confianza"), "umbral": h.get("umbral")},
        "resultado": {k: v for k, v in resultado.items() if k != "contrato_prediccion"},
        "prediccion": {"entregado": bool(entregado), "detalle": detalle},
    })
    return {"veredicto": resultado["veredicto"], "pasa_a_prediccion": resultado["pasa_a_prediccion"],
            "decision": resultado["decision"], "motivos": resultado["motivos"],
            "prediccion_recibio": bool(entregado), "prediccion_error": detalle}


@app.post("/api/recibir")
async def recibir(request: Request):
    crudo = await request.body()
    if len(crudo) > LIMITE_BYTES:
        raise HTTPException(413, "El hallazgo es demasiado grande")
    try:
        h = json.loads(crudo)
        if not isinstance(h, dict):
            raise ValueError("Se requiere un objeto JSON")
        return await run_in_threadpool(_procesar, h)
    except (ValueError, verificacion.ImagenInvalida) as error:
        raise HTTPException(422, str(error)) from None


app.mount("/", StaticFiles(directory=STATIC, html=True), name="interfaz")
