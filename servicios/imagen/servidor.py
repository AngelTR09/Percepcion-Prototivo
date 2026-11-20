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
import threading
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
ETIQUETAS = {"persona": "Persona", "mochila": "Mochila", "telefono": "Telefono", "audifonos": "Audifonos", "reloj": "Reloj",
             "cuaderno": "Cuaderno/Libro", "libro": "Cuaderno/Libro", "laptop": "Laptop"}

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
_version = 0


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


def _completar(mi, trabajo):
    """Segundo tiempo: las imagenes de evidencia (parte pesada). Si llego otro hallazgo, se descarta."""
    global _ultimo
    try:
        extra = verificacion.construir_secciones(trabajo)
    except Exception:
        return
    actual = _ultimo
    if actual.get("version") == mi:
        _ultimo = dict(actual, completo=True, resultado=dict(actual["resultado"], **extra))


def _procesar(h):
    """Primer tiempo: veredicto y entrega a PREDICCION. Es lo unico que espera VIDEO."""
    global _ultimo, _version
    inicio = time.perf_counter()
    r = verificacion.analizar_rapido(_decodificar(h.get("objeto_b64")), str(h.get("clase", "")).strip().lower(),
                                     float(h.get("confianza", 0)))
    trabajo = r.pop("_trabajo")
    # Si IMAGEN reclasifica el objeto (VIDEO dijo laptop, es un telefono), todo lo que sigue usa la clase corregida.
    reclasificado = r["clase"] != r["clase_original"]
    etiqueta = ETIQUETAS.get(r["clase"], r["clase"]) if reclasificado else h.get("etiqueta")
    origen = h.get("etiqueta") if reclasificado else None
    entregado, detalle = None, None
    if r["pasa_a_prediccion"]:   # lo no verificado se descarta: no llega a PREDICCION
        cuerpo = {
            "hallazgo": {"clase": r["clase"], "etiqueta": etiqueta, "confianza": h.get("confianza"), "umbral": h.get("umbral"), "reclasificado_desde": origen},
            "imagen": {"veredicto": r["veredicto"], "motivos": r["motivos"], "decision": r["decision"],
                       "objeto": r["objeto"], "miniaturas": r["miniaturas"]},
            "contrato": r["contrato_prediccion"],
            "persona": h.get("persona"),
        }
        try:
            entregado = _entregar_a_prediccion(cuerpo)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            entregado, detalle = False, str(error)
    _version += 1
    mi = _version
    resumen = {k: v for k, v in r.items() if k not in ("contrato_prediccion", "objeto", "miniaturas")}
    _ultimo = {
        "ts": time.time(), "version": mi, "completo": False, "tiempo_ms": round((time.perf_counter() - inicio) * 1000),
        "hallazgo": {"clase": r["clase"], "etiqueta": etiqueta, "confianza": h.get("confianza"), "umbral": h.get("umbral"), "reclasificado_desde": origen},
        "resultado": dict(resumen, secciones=[], bordes=None),
        "prediccion": {"entregado": bool(entregado), "detalle": detalle},
    }
    threading.Thread(target=_completar, args=(mi, trabajo), daemon=True).start()
    return {"clase": r["clase"], "etiqueta": etiqueta, "reclasificado_desde": origen,
            "veredicto": r["veredicto"], "pasa_a_prediccion": r["pasa_a_prediccion"],
            "decision": r["decision"], "motivos": r["motivos"],
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
