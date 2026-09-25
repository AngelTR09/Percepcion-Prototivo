"""
Smart Classroom Vision - Panel en la nube (Render).
Muestra métricas del modelo, calidad del dataset e indicadores agregados por minuto.
Solo maneja números: no recibe ni guarda imágenes ni video (RNF02). Acceso con usuario y contraseña.
"""
import csv
import json
import os
import secrets
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request

AQUI = Path(__file__).parent
USUARIO = os.getenv("PANEL_USUARIO", "docente")
CLAVE = os.getenv("PANEL_CLAVE") or secrets.token_urlsafe(8)
API_KEY = os.getenv("API_KEY") or secrets.token_urlsafe(16)
ARCHIVO_VIVO = Path(os.getenv("DATOS_VIVOS", "/tmp/indicadores_nube.json"))   # disco efímero en el plan gratuito
if not os.getenv("PANEL_CLAVE") or not os.getenv("API_KEY"):
    print(f"[aviso] Credenciales no definidas; usuario={USUARIO} clave={CLAVE} api_key={API_KEY}", flush=True)

app = Flask(__name__, static_folder=str(AQUI / "static"))
candado = threading.Lock()
# Último cuadro del aula (ya difuminado). Solo en memoria: no se escribe en disco y se descarta si deja de llegar.
VISTA = {"jpg": None, "t": 0.0, "pedida": 0.0}
MAX_JPG = 400_000


def leer_vivos():
    try:
        return json.loads(ARCHIVO_VIVO.read_text())
    except Exception:
        return []


def leer_demo():
    ruta = AQUI / "datos" / "demo_indicadores.csv"
    if not ruta.exists():
        return []
    return [{k: (float(v) if k not in ("iluminacion",) and v not in ("",) else v) for k, v in fila.items()}
            for fila in csv.DictReader(open(ruta))]


def es(a, b):
    return secrets.compare_digest((a or "").encode(), b.encode())


@app.before_request
def autenticar():
    if request.path == "/salud":                                                # chequeo de salud de Render: sin datos
        return None
    if (request.method, request.path) in {("POST", "/api/indicadores"), ("POST", "/api/vista"), ("GET", "/api/vista/pregunta")}:
        # el aula se identifica con clave de API
        if not es(request.headers.get("X-API-Key"), API_KEY):
            return Response("Clave de API incorrecta", 401)
        return None
    a = request.authorization
    if not (a and es(a.username, USUARIO) and es(a.password, CLAVE)):
        return Response("Acceso restringido", 401, {"WWW-Authenticate": 'Basic realm="Smart Classroom Vision"'})


@app.after_request
def sin_cache(r):
    r.headers["Cache-Control"] = "no-store"
    return r


@app.post("/api/indicadores")
def recibir():
    fila = request.get_json(silent=True)
    if not isinstance(fila, dict) or len(fila) > 40:
        return Response("Formato inválido", 400)
    # Solo se aceptan números y textos cortos: nunca datos binarios ni imágenes
    limpia = {str(k)[:30]: (v if isinstance(v, (int, float)) else str(v)[:40]) for k, v in fila.items()}
    with candado:
        filas = leer_vivos() + [limpia]
        ARCHIVO_VIVO.write_text(json.dumps(filas[-2000:]))
    return jsonify(ok=True, total=len(filas))


def mirando():
    return time.time() - VISTA["pedida"] < 15                                  # alguien tiene abierta la cámara en la página


@app.get("/api/vista/pregunta")
def vista_pregunta():
    return jsonify(mirando=mirando())


@app.post("/api/vista")
def vista_recibir():
    jpg = request.get_data(cache=False)
    if request.mimetype != "image/jpeg" or len(jpg) > MAX_JPG or not jpg.startswith(b"\xff\xd8"):
        return Response("Solo se acepta una imagen JPEG de hasta 400 KB", 400)
    with candado:
        VISTA["jpg"], VISTA["t"] = jpg, time.time()
    return jsonify(mirando=mirando())


@app.get("/vista.jpg")
def vista_ver():
    with candado:
        VISTA["pedida"] = time.time()                                          # avisa al aula de que alguien está mirando
        jpg, antigua = VISTA["jpg"], time.time() - VISTA["t"] > 8
        if jpg is None or antigua:
            VISTA["jpg"] = None if antigua else VISTA["jpg"]                   # una imagen vieja se descarta
    if jpg is None or antigua:
        return Response("Sin señal del aula", 404)
    return Response(jpg, mimetype="image/jpeg")


@app.get("/api/indicadores")
def indicadores():
    vivos = leer_vivos()
    if vivos:
        ultima = vivos[-1].get("sesion")
        return jsonify(origen="aula", filas=[f for f in vivos if f.get("sesion") == ultima])
    return jsonify(origen="demostración", filas=leer_demo())


@app.get("/salud")
def salud():
    return "ok"


@app.get("/api/metricas")
def metricas():
    return Response((AQUI / "datos" / "metricas.json").read_text(), mimetype="application/json")


@app.get("/")
def inicio():
    return Response((AQUI / "pagina.html").read_text(), mimetype="text/html")


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=int(os.getenv("PORT", "10000")), _quiet=True)
