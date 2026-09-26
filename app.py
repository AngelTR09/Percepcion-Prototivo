"""
Smart Classroom Vision - Panel en la nube (Render).
Muestra las métricas del modelo, la calidad del dataset y una sesión de demostración de indicadores por minuto.
Solo sirve datos fijos incluidos en el repositorio: no recibe ni guarda nada del aula. Acceso con usuario y contraseña.
"""
import csv
import os
import secrets
from pathlib import Path

from flask import Flask, Response, jsonify, request

AQUI = Path(__file__).parent
USUARIO = os.getenv("PANEL_USUARIO", "docente")
CLAVE = os.getenv("PANEL_CLAVE") or secrets.token_urlsafe(8)
if not os.getenv("PANEL_CLAVE"):
    print(f"[aviso] PANEL_CLAVE no definida; usuario={USUARIO} clave={CLAVE}", flush=True)

app = Flask(__name__, static_folder=str(AQUI / "static"))


def leer_demo():
    ruta = AQUI / "datos" / "demo_indicadores.csv"
    if not ruta.exists():
        return []
    return [{k: (float(v) if k != "iluminacion" and v != "" else v) for k, v in fila.items()}
            for fila in csv.DictReader(open(ruta))]


def es(a, b):
    return secrets.compare_digest((a or "").encode(), b.encode())


@app.before_request
def autenticar():
    if request.path == "/salud":                                    # chequeo de salud de Render: sin datos
        return None
    a = request.authorization
    if not (a and es(a.username, USUARIO) and es(a.password, CLAVE)):
        return Response("Acceso restringido", 401, {"WWW-Authenticate": 'Basic realm="Smart Classroom Vision"'})


@app.after_request
def sin_cache(r):
    r.headers["Cache-Control"] = "no-store"
    return r


@app.get("/salud")
def salud():
    return "ok"


@app.get("/api/indicadores")
def indicadores():
    return jsonify(origen="demostración", filas=leer_demo())


@app.get("/api/metricas")
def metricas():
    return Response((AQUI / "datos" / "metricas.json").read_text(), mimetype="application/json")


@app.get("/")
def inicio():
    return Response((AQUI / "pagina.html").read_text(), mimetype="text/html")


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=int(os.getenv("PORT", "10000")), _quiet=True)
