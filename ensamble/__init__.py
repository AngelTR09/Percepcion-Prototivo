"""Ensamblaje VIDEO -> IMAGEN -> PREDICCION.

Esta carpeta no modifica a ningun grupo: solo conecta sus salidas.
  - integrador/video_base/  VIDEO   (copia de solo lectura)
  - integrador/imagen/      IMAGEN
  - prediccion/             PREDICCION (backend FastAPI + frontend)
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
INTEGRADOR = RAIZ / "integrador"
PREDICCION = RAIZ / "prediccion"

# Cada grupo conserva su propia raiz de imports (`imagen`, `backend`).
for ruta in (INTEGRADOR, PREDICCION):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))
