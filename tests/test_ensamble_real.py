"""Cadena con el detector REAL de VIDEO (best.pt). Se omite si falta ultralytics, el modelo o el video."""
import importlib.util

import cv2
import pytest

from conftest import RAIZ, a_base64
from ensamble import servicio

VIDEO = RAIZ / "integrador" / "datos" / "prueba.mp4"
MODELO = RAIZ / "integrador" / "video_base" / "modelo_ia" / "best.pt"

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None or not VIDEO.exists() or not MODELO.exists(),
    reason="falta ultralytics, el modelo de VIDEO o el video de prueba",
)


def test_cadena_real_video_imagen_prediccion(cliente):
    servicio.establecer_detector(None)
    captura = cv2.VideoCapture(str(VIDEO))
    correcto, frame = captura.read()
    captura.release()
    assert correcto
    r = cliente.post("/api/integrado/analizar", json={"imagen_b64": a_base64(frame), "devolver_imagen": True})
    assert r.status_code == 200, r.text
    d = r.json()
    assert cliente.get("/api/integrado/estado").json()["detector_video"] == "cargado"
    assert all(len(x["bbox"]) == 4 for x in d["video"]["detecciones"])
    assert d["prediccion"]["integracion"]["fuentes_recibidas"] == ["video", "imagen"]
    assert d["imagen"]["tiempo_ms"] > 0
