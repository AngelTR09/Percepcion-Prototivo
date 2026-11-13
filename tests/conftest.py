import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import ensamble  # noqa: E402,F401  (prepara sys.path de integrador/ y prediccion/)
from ensamble import servicio  # noqa: E402


def imagen_sintetica(ancho=640, alto=480):
    """Escena con tres bloques de color: suficiente para que IMAGEN segmente y extraiga rasgos."""
    img = np.full((alto, ancho, 3), 70, np.uint8)
    cv2.rectangle(img, (60, 80), (260, 400), (200, 160, 90), -1)
    cv2.rectangle(img, (320, 120), (560, 300), (60, 180, 220), -1)
    cv2.circle(img, (450, 380), 50, (230, 230, 230), -1)
    return img


def a_base64(img):
    ok, buffer = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buffer.tobytes()).decode("ascii")


class DetectorFalso:
    """Imita al detector de VIDEO: ve una persona y un telefono, y solo recupera el telefono tras mejorar."""

    def __call__(self, frame):
        # La imagen mejorada es mas brillante que la original sintetica oscura: se distingue por la media.
        brillante = float(frame.mean()) > 100
        detecciones = [
            {"clase": "persona", "confianza": 0.91, "bbox": [60, 80, 260, 400]},
            {"clase": "telefono", "confianza": 0.55 if not brillante else 0.80, "bbox": [320, 120, 560, 300]},
        ]
        return detecciones


@pytest.fixture
def detector_falso():
    detector = DetectorFalso()
    servicio.establecer_detector(detector)
    yield detector
    servicio.establecer_detector(None)


@pytest.fixture
def imagen():
    return imagen_sintetica()


@pytest.fixture
def cliente():
    from fastapi.testclient import TestClient
    from servidor import app
    return TestClient(app)
