"""Integración real: video -> adaptador -> detector de VIDEO. Se omite si faltan datos o modelo."""
import os

import pytest

from imagen import adaptador_video

VIDEO = os.path.join(os.path.dirname(__file__), "..", "datos", "prueba.mp4")
MODELO = os.path.join(adaptador_video.RUTA_VIDEO_BASE, "modelo_ia", "best.pt")


@pytest.mark.skipif(not (os.path.exists(VIDEO) and os.path.exists(MODELO)), reason="falta video o modelo")
def test_adaptador_entrega_contrato_y_detector_funciona():
    entrada = next(adaptador_video.leer_frames(VIDEO))
    assert {"frame_id", "timestamp", "fps", "fuente", "frame"} <= set(entrada)
    detecciones = adaptador_video.cargar_detector()(entrada["frame"])
    assert all(len(d["bbox"]) == 4 for d in detecciones)
