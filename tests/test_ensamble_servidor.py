"""El servidor unico (el de Docker/Render): salud, cadena completa y PREDICCION montada sin cambios."""
import base64

import pytest

from conftest import a_base64, imagen_sintetica
from ensamble import servicio


def test_salud_para_render(cliente):
    r = cliente.get("/salud")
    assert r.status_code == 200 and r.json()["estado"] == "ok"


def test_estado_de_la_cadena(cliente):
    r = cliente.get("/api/integrado/estado")
    assert r.status_code == 200
    assert r.json()["cadena"] == ["VIDEO", "IMAGEN", "PREDICCION"]


def test_prediccion_sigue_montada_y_responde(cliente):
    assert cliente.get("/api/salud").status_code == 200
    modulos = cliente.get("/api/modulos").json()["modulos"]
    assert len(modulos) == 4


def test_panel_de_prediccion_se_sirve_en_la_raiz(cliente):
    r = cliente.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]


def test_pagina_de_la_cadena(cliente):
    r = cliente.get("/integrado")
    assert r.status_code == 200 and "/api/integrado/analizar" in r.text


def test_cadena_completa_con_detector_inyectado(cliente, detector_falso):
    r = cliente.post("/api/integrado/analizar", json={
        "imagen_b64": a_base64(imagen_sintetica()), "timestamp": 4, "objetos_permitidos": ["libro"],
        "devolver_imagen": True,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert {x["clase"] for x in d["video"]["detecciones"]} == {"persona", "telefono"}
    assert {o["estado"] for o in d["imagen"]["objetos_verificados"]} == {"confirmada"}
    assert d["prediccion"]["integracion"]["fuentes_recibidas"] == ["video", "imagen"]
    assert d["imagen"]["calidad"]["filtro_usado"]
    assert base64.b64decode(d["imagen"]["mejorada_b64"])[:2] == b"\xff\xd8"  # JPEG


def test_sin_detector_responde_503_claro(cliente, monkeypatch):
    monkeypatch.setenv("INTEGRADO_DETECTOR", "off")
    servicio.establecer_detector(None)
    r = cliente.post("/api/integrado/analizar", json={"imagen_b64": a_base64(imagen_sintetica())})
    assert r.status_code == 503 and r.json()["estado"] == "detector_no_disponible"


@pytest.mark.parametrize("cuerpo", [
    {}, {"imagen_b64": ""}, {"imagen_b64": "esto-no-es-una-imagen"},
    {"imagen_b64": "AAAA", "timestamp": "x"}, {"imagen_b64": "AAAA", "objetos_permitidos": "libro"},
])
def test_entradas_invalidas_dan_422(cliente, detector_falso, cuerpo):
    assert cliente.post("/api/integrado/analizar", json=cuerpo).status_code == 422


def test_cuerpo_no_json_y_no_objeto(cliente, detector_falso):
    assert cliente.post("/api/integrado/analizar", content=b"no json").status_code == 422
    assert cliente.post("/api/integrado/analizar", json=[1, 2]).status_code == 422


def test_limite_de_tamano(cliente, detector_falso):
    r = cliente.post("/api/integrado/analizar", content=b"x" * (8 * 1024 * 1024 + 1))
    assert r.status_code == 413
