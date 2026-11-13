"""El puente debe producir exactamente lo que el adaptador de PREDICCION acepta."""
import pytest

from backend.modulos.imagen.adapter import adaptar_imagen
from backend.modulos.video.adapter import adaptar_video
from backend.modulos.prediccion.integracion import integrar_fuentes
from ensamble import puente

VIDEO = [
    {"clase": "persona", "confianza": 0.9, "bbox": [0, 0, 100, 200]},
    {"clase": "telefono", "confianza": 0.5, "bbox": [200, 50, 260, 120]},
    {"clase": "reloj", "confianza": 0.4, "bbox": [300, 300, 330, 330]},   # la mejora lo pierde
]
MEJORADA = [
    {"clase": "persona", "confianza": 0.95, "bbox": [2, 2, 102, 198]},     # confirmada
    {"clase": "telefono", "confianza": 0.8, "bbox": [202, 52, 258, 118]},  # confirmada
    {"clase": "libro", "confianza": 0.7, "bbox": [400, 400, 480, 460]},    # recuperada (VIDEO no lo veia)
]


def test_estados_confirmada_y_recuperada():
    payload = puente.construir_payload(3, 12.5, 40.0, VIDEO, MEJORADA)
    objetos = {o["clase"]: o for o in payload["imagen"]["frames"][0]["objetos"]}
    assert objetos["persona"]["estado"] == "confirmada"
    assert objetos["telefono"]["estado"] == "confirmada"
    assert objetos["telefono"]["confianza_video"] == 0.5
    assert objetos["libro"]["estado"] == "recuperada"
    assert objetos["libro"]["confianza_video"] == 0.0
    assert "reloj" not in objetos


def test_clase_distinta_o_iou_bajo_no_se_confirma():
    video = [{"clase": "telefono", "confianza": 0.6, "bbox": [0, 0, 50, 50]}]
    lejos = [{"clase": "telefono", "confianza": 0.7, "bbox": [300, 300, 350, 350]}]
    otra = [{"clase": "reloj", "confianza": 0.7, "bbox": [0, 0, 50, 50]}]
    assert puente.construir_payload(0, 0, 40, video, lejos)["imagen"]["frames"][0]["objetos"][0]["estado"] == "recuperada"
    assert puente.construir_payload(0, 0, 40, video, otra)["imagen"]["frames"][0]["objetos"][0]["estado"] == "recuperada"


def test_una_deteccion_de_video_no_confirma_dos_veces():
    video = [{"clase": "telefono", "confianza": 0.6, "bbox": [0, 0, 50, 50]}]
    mejorada = [{"clase": "telefono", "confianza": 0.9, "bbox": [0, 0, 50, 50]},
                {"clase": "telefono", "confianza": 0.8, "bbox": [1, 1, 51, 51]}]
    estados = sorted(o["estado"] for o in puente.construir_payload(0, 0, 40, video, mejorada)["imagen"]["frames"][0]["objetos"])
    assert estados == ["confirmada", "recuperada"]


@pytest.mark.parametrize("contraste,esperado", [(0, 0.0), (32, 0.5), (64, 1.0), (200, 1.0)])
def test_calidad_normalizada_acotada(contraste, esperado):
    assert puente.calidad_normalizada(contraste) == esperado


def test_los_adaptadores_de_prediccion_aceptan_el_payload():
    payload = puente.construir_payload(3, 12.5, 40.0, VIDEO, MEJORADA)
    eventos_video = adaptar_video(payload["video"])
    eventos_imagen = adaptar_imagen(payload["imagen"])
    assert {e["senal"] for e in eventos_video} >= {"persona", "telefono", "reloj"}
    assert {e["senal"] for e in eventos_imagen} == {"persona", "telefono", "libro"}
    assert all(0 <= e["confianza"] <= 1 and 0 <= e["calidad"] <= 1 for e in eventos_video + eventos_imagen)


def test_prediccion_integra_ambas_fuentes_y_marca_objeto_no_permitido():
    payload = puente.construir_payload(3, 12.5, 40.0, VIDEO, MEJORADA, objetos_permitidos=["libro"])
    resultado = integrar_fuentes(payload)
    assert resultado["integracion"]["fuentes_recibidas"] == ["video", "imagen"]
    assert resultado["integracion"]["eventos_generados"] > 0
