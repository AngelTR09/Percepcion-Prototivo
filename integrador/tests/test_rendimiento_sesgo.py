"""S15: rendimiento por etapa y sesgo por condición (sin cargar el modelo de VIDEO)."""
import numpy as np
import pytest

from imagen import config, rendimiento, sesgo


def frame():
    f = np.full((270, 480, 3), 60, np.uint8)
    f[80:200, 100:300] = 190
    return f


def detector_fijo(frame_):
    return [{"clase": "persona", "confianza": 0.9, "bbox": [100.0, 80.0, 300.0, 200.0]}]


# ---- rendimiento ----

def test_medir_etapas_con_y_sin_analisis():
    ms, dets, forma = rendimiento.medir_etapas(frame(), detector_fijo, 320, analisis=True)
    assert set(ms) == set(rendimiento.ETAPAS) and all(v >= 0 for v in ms.values())
    assert forma[1] == 320 and len(dets) == 1
    ms2, _, _ = rendimiento.medir_etapas(frame(), detector_fijo, 320, analisis=False)
    assert "segmentar" not in ms2 and "caracteristicas" not in ms2


def test_medir_etapas_frame_vacio_o_corrupto():
    for malo in (None, np.zeros((0, 0, 3), np.uint8)):
        with pytest.raises(ValueError):
            rendimiento.medir_etapas(malo, detector_fijo, 320)


def test_resumir_calcula_total_y_fps():
    r = rendimiento.resumir([{"gamma": 10.0, "detector": 90.0}, {"gamma": 10.0, "detector": 90.0}])
    assert r["ms_total"] == 100.0 and r["fps"] == 10.0
    assert r["ms_por_etapa"]["clahe"] == 0.0
    assert not rendimiento.es_tiempo_real(r["fps"]) and rendimiento.es_tiempo_real(30.0)
    with pytest.raises(ValueError):
        rendimiento.resumir([])


# ---- sesgo ----

def det(clase, x1, y1, x2, y2):
    return {"clase": clase, "confianza": 0.9, "bbox": [x1, y1, x2, y2]}


def test_grupos_de_tamano_y_region():
    forma = (1000, 1000)
    assert sesgo.grupo_tamano(det("a", 0, 0, 50, 50), forma) == "pequeno"      # 0.25 %
    assert sesgo.grupo_tamano(det("a", 0, 0, 150, 150), forma) == "mediano"    # 2.25 %
    assert sesgo.grupo_tamano(det("a", 0, 0, 400, 400), forma) == "grande"     # 16 %
    assert sesgo.grupo_region(det("a", 450, 450, 550, 550), forma) == "centro"
    assert sesgo.grupo_region(det("a", 0, 0, 100, 100), forma) == "borde"


def test_recuperadas_por_referencia_no_reutiliza_candidatas():
    ref = [det("a", 0, 0, 10, 10), det("a", 0, 0, 10, 10)]
    assert sesgo.recuperadas_por_referencia(ref, [det("a", 0, 0, 10, 10)], 0.5) == [True, False]
    assert sesgo.recuperadas_por_referencia(ref, [det("b", 0, 0, 10, 10)], 0.5) == [False, False]


def test_acumular_y_brecha_detectan_un_grupo_desfavorecido():
    forma = (1000, 1000)
    tabla = {}
    grandes = [det("persona", 100 + i, 100, 400 + i, 400) for i in range(12)]     # grandes y al centro/borde
    chicos = [det("mochila", 10 + 60 * i, 900, 40 + 60 * i, 930) for i in range(12)]
    # el detector "degradado" acierta todos los grandes y ningún chico; el "mejorado" recupera la mitad de los chicos
    degradado = list(grandes)
    mejorado = list(grandes) + chicos[:6]
    sesgo.acumular(tabla, grandes + chicos, {"degradado": degradado, "mejorado": mejorado}, forma, 0.5)
    assert tabla["tamano"]["grande"] == {"n": 12, "degradado": 12, "mejorado": 12}
    assert tabla["tamano"]["pequeno"] == {"n": 12, "degradado": 0, "mejorado": 6}
    assert sesgo.brecha(tabla["tamano"], "degradado") == 1.0
    assert sesgo.brecha(tabla["tamano"], "mejorado") == 0.5      # la mejora reduce la brecha
    assert sesgo.brecha(tabla["clase"], "mejorado") == 0.5


def test_brecha_ignora_grupos_con_pocas_referencias():
    tabla = {"a": {"n": 50, "v": 50}, "b": {"n": 2, "v": 0}}
    assert sesgo.brecha(tabla, "v") is None                       # b tiene n < min_por_grupo
    assert sesgo.brecha(tabla, "v", minimo=1) == 1.0
    assert set(sesgo.tasas(tabla, "v")) == {"a"}
