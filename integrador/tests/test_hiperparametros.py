"""S14: búsqueda de hiperparámetros. Las de Keras se omiten si Keras no está instalado (requirements-lab.txt)."""
import importlib.util
import inspect

import numpy as np
import pytest

from imagen import config, hiperparametros as h, mlp

HAY_KERAS = importlib.util.find_spec("keras") is not None
con_keras = pytest.mark.skipif(not HAY_KERAS, reason="Keras no está instalado (requirements-lab.txt)")

ESPACIO = {"capas": [(8,), (16,)], "dropout": [0.0, 0.3], "l2": [0.0, 1e-4], "tasa": [1e-3, 3e-3]}


def test_muestrear_es_reproducible_sin_repetidos_y_dentro_del_espacio():
    a = h.muestrear(ESPACIO, 5, semilla=1)
    assert a == h.muestrear(ESPACIO, 5, semilla=1)
    assert len({str(c) for c in a}) == 5
    assert all(c["capas"] in ESPACIO["capas"] and c["tasa"] in ESPACIO["tasa"] for c in a)
    assert h.muestrear(ESPACIO, 5, semilla=2) != a


def test_muestrear_pide_mas_de_lo_que_hay_y_devuelve_todo():
    assert len(h.muestrear(ESPACIO, 1000, semilla=1)) == 2 * 2 * 2 * 2


def test_efecto_marginal_promedia_por_valor():
    res = [{"hp": {"d": 0.0}, "puntaje": 0.8}, {"hp": {"d": 0.0}, "puntaje": 0.6}, {"hp": {"d": 0.3}, "puntaje": 0.9}]
    assert h.efecto_marginal(res, "d") == {"0.0": 0.7, "0.3": 0.9}


def test_la_busqueda_no_recibe_datos_de_prueba():
    """Garantía de diseño: ninguna función de selección tiene un parámetro de prueba."""
    for f in (h.buscar, h.evaluar_configuracion, h.entrenar_validando):
        assert not [p for p in inspect.signature(f).parameters if "prueba" in p or p in ("xp", "yp", "xt", "yt")]


def datos_sinteticos(n=360, semilla=0):
    rng = np.random.default_rng(semilla)
    y = rng.integers(0, 2, n)
    x = rng.normal(0.3, 0.05, (n, 8, 8)).astype(np.float32)
    x[y == 1, :4, :] += 0.5
    x = np.clip(x, 0, 1)
    return {"xf": x[:240], "yf": y[:240], "xv": x[240:], "yv": y[240:], "n_clases": 2}


def construir(hp, semilla):
    return h.construir_mlp_hp(hp, semilla, (8, 8), 2)


@con_keras
def test_evaluar_configuracion_devuelve_las_metricas_esperadas():
    r = h.evaluar_configuracion(construir, h.muestrear(ESPACIO, 1, 0)[0], datos_sinteticos(), [1, 2], grados=(0.0,))
    assert {"puntaje", "desviacion", "brecha_sobreajuste", "epocas", "segundos", "hp"} <= set(r)
    assert 0.0 <= r["puntaje"] <= 1.0 and r["desviacion"] >= 0 and r["epocas"] >= 1


@con_keras
def test_buscar_ordena_de_mejor_a_peor_y_prefiere_una_configuracion_que_aprende():
    res = h.buscar(construir, ESPACIO, datos_sinteticos(), n_configuraciones=4, semillas=[0, 1])
    puntajes = [r["puntaje"] for r in res]
    assert len(res) == 4 and puntajes == sorted(puntajes, reverse=True)
    assert puntajes[0] > 0.8                                  # el problema es separable: la mejor debe aprenderlo


@con_keras
def test_parada_temprana_se_puede_apagar():
    d = datos_sinteticos()
    m1 = mlp.construir_mlp((8, 8), 2, capas=(8,), semilla=1)
    corto = mlp.entrenar(m1, d["xf"], d["yf"], epocas=25, paciencia=1)
    m2 = mlp.construir_mlp((8, 8), 2, capas=(8,), semilla=1)
    completo = mlp.entrenar(m2, d["xf"], d["yf"], epocas=25, parada_temprana=False)
    assert len(completo["loss"]) == 25 and len(corto["loss"]) <= 25
