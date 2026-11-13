"""S12: MLP. Las funciones de evaluación y de partición se prueban siempre; las de Keras solo si Keras está instalado
(entorno del laboratorio, `requirements-lab.txt`)."""
import importlib.util

import numpy as np
import pytest

from imagen import config, mlp

HAY_KERAS = importlib.util.find_spec("keras") is not None
con_keras = pytest.mark.skipif(not HAY_KERAS, reason="Keras no está instalado (requirements-lab.txt)")


# ---- evaluación y partición (NumPy) ----

def test_matriz_confusion_y_metricas():
    real = [0, 0, 0, 1, 1, 2]
    pred = [0, 0, 1, 1, 1, 0]
    m = mlp.matriz_confusion(real, pred, 3)
    assert m.tolist() == [[2, 1, 0], [0, 2, 0], [1, 0, 0]]
    r = mlp.metricas_clasificacion(m)
    assert r["exactitud"] == pytest.approx(4 / 6)
    assert r["recall"] == [pytest.approx(2 / 3, abs=1e-3), 1.0, 0.0]
    assert r["precision"][0] == pytest.approx(2 / 3, abs=1e-3) and r["precision"][2] == 0.0   # clase 2 nunca predicha
    assert r["soporte"] == [3, 2, 1] and r["f1"][2] == 0.0


def test_metricas_con_matriz_vacia_no_fallan():
    assert mlp.metricas_clasificacion(np.zeros((2, 2)))["exactitud"] == 0.0


def test_linea_base_mayoritaria():
    assert mlp.linea_base_mayoritaria([0, 0, 0, 1], [0, 0, 1, 1]) == 0.5


def test_dividir_por_tiempo_no_mezcla_frames():
    frames = np.repeat(np.arange(10), 5)            # 10 frames, 5 recortes cada uno
    prueba = mlp.dividir_por_tiempo(frames, 0.3)
    assert set(frames[prueba]) == {7, 8, 9} and set(frames[~prueba]) == set(range(7))
    assert not set(frames[prueba]) & set(frames[~prueba])
    with pytest.raises(ValueError):
        mlp.dividir_por_tiempo(np.zeros(10), 0.3)   # un solo grupo: no se puede dividir


def test_recortes_del_frame():
    from imagen import ejecutar_recortes as r
    frame = np.full((200, 300, 3), 80, np.uint8)
    dets = [{"clase": "persona", "bbox": [10, 10, 110, 150]}, {"clase": "mochila", "bbox": [200, 100, 260, 190]},
            {"clase": "persona", "bbox": [0, 0, 10, 10]},          # muy chico
            {"clase": "telefono", "bbox": [50, 50, 120, 120]}]     # clase fuera del experimento
    salida = r.recortes_del_frame(frame, dets, ["persona", "mochila"], 32, 24)
    assert [c for _, c in salida] == [0, 1] and salida[0][0].shape == (32, 32, 3)


# ---- Keras ----

@con_keras
def test_construir_mlp_forma_y_parametros():
    m = mlp.construir_mlp((28, 28), 10, capas=(16, 8), dropout=0.0)
    assert m.output_shape == (None, 10)
    assert m.count_params() == 784 * 16 + 16 + 16 * 8 + 8 + 8 * 10 + 10


@con_keras
def test_mlp_aprende_un_problema_separable_y_supera_la_linea_base():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    x = rng.normal(0, 0.3, (600, 8, 8)).astype(np.float32)
    x[y == 1, :4, :] += 1.0                          # la clase 1 tiene la mitad superior más clara
    m = mlp.construir_mlp((8, 8), 2, capas=(16,), semilla=1)
    historia = mlp.entrenar(m, x[:450], y[:450], epocas=40)
    metricas, matriz = mlp.evaluar(m, x[450:], y[450:], 2)
    assert metricas["exactitud"] > 0.95 and matriz.sum() == 150
    assert historia["loss"][-1] < historia["loss"][0]


@con_keras
def test_misma_semilla_misma_prediccion():
    x = np.random.default_rng(3).random((20, 4, 4)).astype(np.float32)
    a = mlp.construir_mlp((4, 4), 3, capas=(8,), semilla=5).predict(x, verbose=0)
    b = mlp.construir_mlp((4, 4), 3, capas=(8,), semilla=5).predict(x, verbose=0)
    assert np.allclose(a, b)


@con_keras
def test_parada_temprana_corta_el_entrenamiento():
    rng = np.random.default_rng(0)
    x = rng.random((200, 4, 4)).astype(np.float32)
    y = rng.integers(0, 2, 200)                      # etiquetas al azar: la validación no mejora
    m = mlp.construir_mlp((4, 4), 2, capas=(16,), semilla=2)
    historia = mlp.entrenar(m, x, y, epocas=200, paciencia=3)
    assert len(historia["loss"]) < 200


def test_sin_keras_el_error_es_claro(monkeypatch):
    if HAY_KERAS:
        pytest.skip("Keras está instalado")
    with pytest.raises(RuntimeError, match="requirements-lab"):
        mlp.construir_mlp((4, 4), 2)
