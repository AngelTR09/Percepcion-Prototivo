"""S13: CNN y transfer learning. Todo lo de Keras se omite si Keras no está instalado (requirements-lab.txt).
El transfer learning se prueba con pesos al azar (`pesos=None`): no hay descarga de red en las pruebas."""
import importlib.util

import numpy as np
import pytest

from imagen import cnn, config, mlp

HAY_KERAS = importlib.util.find_spec("keras") is not None
con_keras = pytest.mark.skipif(not HAY_KERAS, reason="Keras no está instalado (requirements-lab.txt)")


def parches(n=240, semilla=0):
    """Imágenes 16x16 RGB: la clase 1 tiene un cuadrado brillante en una posición variable (la CNN debe hallarlo)."""
    rng = np.random.default_rng(semilla)
    y = rng.integers(0, 2, n)
    x = rng.normal(0.3, 0.05, (n, 16, 16, 3)).astype(np.float32)
    for i in np.nonzero(y == 1)[0]:
        r, c = rng.integers(0, 11, 2)
        x[i, r:r + 5, c:c + 5] += 0.5
    return np.clip(x, 0, 1), y


@con_keras
def test_cnn_forma_y_parametros():
    m = cnn.construir_cnn((16, 16, 3), 2, filtros=(4, 8), denso=6)
    assert m.output_shape == (None, 2)
    esperado = (3 * 3 * 3 * 4 + 4) + (3 * 3 * 4 * 8 + 8) + (4 * 4 * 8 * 6 + 6) + (6 * 2 + 2)
    assert m.count_params() == esperado


@con_keras
def test_cnn_aprende_a_encontrar_el_objeto_donde_este():
    x, y = parches()
    m = cnn.construir_cnn((16, 16, 3), 2, filtros=(8, 8), denso=16, semilla=1)
    mlp.entrenar(m, x[:180], y[:180], epocas=40)
    metricas, _ = mlp.evaluar(m, x[180:], y[180:], 2)
    assert metricas["exactitud"] > 0.9


@con_keras
def test_aumento_solo_actua_al_entrenar():
    x, _ = parches(8)
    m = cnn.construir_cnn((16, 16, 3), 2, filtros=(4,), denso=4, aumento=True, semilla=3)
    a, b = m.predict(x, verbose=0), m.predict(x, verbose=0)
    assert np.allclose(a, b)                               # en inferencia es determinista
    assert any("random" in c.name for c in m.layers)


@con_keras
def test_transfer_congela_la_base_y_entrena_solo_la_cabeza():
    m = cnn.construir_transfer(2, lado=32, pesos=None)
    entrenables, congelados = cnn.parametros(m)
    assert entrenables == 1280 * 2 + 2                     # solo la capa Dense final
    assert congelados > 2_000_000
    assert m.output_shape == (None, 2)


@con_keras
def test_afinar_descongela_las_ultimas_capas_sin_tocar_batchnorm():
    import keras
    m = cnn.construir_transfer(2, lado=32, pesos=None)
    antes = cnn.parametros(m)[0]
    cnn.afinar(m, n_capas=20)
    despues = cnn.parametros(m)[0]
    assert despues > antes
    base = next(c for c in m.layers if isinstance(c, keras.Model))
    assert not any(c.trainable for c in base.layers if isinstance(c, keras.layers.BatchNormalization))
    assert all(not c.trainable for c in base.layers[:-20])


@con_keras
def test_transfer_pasa_una_imagen_de_prueba():
    m = cnn.construir_transfer(2, lado=32, pesos=None)
    salida = m.predict(np.random.default_rng(0).random((3, 32, 32, 3)).astype(np.float32), verbose=0)
    assert salida.shape == (3, 2) and np.allclose(salida.sum(axis=1), 1.0, atol=1e-5)


def test_rotar_lote_con_cero_grados_no_cambia_y_con_giro_si():
    x = np.zeros((2, 16, 16, 3), np.float32)
    x[:, 2:6, 2:12] = 1.0
    assert np.array_equal(cnn.rotar_lote(x, 0), x)
    girado = cnn.rotar_lote(x, 45)
    assert girado.shape == x.shape and not np.allclose(girado, x)
    assert girado.dtype == np.float32 and 0.0 <= girado.min() and girado.max() <= 1.0


def test_sin_keras_el_error_es_claro():
    if HAY_KERAS:
        pytest.skip("Keras está instalado")
    with pytest.raises(RuntimeError, match="requirements-lab"):
        cnn.construir_cnn((16, 16, 3), 2)
