"""S04: transformaciones geométricas y robustez del detector (sin cargar el modelo de VIDEO)."""
import cv2
import numpy as np
import pytest

from imagen import config, metricas, transformaciones as t


def escena():
    f = np.zeros((200, 300, 3), np.uint8)
    f[50:110, 80:180] = 255          # un rectángulo blanco
    return f


def detector_blanco(frame):
    """Detector de prueba: devuelve la caja del bloque blanco. Permite verificar que las cajas se transforman bien."""
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gris > 128)
    if len(xs) < 200:
        return []
    return [{"clase": "bloque", "confianza": 0.9,
             "bbox": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]}]


def test_trasladar_mueve_el_contenido():
    img, M = t.trasladar(escena(), 40, 20)
    assert img[50 + 20 + 5, 80 + 40 + 5].all() and not img[55, 85].any()
    assert img.shape == escena().shape and M[0, 2] == 40 and M[1, 2] == 20


def test_rotar_90_equivale_a_rot90_en_imagen_cuadrada():
    cuadrada = np.zeros((100, 100, 3), np.uint8)
    cuadrada[10:40, 10:70] = 200
    img, _ = t.rotar(cuadrada, 90)
    assert np.mean(np.abs(img.astype(int) - np.rot90(cuadrada).astype(int))) < 3


def test_escalar_reduce_y_amplia_el_objeto():
    base = (escena()[..., 0] > 128).sum()
    assert (t.escalar(escena(), 0.5)[0][..., 0] > 128).sum() < base
    assert (t.escalar(escena(), 1.2)[0][..., 0] > 128).sum() > base
    with pytest.raises(ValueError):
        t.escalar(escena(), 0)


def test_voltear_es_involutivo():
    f = escena()
    assert np.array_equal(t.voltear(t.voltear(f)[0])[0], f)


@pytest.mark.parametrize("tr", config.TRANSFORMACIONES, ids=lambda tr: tr["nombre"])
def test_cajas_siguen_al_objeto_transformado(tr):
    f = escena()
    img, M = t.aplicar(f, tr)
    caja_ref = detector_blanco(f)[0]["bbox"]
    (_, caja), = t.transformar_cajas([caja_ref], M, f.shape)
    assert metricas.iou(caja, detector_blanco(img)[0]["bbox"]) > 0.7


def test_caja_fuera_del_encuadre_se_descarta():
    M = np.array([[1, 0, 1000], [0, 1, 0]], np.float64)
    assert t.transformar_cajas([[10, 10, 50, 50]], M, (200, 300, 3)) == []


def test_evaluar_robustez_con_detector_ideal_recupera_todo():
    filas = t.evaluar_robustez(escena(), detector_blanco, config.TRANSFORMACIONES)
    assert len(filas) == len(config.TRANSFORMACIONES)
    assert all(f["tasa"] == 1.0 for f in filas if f["referencia"])


def test_evaluar_robustez_con_detector_ciego_no_recupera_nada():
    llamadas = []

    def ciego_tras_la_referencia(frame):
        llamadas.append(1)
        return detector_blanco(frame) if len(llamadas) == 1 else []

    filas = t.evaluar_robustez(escena(), ciego_tras_la_referencia, config.TRANSFORMACIONES)
    assert all(f["referencia"] > 0 and f["recuperadas"] == 0 for f in filas)


def test_frame_vacio_o_corrupto_y_transformacion_desconocida():
    for malo in (None, np.zeros((0, 0, 3), np.uint8)):
        with pytest.raises(ValueError):
            t.rotar(malo, 10)
        with pytest.raises(ValueError):
            t.evaluar_robustez(malo, detector_blanco, config.TRANSFORMACIONES)
    with pytest.raises(ValueError):
        t.aplicar(escena(), {"tipo": "inventada"})
