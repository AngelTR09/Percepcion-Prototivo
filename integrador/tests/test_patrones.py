"""S06: Sobel vs Canny y reconocimiento de patrones (ORB + BFMatcher vs matchTemplate), con verdad conocida."""
import cv2
import numpy as np
import pytest

from imagen import caracteristicas as c, config, metricas, transformaciones


def escena_texturada(semilla=1, ancho=480, alto=360):
    """Muchas figuras de colores al azar: abundan las esquinas, así ORB tiene de dónde agarrarse."""
    rng = np.random.default_rng(semilla)
    img = np.full((alto, ancho, 3), 90, np.uint8)
    for _ in range(140):
        color = tuple(int(v) for v in rng.integers(0, 256, 3))
        x, y = int(rng.integers(0, ancho)), int(rng.integers(0, alto))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(8, 40)), y + int(rng.integers(8, 40))), color, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(5, 22)), color, -1)
    # textura fina (ruido suavizado): las fotos reales no son colores planos y sin ella los descriptores no se distinguen
    ruido = cv2.GaussianBlur(rng.normal(0, 1, (alto, ancho, 3)).astype(np.float32), (0, 0), 1.2)
    return np.clip(img + 45 * ruido / ruido.std(), 0, 255).astype(np.uint8)


def gris(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---- Sobel ----

def test_sobel_detecta_el_borde_vertical_y_no_inventa_en_imagen_plana():
    g = np.zeros((60, 100), np.uint8)
    g[:, 50:] = 200
    b = c.bordes_sobel(g, 3, 0.2)
    columnas = np.nonzero(b.any(axis=0))[0]
    assert columnas.min() >= 48 and columnas.max() <= 51
    assert not c.bordes_sobel(np.full((60, 100), 90, np.uint8), 3, 0.2).any()


def test_comparar_bordes_canny_y_sobel_coinciden_en_un_escalon():
    g = np.zeros((80, 120), np.uint8)
    g[:, 60:] = 200
    r = c.comparar_bordes(g, config.CANNY, config.SOBEL)
    assert r["canny_cubierto_por_sobel"] > 0.9 and r["sobel_cubierto_por_canny"] > 0.5
    assert r["densidad_canny"] > 0 and r["densidad_sobel"] > 0 and r["ms_sobel"] >= 0


def test_comparar_bordes_imagen_plana_no_falla():
    r = c.comparar_bordes(np.full((50, 50), 90, np.uint8), config.CANNY, config.SOBEL)
    assert r["densidad_canny"] == 0 and r["canny_cubierto_por_sobel"] == 1.0


# ---- ORB, BFMatcher y plantillas ----

def test_orb_devuelve_descriptores_y_se_empareja_consigo_mismo():
    g = gris(escena_texturada())
    kp, des = c.descriptores_orb(g, 500)
    assert len(kp) == len(des) > 50 and des.shape[1] == 32
    assert len(c.emparejar_orb(des, des, 0.75)) > 0.5 * len(des)


def test_emparejar_orb_sin_descriptores():
    assert c.emparejar_orb(None, None, 0.75) == []
    assert c.emparejar_orb(np.zeros((1, 32), np.uint8), np.zeros((5, 32), np.uint8), 0.75) == []


def caja_y_parche(img):
    x1, y1, x2, y2 = 150, 100, 270, 220
    return [x1, y1, x2, y2], gris(img[y1:y2, x1:x2])


def test_orb_localiza_el_parche_aunque_la_escena_este_rotada_o_ampliada():
    img = escena_texturada()
    caja, parche = caja_y_parche(img)
    for t in ({"tipo": "rotar", "grados": 20.0}, {"tipo": "escalar", "factor": 0.9}, {"tipo": "escalar", "factor": 1.25}, {"tipo": "rotar", "grados": -15.0, "escala": 1.1}):
        movida, M = transformaciones.aplicar(img, t)
        (_, esperada), = transformaciones.transformar_cajas([caja], M, img.shape)
        hallado = c.localizar_con_orb(parche, gris(movida), config.ORB_EMPAREJAR)
        assert hallado is not None, t
        assert metricas.iou(hallado["bbox"], esperada) > 0.5, t


def test_plantilla_exacta_sin_transformar_y_se_pierde_con_rotacion():
    img = escena_texturada()
    caja, parche = caja_y_parche(img)
    exacto = c.localizar_con_plantilla(parche, gris(img), config.PLANTILLA_UMBRAL)
    assert exacto["puntaje"] > 0.99 and metricas.iou(exacto["bbox"], caja) > 0.95
    rotada, M = transformaciones.rotar(img, 30.0)
    (_, esperada), = transformaciones.transformar_cajas([caja], M, img.shape)
    hallado = c.localizar_con_plantilla(parche, gris(rotada), config.PLANTILLA_UMBRAL)
    assert hallado is None or metricas.iou(hallado["bbox"], esperada) < 0.5


def test_un_patron_ajeno_no_se_encuentra():
    ajeno = gris(escena_texturada(semilla=99))[100:220, 150:270]
    assert c.localizar_con_orb(ajeno, gris(escena_texturada(semilla=1)), config.ORB_EMPAREJAR) is None


def test_plantilla_mas_grande_que_la_escena_da_none():
    assert c.localizar_con_plantilla(np.zeros((50, 50), np.uint8), np.zeros((20, 20), np.uint8), 0.6) is None


def test_entradas_vacias_o_corruptas_levantan_error():
    bueno = gris(escena_texturada())
    for malo in (None, np.zeros((0, 0), np.uint8)):
        with pytest.raises(ValueError):
            c.bordes_sobel(malo, 3, 0.2)
        with pytest.raises(ValueError):
            c.localizar_con_orb(malo, bueno, config.ORB_EMPAREJAR)
        with pytest.raises(ValueError):
            c.localizar_con_plantilla(bueno, malo, 0.6)
        with pytest.raises(ValueError):
            c.comparar_bordes(malo, config.CANNY, config.SOBEL)
