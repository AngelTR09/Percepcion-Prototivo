"""S05: umbral adaptativo, watershed e IoU de máscaras (imágenes sintéticas con verdad conocida)."""
import cv2
import numpy as np
import pytest

from imagen import config, metricas, segmentacion


def escena_con_gradiente():
    """Fondo con iluminación que sube de 30 a 200 y dos cuadrados un poco más claros que su entorno."""
    alto, ancho = 200, 400
    fondo = np.tile(np.linspace(30, 200, ancho, dtype=np.float32), (alto, 1))
    verdad = np.zeros((alto, ancho), np.uint8)
    for x in (40, 300):
        fondo[70:130, x:x + 60] += 45
        verdad[70:130, x:x + 60] = 255
    gris = np.clip(fondo, 0, 255).astype(np.uint8)
    return gris, verdad


def dos_circulos_pegados():
    f = np.zeros((200, 300, 3), np.uint8)
    cv2.circle(f, (100, 100), 45, (255, 255, 255), -1)
    cv2.circle(f, (180, 100), 45, (255, 255, 255), -1)
    return f


def test_adaptativo_supera_a_otsu_con_iluminacion_despareja():
    gris, verdad = escena_con_gradiente()
    k = config.KERNEL_MORFOLOGIA
    otsu = segmentacion.mascara_otsu(gris, k)
    ad = config.SEGMENTACION["adaptativo"]
    adaptativo = segmentacion.mascara_adaptativa(gris, ad["bloque"], ad["c"], k)
    iou_otsu = metricas.iou_mascaras(otsu, verdad)
    iou_adaptativo = metricas.iou_mascaras(adaptativo, verdad)
    assert iou_adaptativo > 0.6 and iou_adaptativo > iou_otsu + 0.3


def test_adaptativo_rechaza_bloque_par_o_muy_chico():
    gris, _ = escena_con_gradiente()
    for bloque in (50, 1):
        with pytest.raises(ValueError):
            segmentacion.mascara_adaptativa(gris, bloque, 7, 5)


def test_watershed_separa_objetos_que_se_tocan_y_otsu_no():
    f = dos_circulos_pegados()
    k, area, maximo = config.KERNEL_MORFOLOGIA, config.AREA_MINIMA, config.MAX_OBJETOS
    con_otsu = segmentacion.extraer_objetos(segmentacion.mascara_otsu(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), k), area, maximo)
    _, etiquetas = segmentacion.segmentar_watershed(f, k, config.SEGMENTACION["watershed"]["fraccion_distancia"])
    con_watershed = segmentacion.objetos_de_etiquetas(etiquetas, area, maximo)
    assert len(con_otsu) == 1
    assert len(con_watershed) == 2
    xs = sorted(o["centroide"][0] for o in con_watershed)
    assert xs[0] == pytest.approx(100, abs=8) and xs[1] == pytest.approx(180, abs=8)


def test_objetos_de_etiquetas_tiene_el_formato_de_extraer_objetos():
    f = dos_circulos_pegados()
    _, etiquetas = segmentacion.segmentar_watershed(f, 5, 0.4)
    o = segmentacion.objetos_de_etiquetas(etiquetas, 500, 10)[0]
    assert set(o) == {"id", "bbox", "area", "perimetro", "centroide"} and o["area"] > 3000


def test_watershed_en_imagen_vacia_no_falla():
    mascara, etiquetas = segmentacion.segmentar_watershed(np.zeros((100, 100, 3), np.uint8), 5, 0.4)
    assert not mascara.any() and not etiquetas.any()
    assert segmentacion.objetos_de_etiquetas(etiquetas, 500, 10) == []


def test_iou_mascaras():
    a = np.zeros((10, 10), np.uint8)
    a[:5] = 255
    b = np.zeros((10, 10), np.uint8)
    b[:, :5] = 255
    assert metricas.iou_mascaras(a, a) == 1.0
    assert metricas.iou_mascaras(a, b) == pytest.approx(25 / 75)
    assert metricas.iou_mascaras(a, np.zeros_like(a)) == 0.0
    assert metricas.iou_mascaras(np.zeros_like(a), np.zeros_like(a)) == 1.0
    with pytest.raises(ValueError):
        metricas.iou_mascaras(a, np.zeros((5, 5), np.uint8))


def test_mascara_de_cajas_recorta_al_encuadre():
    m = metricas.mascara_de_cajas((20, 30, 3), [[5, 5, 15, 10], [-5, -5, 3, 3], [25, 15, 50, 50]])
    assert m[5:10, 5:15].all() and m[0:3, 0:3].all() and m[15:20, 25:30].all()
    assert (m > 0).sum() == 50 + 9 + 25
