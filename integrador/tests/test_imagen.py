"""Pruebas unitarias del módulo IMAGEN (sin cargar el modelo de VIDEO)."""
import numpy as np
import pytest

from imagen import caracteristicas, config, mejoramiento, metricas, pipeline, segmentacion


def frame_sintetico():
    f = np.full((120, 160, 3), 40, np.uint8)
    f[30:90, 40:120] = 200
    return f


def test_degradar_oscurece():
    f = frame_sintetico()
    assert mejoramiento.degradar(f, 0.3, 10, 1).mean() < f.mean()


def test_gamma_sube_brillo():
    oscuro = (frame_sintetico() * 0.3).astype(np.uint8)
    claro, gamma = mejoramiento.correccion_gamma(oscuro, 0.45)
    assert claro.mean() > oscuro.mean() and gamma < 1


def test_segmentacion_encuentra_rectangulo():
    objetos, _ = segmentacion.segmentar(frame_sintetico(), 500, 10, 5)
    assert len(objetos) >= 1 and objetos[0]["area"] > 3000


def test_caracteristicas_claves():
    r = caracteristicas.extraer(frame_sintetico(), 1, config)
    assert set(r) == {"n_objetos", "densidad_bordes", "n_esquinas", "n_keypoints_orb", "hist_hsv"}


def test_iou_y_recuperadas():
    a = [{"clase": "persona", "bbox": [0, 0, 10, 10]}]
    assert metricas.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert metricas.recuperadas(a, a, 0.5) == 1
    assert metricas.recuperadas(a, [{"clase": "mochila", "bbox": [0, 0, 10, 10]}], 0.5) == 0


def test_pipeline_contrato_y_frame_corrupto():
    entrada = {"frame_id": 3, "timestamp": 1.5, "fps": 2.0, "fuente": "x", "frame": frame_sintetico()}
    registro, _, _ = pipeline.procesar_frame(entrada, lambda f: [])
    assert registro["version_contrato"] == "0.1" and registro["timestamp"] == 1.5
    with pytest.raises(ValueError):
        pipeline.procesar_frame({**entrada, "frame": None}, lambda f: [])


def test_pipeline_produccion_sin_referencia():
    entrada = {"frame_id": 1, "timestamp": 0.5, "fps": 2.0, "fuente": "x", "frame": frame_sintetico()}
    registro, mejorado, dets = pipeline.procesar_frame_produccion(entrada, lambda f: [])
    assert registro["calidad"]["psnr"] is None and registro["deteccion_video"]["n"] == 0
    assert mejorado.shape[2] == 3 and dets == []


def test_mejora_adaptativa_solo_si_hace_falta():
    normal = (np.random.default_rng(0).normal(128, 60, (120, 160, 3))).clip(0, 255).astype(np.uint8)
    _, _, _, pasos = mejoramiento.mejorar_adaptativo(normal, config)
    assert "gamma" not in pasos                       # imagen con buen brillo: no se aclara
    oscuro = mejoramiento.degradar(frame_sintetico(), 0.2, 15, 1)
    mejorada, antes, despues, pasos = mejoramiento.mejorar_adaptativo(oscuro, config)
    assert "gamma" in pasos and despues["brillo"] > antes["brillo"]


def test_recorte_y_ampliacion_de_alerta():
    from imagen import alerta_mejorada
    frame = frame_sintetico()
    recorte = alerta_mejorada.recorte_con_margen(frame, [40, 30, 120, 90])
    assert recorte.shape[0] >= 60 and recorte.shape[1] >= 80
    assert max(alerta_mejorada.ampliar(recorte, 320).shape[:2]) >= 320
    # un bbox pegado al borde no debe salirse de la imagen
    assert alerta_mejorada.recorte_con_margen(frame, [0, 0, 20, 20]).size > 0
