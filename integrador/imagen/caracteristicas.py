"""Extracción de características (S06): bordes, esquinas, ORB y color."""
import cv2
import numpy as np


def densidad_bordes(gris, umbrales):
    bordes = cv2.Canny(gris, *umbrales)
    return float(np.count_nonzero(bordes) / bordes.size)


def contar_esquinas(gris, p):
    respuesta = cv2.cornerHarris(np.float32(gris), p["bloque"], p["apertura"], p["k"])
    return int(np.count_nonzero(respuesta > p["umbral"] * respuesta.max()))


def contar_orb(gris, max_puntos):
    puntos = cv2.ORB_create(nfeatures=max_puntos).detect(gris, None)
    return len(puntos)


def media_hsv(frame):
    """Media de H, S y V normalizada a [0, 1]."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
    media = hsv.mean(axis=0) / np.array([179.0, 255.0, 255.0])
    return [round(float(v), 4) for v in media]


def extraer(frame, n_objetos, cfg):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "n_objetos": n_objetos,
        "densidad_bordes": round(densidad_bordes(gris, cfg.CANNY), 5),
        "n_esquinas": contar_esquinas(gris, cfg.HARRIS),
        "n_keypoints_orb": contar_orb(gris, cfg.ORB_PUNTOS),
        "hist_hsv": media_hsv(frame),
    }
