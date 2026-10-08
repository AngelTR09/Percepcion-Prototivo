"""Color, normalización de tamaño y reducción de ruido (S02-S03)."""
import cv2

from . import metricas


def a_grises(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def a_hsv(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


def normalizar_tamano(frame, ancho):
    alto = int(frame.shape[0] * ancho / frame.shape[1])
    return cv2.resize(frame, (ancho, alto), interpolation=cv2.INTER_AREA)


def aplicar_filtro(frame, nombre, p):
    if nombre == "media":
        return cv2.blur(frame, (p["k"], p["k"]))
    if nombre == "gaussiano":
        return cv2.GaussianBlur(frame, (p["k"], p["k"]), p["sigma"])
    if nombre == "mediana":
        return cv2.medianBlur(frame, p["k"])
    if nombre == "bilateral":
        return cv2.bilateralFilter(frame, p["d"], p["sigma_color"], p["sigma_espacio"])
    raise ValueError(f"Filtro desconocido: {nombre}")


def comparar_filtros(frame_ruidoso, referencia, filtros):
    """Aplica cada filtro y mide PSNR/SSIM contra la referencia y el tiempo (RF-03).
    Devuelve (mejor_nombre, mejor_imagen, tabla)."""
    tabla, imagenes = {}, {}
    for nombre, params in filtros.items():
        imagen, ms = metricas.cronometrar(aplicar_filtro, frame_ruidoso, nombre, params)
        imagenes[nombre] = imagen
        tabla[nombre] = {
            "psnr": metricas.psnr(referencia, imagen),
            "ssim": metricas.ssim(referencia, imagen),
            "ms": ms,
        }
    mejor = max(tabla, key=lambda n: tabla[n]["ssim"])
    return mejor, imagenes[mejor], tabla
