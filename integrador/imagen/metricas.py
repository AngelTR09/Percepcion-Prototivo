"""Métricas de calidad (S03) y de comparación de detecciones (RF-09)."""
import time

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(referencia, imagen):
    return float(peak_signal_noise_ratio(referencia, imagen, data_range=255))


def ssim(referencia, imagen):
    return float(structural_similarity(referencia, imagen, channel_axis=2, data_range=255))


def contraste(imagen):
    """Desviación estándar de la luminancia."""
    return float(np.std(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)))


def cronometrar(funcion, *args):
    """Devuelve (resultado, milisegundos)."""
    inicio = time.perf_counter()
    resultado = funcion(*args)
    return resultado, (time.perf_counter() - inicio) * 1000


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def recuperadas(referencia, candidatas, iou_minimo):
    """Cuántas detecciones de referencia reaparecen (misma clase, IoU >= mínimo)."""
    usadas, aciertos = set(), 0
    for r in referencia:
        for i, c in enumerate(candidatas):
            if i not in usadas and c["clase"] == r["clase"] and iou(r["bbox"], c["bbox"]) >= iou_minimo:
                usadas.add(i)
                aciertos += 1
                break
    return aciertos
