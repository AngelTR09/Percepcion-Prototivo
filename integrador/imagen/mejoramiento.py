"""Mejora de contraste e iluminación (S04) y degradación sintética para la prueba."""
import cv2
import numpy as np


def degradar(frame, factor_luz, sigma_ruido, semilla):
    """Simula poca luz y ruido del sensor. Genera la referencia para medir la mejora."""
    rng = np.random.default_rng(semilla)
    oscuro = frame.astype(np.float32) * factor_luz
    ruido = rng.normal(0, sigma_ruido * factor_luz + 2, frame.shape)
    return np.clip(oscuro + ruido, 0, 255).astype(np.uint8)


def correccion_gamma(frame, brillo_objetivo):
    """Gamma automática: lleva la luminancia media hacia brillo_objetivo."""
    media = max(float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) / 255.0, 1e-3)
    gamma = float(np.clip(np.log(brillo_objetivo) / np.log(media), 0.3, 1.0))
    tabla = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(frame, tabla), gamma


def clahe(frame, clip, grilla):
    """CLAHE sobre el canal L de LAB para no alterar el color."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grilla, grilla)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def ecualizar_histograma(frame):
    """Ecualización global sobre el canal Y (alternativa a CLAHE para comparar)."""
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def mejorar(frame, cfg_clahe, brillo_objetivo):
    """Gamma + CLAHE. Devuelve (imagen, gamma_usada)."""
    corregido, gamma = correccion_gamma(frame, brillo_objetivo)
    return clahe(corregido, cfg_clahe["clip"], cfg_clahe["grilla"]), gamma
