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


def mejorar_adaptativo(frame, c):
    """Aplica solo lo que la imagen necesita según su calidad. `c` es el módulo config.
    Devuelve (imagen, calidad_antes, calidad_despues, pasos_aplicados). Si ya está bien, no la toca."""
    from . import metricas, preprocesamiento  # import local: evita ciclo con preprocesamiento

    antes = metricas.calidad_sin_referencia(frame)
    imagen, pasos = frame, []
    if antes["brillo"] < c.UMBRAL_BRILLO:
        imagen, _ = correccion_gamma(imagen, c.BRILLO_OBJETIVO)
        pasos.append("gamma")
    if metricas.calidad_sin_referencia(imagen)["ruido"] > c.UMBRAL_RUIDO:
        imagen = preprocesamiento.aplicar_filtro(imagen, c.FILTRO_SALIDAS, c.FILTROS[c.FILTRO_SALIDAS])
        pasos.append(c.FILTRO_SALIDAS)
    if metricas.calidad_sin_referencia(imagen)["contraste"] < c.UMBRAL_CONTRASTE:
        imagen = clahe(imagen, c.CLAHE["clip"], c.CLAHE["grilla"])
        pasos.append("clahe")
    return imagen, antes, metricas.calidad_sin_referencia(imagen), pasos
