"""Transformaciones geométricas (S04): rotación, escalamiento, traslación y volteo con `cv2.warpAffine`.

Cada transformación devuelve `(imagen, M)`; `M` es la matriz afín 2x3 que lleva coordenadas de la imagen
original a la nueva. Con `M` se transforman también las cajas, así se mide la robustez del detector de VIDEO
(¿encuentra los mismos objetos en la imagen transformada?). La imagen mantiene su tamaño; lo que sale del encuadre se recorta.
"""
import cv2
import numpy as np

from . import metricas


def _validar(frame):
    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError("frame vacío o corrupto")


def _aplicar(frame, M):
    alto, ancho = frame.shape[:2]
    return cv2.warpAffine(frame, M, (ancho, alto), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def trasladar(frame, dx, dy):
    """Desplaza `dx` píxeles a la derecha y `dy` hacia abajo."""
    _validar(frame)
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float64)
    return _aplicar(frame, M), M


def rotar(frame, grados, escala=1.0):
    """Rota `grados` (positivo = antihorario) alrededor del centro."""
    _validar(frame)
    alto, ancho = frame.shape[:2]
    M = cv2.getRotationMatrix2D(((ancho - 1) / 2.0, (alto - 1) / 2.0), grados, escala)
    return _aplicar(frame, M), M


def escalar(frame, factor):
    """Escala alrededor del centro: factor < 1 reduce (queda borde negro), factor > 1 amplía y recorta."""
    if factor <= 0:
        raise ValueError("el factor de escala debe ser positivo")
    return rotar(frame, 0.0, factor)


def voltear(frame):
    """Espejo horizontal."""
    _validar(frame)
    ancho = frame.shape[1]
    M = np.array([[-1, 0, ancho - 1], [0, 1, 0]], np.float64)
    return _aplicar(frame, M), M


def aplicar(frame, t):
    """Aplica una transformación descrita por un dict de `config.TRANSFORMACIONES`."""
    _validar(frame)
    tipo = t["tipo"]
    alto, ancho = frame.shape[:2]
    if tipo == "rotar":
        return rotar(frame, t["grados"], t.get("escala", 1.0))
    if tipo == "escalar":
        return escalar(frame, t["factor"])
    if tipo == "trasladar":
        return trasladar(frame, t["dx"] * ancho, t["dy"] * alto)
    if tipo == "voltear":
        return voltear(frame)
    raise ValueError(f"transformación desconocida: {tipo}")


def transformar_cajas(cajas, M, forma, visible_minimo=0.5):
    """Lleva cajas [x1, y1, x2, y2] por `M`. Devuelve [(indice_original, caja_nueva)] solo de las que
    siguen visibles (>= `visible_minimo` de su área dentro del encuadre). La caja nueva es la envolvente de las 4 esquinas."""
    alto, ancho = forma[:2]
    salida = []
    for i, (x1, y1, x2, y2) in enumerate(cajas):
        esquinas = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], np.float64) @ M.T
        nx1, ny1 = esquinas.min(axis=0)
        nx2, ny2 = esquinas.max(axis=0)
        area = max((nx2 - nx1) * (ny2 - ny1), 1e-9)
        cx1, cy1, cx2, cy2 = max(nx1, 0), max(ny1, 0), min(nx2, ancho), min(ny2, alto)
        dentro = max(cx2 - cx1, 0) * max(cy2 - cy1, 0)
        if dentro / area >= visible_minimo and cx2 > cx1 and cy2 > cy1:
            salida.append((i, [float(cx1), float(cy1), float(cx2), float(cy2)]))
    return salida


def evaluar_robustez(frame, detector, transformaciones, iou_minimo=0.5, visible_minimo=0.5):
    """Para cada transformación: cuántas detecciones de referencia (frame original) reaparecen en el frame transformado.
    Devuelve una lista de filas {nombre, referencia, recuperadas, tasa, ms}."""
    _validar(frame)
    referencia = detector(frame)
    filas = []
    for t in transformaciones:
        imagen, M = aplicar(frame, t)
        detectadas, ms = metricas.cronometrar(detector, imagen)
        visibles = transformar_cajas([d["bbox"] for d in referencia], M, frame.shape, visible_minimo)
        esperadas = [{"clase": referencia[i]["clase"], "bbox": caja} for i, caja in visibles]
        rec = metricas.recuperadas(esperadas, detectadas, iou_minimo)
        filas.append({
            "nombre": t["nombre"], "referencia": len(esperadas), "recuperadas": rec,
            "tasa": round(rec / len(esperadas), 3) if esperadas else None, "ms": round(ms, 1),
        })
    return filas
