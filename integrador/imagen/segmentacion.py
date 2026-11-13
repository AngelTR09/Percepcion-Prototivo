"""Segmentación de objetos de interés (S05): Otsu + morfología + contornos."""
import cv2
import numpy as np
from scipy import ndimage


def mascara_otsu(gris, kernel):
    suave = cv2.GaussianBlur(gris, (5, 5), 0)
    _, binaria = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, nucleo)   # quita ruido pequeño
    return cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, nucleo)     # rellena huecos


def extraer_objetos(mascara, area_minima, max_objetos):
    """Contornos -> bbox, área, perímetro y centroide (RF-07), de mayor a menor área."""
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objetos = []
    for c in sorted(contornos, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area < area_minima or len(objetos) >= max_objetos:
            continue
        x, y, w, h = cv2.boundingRect(c)
        m = cv2.moments(c)
        cx, cy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (x + w / 2, y + h / 2)
        objetos.append({
            "id": len(objetos),
            "bbox": [x, y, w, h],
            "area": float(area),
            "perimetro": float(cv2.arcLength(c, True)),
            "centroide": [round(cx, 1), round(cy, 1)],
        })
    return objetos


def segmentar(frame, area_minima, max_objetos, kernel):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mascara = mascara_otsu(gris, kernel)
    return extraer_objetos(mascara, area_minima, max_objetos), mascara


# ---- Alternativas a Otsu (S05) ----

def _nucleo(kernel):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))


def mascara_adaptativa(gris, bloque, c, kernel):
    """Umbral adaptativo: cada píxel se compara con la media gaussiana de su vecindario (`bloque` impar).
    Funciona con iluminación despareja, donde un umbral global como Otsu falla."""
    if bloque % 2 == 0 or bloque < 3:
        raise ValueError("el bloque debe ser impar y >= 3")
    suave = cv2.GaussianBlur(gris, (5, 5), 0)
    binaria = cv2.adaptiveThreshold(suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bloque, -c)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, _nucleo(kernel))
    return cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, _nucleo(kernel))


def segmentar_watershed(frame, kernel, fraccion_distancia):
    """Segmentación por regiones con watershed: separa objetos que se tocan, cosa que los contornos de una
    máscara binaria no hacen. Devuelve (mascara 0/255, etiquetas int32 con 0 = fondo y 1..n = región)."""
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    base = mascara_otsu(gris, kernel)
    distancia = cv2.distanceTransform(base, cv2.DIST_L2, 5)
    if distancia.max() <= 0:
        return np.zeros_like(base), np.zeros(base.shape, np.int32)
    # La fracción se aplica por componente conexa: con un umbral global, un objeto grande apagaría
    # las semillas de los pequeños (se probó: el IoU contra las cajas de VIDEO caía de 0.22 a 0.06).
    n, componentes = cv2.connectedComponents(base)
    maximos = np.asarray(ndimage.maximum(distancia, componentes, index=np.arange(n)), np.float32)
    semillas = ((componentes > 0) & (distancia >= fraccion_distancia * maximos[componentes])).astype(np.uint8) * 255
    desconocido = cv2.subtract(cv2.dilate(base, _nucleo(kernel), iterations=2), semillas)
    _, marcadores = cv2.connectedComponents(semillas)
    marcadores = marcadores + 1                      # 1 = fondo seguro; semillas desde 2
    marcadores[desconocido == 255] = 0               # 0 = zona por decidir
    marcadores = cv2.watershed(np.ascontiguousarray(frame), marcadores)
    etiquetas = np.where(marcadores > 1, marcadores - 1, 0).astype(np.int32)
    return ((etiquetas > 0) * 255).astype(np.uint8), etiquetas


def objetos_de_etiquetas(etiquetas, area_minima, max_objetos):
    """Mismo formato que `extraer_objetos`, pero a partir de una imagen de etiquetas (una región = un objeto)."""
    objetos = []
    ids = [i for i in np.unique(etiquetas) if i > 0]
    ids.sort(key=lambda i: -int((etiquetas == i).sum()))
    for i in ids:
        region = (etiquetas == i).astype(np.uint8)
        area = float(region.sum())
        if area < area_minima:
            continue
        contornos, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contornos, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        ys, xs = np.nonzero(region)
        objetos.append({
            "id": len(objetos), "bbox": [x, y, w, h], "area": area,
            "perimetro": float(cv2.arcLength(c, True)),
            "centroide": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
        })
        if len(objetos) >= max_objetos:
            break
    return objetos
