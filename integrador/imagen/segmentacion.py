"""Segmentación de objetos de interés (S05): Otsu + morfología + contornos."""
import cv2
import numpy as np


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
