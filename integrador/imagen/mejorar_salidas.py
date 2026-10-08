"""Mejora las SALIDAS del sistema de VIDEO (capturas de alerta, imágenes o videos ya generados).

VIDEO guarda sus alertas en video_base/alertas/ y no se modifica; este módulo solo lee esos archivos
y escribe las versiones mejoradas en otra carpeta. Decide por imagen qué etapas aplicar (gamma, filtro,
CLAHE) según su brillo, ruido y contraste.

  python -m imagen.mejorar_salidas --entrada video_base/alertas --salida salidas/mejoradas
  python -m imagen.mejorar_salidas --entrada alerta.jpg --salida salidas/mejoradas
  python -m imagen.mejorar_salidas --entrada resultado.mp4 --salida salidas/mejoradas
  python -m imagen.mejorar_salidas --entrada video_base/alertas --salida salidas/mejoradas --vigilar
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np

from . import config, mejoramiento

COLUMNAS = ["archivo", "pasos", "brillo_antes", "brillo_despues", "contraste_antes", "contraste_despues",
            "nitidez_antes", "nitidez_despues", "ruido_antes", "ruido_despues"]


def fila(nombre, antes, despues, pasos):
    return {"archivo": nombre, "pasos": "+".join(pasos) or "ninguno",
            **{f"{k}_antes": antes[k] for k in ("brillo", "contraste", "nitidez", "ruido")},
            **{f"{k}_despues": despues[k] for k in ("brillo", "contraste", "nitidez", "ruido")}}


def mejorar_imagen(ruta, carpeta_salida):
    """Mejora una imagen y guarda original|mejorada lado a lado. Devuelve la fila de métricas o None."""
    imagen = cv2.imread(ruta)
    if imagen is None:  # RNF-05: un archivo corrupto no detiene el lote
        print(f"omitido (no es imagen válida): {ruta}")
        return None
    mejorada, antes, despues, pasos = mejoramiento.mejorar_adaptativo(imagen, config)
    base = os.path.splitext(os.path.basename(ruta))[0]
    cv2.imwrite(os.path.join(carpeta_salida, f"{base}_mejorada.jpg"), mejorada)
    cv2.imwrite(os.path.join(carpeta_salida, f"{base}_comparacion.jpg"), np.hstack([imagen, mejorada]))
    return fila(os.path.basename(ruta), antes, despues, pasos)


def mejorar_video(ruta, carpeta_salida):
    """Mejora cada frame de un video con los mismos criterios. Devuelve una fila con los promedios."""
    captura = cv2.VideoCapture(ruta)
    if not captura.isOpened():
        print(f"omitido (no se pudo abrir): {ruta}")
        return None
    fps = captura.get(cv2.CAP_PROP_FPS) or 20.0
    base = os.path.splitext(os.path.basename(ruta))[0]
    escritor, acumulado, n, usados = None, {}, 0, set()
    while True:
        correcto, frame = captura.read()
        if not correcto:
            break
        mejorado, antes, despues, pasos = mejoramiento.mejorar_adaptativo(frame, config)
        if escritor is None:
            alto, ancho = mejorado.shape[:2]
            escritor = cv2.VideoWriter(os.path.join(carpeta_salida, f"{base}_mejorado.mp4"),
                                       cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))
        escritor.write(mejorado)
        usados.update(pasos)
        for k in ("brillo", "contraste", "nitidez", "ruido"):
            acumulado[f"{k}_antes"] = acumulado.get(f"{k}_antes", 0) + antes[k]
            acumulado[f"{k}_despues"] = acumulado.get(f"{k}_despues", 0) + despues[k]
        n += 1
    captura.release()
    if escritor:
        escritor.release()
    if n == 0:
        return None
    fila_video = {"archivo": os.path.basename(ruta), "pasos": "+".join(sorted(usados)) or "ninguno"}
    fila_video.update({k: round(v / n, 3) for k, v in acumulado.items()})
    return fila_video


def procesar_ruta(ruta, carpeta_salida):
    """Despacha según el tipo de archivo; una carpeta se recorre completa."""
    if os.path.isdir(ruta):
        filas = []
        for nombre in sorted(os.listdir(ruta)):
            filas.extend(procesar_ruta(os.path.join(ruta, nombre), carpeta_salida))
        return filas
    extension = os.path.splitext(ruta)[1].lower()
    if extension in config.EXTENSIONES_IMAGEN:
        resultado = mejorar_imagen(ruta, carpeta_salida)
    elif extension in config.EXTENSIONES_VIDEO:
        resultado = mejorar_video(ruta, carpeta_salida)
    else:
        resultado = None
    return [resultado] if resultado else []


def escribir_tabla(filas, carpeta_salida):
    ruta = os.path.join(carpeta_salida, "calidad_antes_despues.csv")
    nuevo = not os.path.exists(ruta)
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS, extrasaction="ignore")
        if nuevo:
            escritor.writeheader()
        escritor.writerows(filas)


def main():
    p = argparse.ArgumentParser(description="Mejora las salidas (imágenes o videos) del sistema de VIDEO")
    p.add_argument("--entrada", required=True, help="imagen, video o carpeta (p. ej. video_base/alertas)")
    p.add_argument("--salida", default="salidas/mejoradas")
    p.add_argument("--vigilar", action="store_true", help="queda atento a archivos nuevos en la carpeta (Ctrl+C para salir)")
    a = p.parse_args()
    os.makedirs(a.salida, exist_ok=True)

    vistos = set()
    while True:
        pendientes = []
        for ruta in ([a.entrada] if not os.path.isdir(a.entrada) else
                     [os.path.join(a.entrada, n) for n in sorted(os.listdir(a.entrada))]):
            if ruta not in vistos:
                vistos.add(ruta)
                pendientes.append(ruta)
        filas = [f for ruta in pendientes for f in procesar_ruta(ruta, a.salida)]
        if filas:
            escribir_tabla(filas, a.salida)
            for f in filas:
                print(f"{f['archivo']}: {f['pasos']} | brillo {f['brillo_antes']} -> {f['brillo_despues']}, "
                      f"contraste {f['contraste_antes']} -> {f['contraste_despues']}")
        if not a.vigilar:
            break
        time.sleep(1.0)


if __name__ == "__main__":
    main()
