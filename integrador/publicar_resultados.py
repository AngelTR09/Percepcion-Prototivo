"""Copia las salidas de salidas/ a resultados/ (esta sí se sube a git) para que otros grupos las usen.

  python publicar_resultados.py
Los videos se reducen a 640 px de ancho para no inflar el repositorio.
"""
import os
import shutil

import cv2

ORIGEN, DESTINO = "salidas", "resultados"
# (subcarpeta de salidas, nombre en resultados, extensiones a copiar o None = todas)
COPIAR = [
    ("video", "video", None),
    ("imagen", "imagen_evaluacion", None),
    ("integrado", "imagen_integrado", None),
    ("alertas", "alertas_mejoradas", None),
    ("mejoradas", "salidas_mejoradas", None),
]


def copiar_video_reducido(origen, destino, ancho=640):
    cap = cv2.VideoCapture(origen)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    escritor = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        alto = int(frame.shape[0] * ancho / frame.shape[1])
        frame = cv2.resize(frame, (ancho, alto), interpolation=cv2.INTER_AREA)
        if escritor is None:
            escritor = cv2.VideoWriter(destino, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))
        escritor.write(frame)
    cap.release()
    if escritor:
        escritor.release()


def main():
    if os.path.exists(DESTINO):
        shutil.rmtree(DESTINO)
    for sub, nombre, _ in COPIAR:
        raiz = os.path.join(ORIGEN, sub)
        if not os.path.isdir(raiz):
            print(f"no existe {raiz}: ejecuta primero correr_todo.sh / ejecutar_integrado")
            continue
        for carpeta, _, archivos in os.walk(raiz):
            salida = os.path.join(DESTINO, nombre, os.path.relpath(carpeta, raiz))
            os.makedirs(salida, exist_ok=True)
            for a in archivos:
                if a.endswith(".mp4"):
                    copiar_video_reducido(os.path.join(carpeta, a), os.path.join(salida, a))
                else:
                    shutil.copy2(os.path.join(carpeta, a), salida)
    print("Resultados en", DESTINO)


if __name__ == "__main__":
    main()
