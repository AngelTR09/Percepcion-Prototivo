"""Genera el dataset de recortes de VIDEO para los laboratorios de aprendizaje (S12-S14).

  python -m imagen.ejecutar_recortes --video datos/padang.webm --salida salidas/recortes

Cada detección de VIDEO (persona o mochila) se recorta del frame y se reduce a lado x lado píxeles. La etiqueta es la clase que
dio VIDEO (etiqueta débil: no hay anotación manual). Guarda `recortes.npz` con x, y, frame_id y los nombres de clase.
Se ejecuta en el entorno principal (necesita OpenCV y el detector); el entrenamiento solo lee el .npz.
"""
import argparse
import os

import cv2
import numpy as np

from . import adaptador_video, config, preprocesamiento


def recortes_del_frame(frame, detecciones, clases, lado, min_lado_px):
    """[(recorte lado x lado BGR, indice_de_clase)] de las detecciones de `clases` con un tamaño mínimo."""
    salida = []
    alto, ancho = frame.shape[:2]
    for d in detecciones:
        if d["clase"] not in clases:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in d["bbox"])
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, ancho), min(y2, alto)
        if x2 - x1 < min_lado_px or y2 - y1 < min_lado_px:
            continue
        salida.append((cv2.resize(frame[y1:y2, x1:x2], (lado, lado), interpolation=cv2.INTER_AREA), clases.index(d["clase"])))
    return salida


def main():
    r = config.RECORTES
    p = argparse.ArgumentParser(description="Recortes etiquetados por VIDEO")
    p.add_argument("--video", required=True)
    p.add_argument("--salida", default="salidas/recortes")
    p.add_argument("--cada-n", type=int, default=r["cada_n"])
    p.add_argument("--lado", type=int, default=r["lado"], help="píxeles de lado del recorte (32 para MLP/CNN; 96 para transfer learning)")
    a = p.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    detector = adaptador_video.cargar_detector()
    xs, ys, fs = [], [], []
    for entrada in adaptador_video.leer_frames(a.video, a.cada_n):
        try:
            frame = preprocesamiento.normalizar_tamano(entrada["frame"], config.ANCHO_PROCESO)
            for recorte, clase in recortes_del_frame(frame, detector(frame), list(r["clases"]), a.lado, r["min_lado_px"]):
                xs.append(recorte)
                ys.append(clase)
                fs.append(entrada["frame_id"])
        except Exception as e:  # RNF-05
            print(f"frame {entrada['frame_id']} omitido: {e}")
    if not xs:
        raise SystemExit("no se obtuvo ningún recorte")
    ruta = os.path.join(a.salida, "recortes.npz")
    np.savez_compressed(ruta, x=np.stack(xs), y=np.array(ys), frame_id=np.array(fs), clases=np.array(r["clases"]))
    conteo = {c: int(np.sum(np.array(ys) == i)) for i, c in enumerate(r["clases"])}
    print(f"{len(xs)} recortes {conteo} de {len(set(fs))} frames -> {ruta}")


if __name__ == "__main__":
    main()
