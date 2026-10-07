"""Ejecuta el detector de VIDEO sobre un archivo y guarda sus resultados para IMAGEN.

python -m imagen.ejecutar_video --video datos/padang.webm --salida salidas/video
Salidas: detecciones.jsonl (una línea por frame), video_anotado.mp4, resumen_video.json
"""
import argparse
import collections
import json
import os

import cv2

from . import adaptador_video, metricas


def main():
    p = argparse.ArgumentParser(description="Ejecuta VIDEO (detector YOLO) y guarda sus resultados")
    p.add_argument("--video", required=True)
    p.add_argument("--salida", default="salidas/video")
    p.add_argument("--cada-n", type=int, default=1, help="procesa 1 de cada N frames")
    a = p.parse_args()

    os.makedirs(a.salida, exist_ok=True)
    detectar = adaptador_video.cargar_detector()
    dibujar = adaptador_video.cargar_dibujante()

    escritor, conteo, n, ms_total = None, collections.Counter(), 0, 0.0
    with open(os.path.join(a.salida, "detecciones.jsonl"), "w", encoding="utf-8") as f:
        for entrada in adaptador_video.leer_frames(a.video, a.cada_n):
            detecciones, ms = metricas.cronometrar(detectar, entrada["frame"])
            f.write(json.dumps({
                "frame_id": entrada["frame_id"], "timestamp": round(entrada["timestamp"], 3),
                "fps": entrada["fps"], "fuente": entrada["fuente"], "detecciones": detecciones,
            }, ensure_ascii=False) + "\n")
            conteo.update(d["clase"] for d in detecciones)
            n += 1
            ms_total += ms
            anotado = dibujar(entrada["frame"], detecciones)
            if escritor is None:
                alto, ancho = anotado.shape[:2]
                fps_salida = max(entrada["fps"] / a.cada_n, 1.0)
                escritor = cv2.VideoWriter(os.path.join(a.salida, "video_anotado.mp4"),
                                           cv2.VideoWriter_fourcc(*"mp4v"), fps_salida, (ancho, alto))
            escritor.write(anotado)
    if escritor:
        escritor.release()

    resumen = {"frames_procesados": n, "detecciones_por_clase": dict(conteo),
               "ms_por_frame": round(ms_total / max(n, 1), 1)}
    with open(os.path.join(a.salida, "resumen_video.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
