"""S04: robustez del detector de VIDEO ante transformaciones geométricas.

  python -m imagen.ejecutar_geometricas --video datos/padang.webm --salida salidas/geometricas --cada-n 60

Genera tabla_geometricas.csv (acumulado por transformación) y evidencias con las cajas de VIDEO.
"""
import argparse
import csv
import os

import cv2

from . import adaptador_video, config, preprocesamiento, transformaciones


def main():
    p = argparse.ArgumentParser(description="Robustez del detector ante rotación, escala y traslación")
    p.add_argument("--video", required=True)
    p.add_argument("--salida", default="salidas/geometricas")
    p.add_argument("--cada-n", type=int, default=60)
    p.add_argument("--evidencias", type=int, default=1, help="en cuántos frames guardar cada transformación")
    a = p.parse_args()

    os.makedirs(a.salida, exist_ok=True)
    detector = adaptador_video.cargar_detector()
    acum, n_frames = {}, 0
    for entrada in adaptador_video.leer_frames(a.video, a.cada_n):
        try:
            frame = preprocesamiento.normalizar_tamano(entrada["frame"], config.ANCHO_PROCESO)
            filas = transformaciones.evaluar_robustez(
                frame, detector, config.TRANSFORMACIONES, config.IOU_MINIMO, config.VISIBLE_MINIMO)
        except Exception as e:  # RNF-05: un frame malo no detiene el proceso
            print(f"frame {entrada['frame_id']} omitido: {e}")
            continue
        n_frames += 1
        for f in filas:
            s = acum.setdefault(f["nombre"], {"referencia": 0, "recuperadas": 0, "ms": 0.0})
            s["referencia"] += f["referencia"]
            s["recuperadas"] += f["recuperadas"]
            s["ms"] += f["ms"]
        if n_frames <= a.evidencias:
            for t in config.TRANSFORMACIONES:
                imagen, _ = transformaciones.aplicar(frame, t)
                for d in detector(imagen):
                    x1, y1, x2, y2 = map(int, d["bbox"])
                    cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(a.salida, f"frame{entrada['frame_id']:04d}_{t['nombre']}.jpg"), imagen)

    ruta = os.path.join(a.salida, "tabla_geometricas.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["transformacion", "referencia", "recuperadas", "tasa", "ms_por_frame"])
        for nombre, s in acum.items():
            tasa = round(s["recuperadas"] / s["referencia"], 3) if s["referencia"] else ""
            w.writerow([nombre, s["referencia"], s["recuperadas"], tasa, round(s["ms"] / max(n_frames, 1), 1)])
            print(f"{nombre:20s} referencia={s['referencia']:4d} recuperadas={s['recuperadas']:4d} tasa={tasa}")
    print(f"{n_frames} frames -> {ruta}")


if __name__ == "__main__":
    main()
