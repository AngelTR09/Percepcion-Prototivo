"""S06: Sobel vs Canny y reconocimiento de patrones (ORB + BFMatcher vs matchTemplate) sobre los frames del proyecto.

  python -m imagen.ejecutar_patrones --video datos/padang.webm --salida salidas/patrones --cada-n 60

Bordes: Canny y Sobel sobre cada frame (densidad, ms y cuánto del borde de uno cubre el otro).
Patrones: el recorte de la detección de VIDEO más segura se busca en el mismo frame transformado (rotado, escalado,
trasladado, volteado: las de config.TRANSFORMACIONES). Acierto = IoU >= IOU_MINIMO con la caja transformada.
"""
import argparse
import csv
import os

import cv2

from . import adaptador_video, caracteristicas, config, metricas, preprocesamiento, transformaciones

MIN_LADO = 40  # un recorte de menos de 40 px casi no tiene puntos clave


def main():
    p = argparse.ArgumentParser(description="Sobel vs Canny; ORB vs matchTemplate")
    p.add_argument("--video", required=True)
    p.add_argument("--salida", default="salidas/patrones")
    p.add_argument("--cada-n", type=int, default=60)
    a = p.parse_args()

    os.makedirs(a.salida, exist_ok=True)
    detector = adaptador_video.cargar_detector()
    bordes, patrones, n_frames, n_patrones = [], {}, 0, 0
    for entrada in adaptador_video.leer_frames(a.video, a.cada_n):
        try:
            frame = preprocesamiento.normalizar_tamano(entrada["frame"], config.ANCHO_PROCESO)
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bordes.append(caracteristicas.comparar_bordes(gris, config.CANNY, config.SOBEL))
            n_frames += 1
            grandes = [d for d in detector(frame)
                       if d["bbox"][2] - d["bbox"][0] >= MIN_LADO and d["bbox"][3] - d["bbox"][1] >= MIN_LADO]
            if not grandes:
                continue
            mejor = max(grandes, key=lambda d: d["confianza"])
            x1, y1, x2, y2 = (int(v) for v in mejor["bbox"])
            plantilla = gris[y1:y2, x1:x2]
            n_patrones += 1
            for t in config.TRANSFORMACIONES:
                movida, M = transformaciones.aplicar(frame, t)
                visibles = transformaciones.transformar_cajas([mejor["bbox"]], M, frame.shape, config.VISIBLE_MINIMO)
                if not visibles:
                    continue
                esperada = visibles[0][1]
                escena = cv2.cvtColor(movida, cv2.COLOR_BGR2GRAY)
                intentos = {
                    "orb": lambda: caracteristicas.localizar_con_orb(plantilla, escena, config.ORB_EMPAREJAR),
                    "plantilla": lambda: caracteristicas.localizar_con_plantilla(plantilla, escena, config.PLANTILLA_UMBRAL),
                }
                for metodo, buscar in intentos.items():
                    hallado, ms = metricas.cronometrar(buscar)
                    acierto = bool(hallado) and metricas.iou(hallado["bbox"], esperada) >= config.IOU_MINIMO
                    s = patrones.setdefault((t["nombre"], metodo), {"n": 0, "aciertos": 0, "ms": 0.0})
                    s["n"] += 1
                    s["aciertos"] += acierto
                    s["ms"] += ms
        except Exception as e:  # RNF-05: un frame malo no detiene el proceso
            print(f"frame {entrada['frame_id']} omitido: {e}")

    claves = ["densidad_canny", "densidad_sobel", "canny_cubierto_por_sobel", "sobel_cubierto_por_canny", "ms_canny", "ms_sobel"]
    prom = {k: round(sum(b[k] for b in bordes) / max(len(bordes), 1), 4) for k in claves}
    with open(os.path.join(a.salida, "tabla_bordes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(claves)
        w.writerow([prom[k] for k in claves])
    print(f"BORDES ({n_frames} frames): {prom}")

    with open(os.path.join(a.salida, "tabla_patrones.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["transformacion", "metodo", "intentos", "aciertos", "tasa", "ms_por_busqueda"])
        for (nombre, metodo), s in patrones.items():
            tasa = round(s["aciertos"] / s["n"], 3)
            w.writerow([nombre, metodo, s["n"], s["aciertos"], tasa, round(s["ms"] / s["n"], 1)])
            print(f"{nombre:20s} {metodo:9s} aciertos={s['aciertos']:2d}/{s['n']:2d} tasa={tasa} ms={s['ms'] / s['n']:.1f}")
    print(f"{n_patrones} patrones de VIDEO buscados")


if __name__ == "__main__":
    main()
