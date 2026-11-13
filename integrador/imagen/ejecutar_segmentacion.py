"""S05: compara Otsu, umbral adaptativo y watershed sobre los frames del proyecto.

  python -m imagen.ejecutar_segmentacion --video datos/padang.webm --salida salidas/segmentacion --cada-n 60

Por método y frame mide: objetos encontrados, ms, IoU de su máscara contra la de Otsu (acuerdo) e IoU contra la
unión de las cajas de VIDEO (referencia aproximada: una caja contiene también fondo, así que ningún método llega a 1.0).
"""
import argparse
import csv
import os

import cv2
import numpy as np

from . import adaptador_video, config, metricas, preprocesamiento, segmentacion

NOMBRES = ("otsu", "adaptativo", "watershed")


def metodos(frame):
    """Devuelve {nombre: (mascara, objetos, ms)} para un frame BGR ya normalizado."""
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k, area, maximo = config.KERNEL_MORFOLOGIA, config.AREA_MINIMA, config.MAX_OBJETOS
    ad, ws = config.SEGMENTACION["adaptativo"], config.SEGMENTACION["watershed"]

    def otsu():
        m = segmentacion.mascara_otsu(gris, k)
        return m, segmentacion.extraer_objetos(m, area, maximo)

    def adaptativo():
        m = segmentacion.mascara_adaptativa(gris, ad["bloque"], ad["c"], k)
        return m, segmentacion.extraer_objetos(m, area, maximo)

    def watershed():
        m, etiquetas = segmentacion.segmentar_watershed(frame, k, ws["fraccion_distancia"])
        return m, segmentacion.objetos_de_etiquetas(etiquetas, area, maximo)

    salida = {}
    for nombre, funcion in zip(NOMBRES, (otsu, adaptativo, watershed)):
        (mascara, objetos), ms = metricas.cronometrar(funcion)
        salida[nombre] = (mascara, objetos, ms)
    return salida


def main():
    p = argparse.ArgumentParser(description="Otsu vs umbral adaptativo vs watershed")
    p.add_argument("--video", required=True)
    p.add_argument("--salida", default="salidas/segmentacion")
    p.add_argument("--cada-n", type=int, default=60)
    p.add_argument("--evidencias", type=int, default=2)
    a = p.parse_args()

    os.makedirs(a.salida, exist_ok=True)
    detector = adaptador_video.cargar_detector()
    acum, n_frames = {}, 0
    for entrada in adaptador_video.leer_frames(a.video, a.cada_n):
        try:
            frame = preprocesamiento.normalizar_tamano(entrada["frame"], config.ANCHO_PROCESO)
            cajas = metricas.mascara_de_cajas(frame.shape, [d["bbox"] for d in detector(frame)])
            resultados = metodos(frame)
        except Exception as e:  # RNF-05: un frame malo no detiene el proceso
            print(f"frame {entrada['frame_id']} omitido: {e}")
            continue
        n_frames += 1
        otsu = resultados["otsu"][0]
        for nombre, (mascara, objetos, ms) in resultados.items():
            s = acum.setdefault(nombre, {"objetos": 0, "iou_otsu": 0.0, "iou_cajas": 0.0, "ms": 0.0})
            s["objetos"] += len(objetos)
            s["iou_otsu"] += metricas.iou_mascaras(mascara, otsu)
            s["iou_cajas"] += metricas.iou_mascaras(mascara, cajas)
            s["ms"] += ms
        if n_frames <= a.evidencias:
            paneles = [frame] + [cv2.cvtColor(resultados[n][0], cv2.COLOR_GRAY2BGR) for n in NOMBRES]
            cv2.imwrite(os.path.join(a.salida, f"frame{entrada['frame_id']:04d}_original_otsu_adaptativo_watershed.jpg"),
                        np.hstack([cv2.resize(x, (480, 270)) for x in paneles]))

    ruta = os.path.join(a.salida, "tabla_segmentacion.csv")
    n = max(n_frames, 1)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metodo", "objetos_por_frame", "iou_vs_otsu", "iou_vs_cajas_video", "ms_por_frame"])
        for nombre in NOMBRES:
            s = acum.get(nombre)
            if not s:
                continue
            fila = [nombre, round(s["objetos"] / n, 1), round(s["iou_otsu"] / n, 3),
                    round(s["iou_cajas"] / n, 3), round(s["ms"] / n, 1)]
            w.writerow(fila)
            print(f"{nombre:11s} objetos/frame={fila[1]:5} iou_vs_otsu={fila[2]:.3f} "
                  f"iou_vs_cajas={fila[3]:.3f} ms={fila[4]}")
    print(f"{n_frames} frames -> {ruta}")


if __name__ == "__main__":
    main()
