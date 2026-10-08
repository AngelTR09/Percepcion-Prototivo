"""Salidas para PREDICCIÓN (.jsonl y .csv) y evidencias visuales."""
import csv
import json
import os

import cv2

COLUMNAS_CSV = [
    "frame_id", "timestamp", "psnr", "ssim", "contraste", "filtro_usado", "n_objetos",
    "densidad_bordes", "n_esquinas", "n_keypoints_orb", "h_media", "s_media", "v_media", "tiempo_ms",
]


def fila_csv(r):
    h, s, v = r["caracteristicas"]["hist_hsv"]
    c, k = r["calidad"], r["caracteristicas"]
    return {
        "frame_id": r["frame_id"], "timestamp": r["timestamp"], "psnr": c["psnr"], "ssim": c["ssim"],
        "contraste": c["contraste"], "filtro_usado": c["filtro_usado"], "n_objetos": k["n_objetos"],
        "densidad_bordes": k["densidad_bordes"], "n_esquinas": k["n_esquinas"],
        "n_keypoints_orb": k["n_keypoints_orb"], "h_media": h, "s_media": s, "v_media": v,
        "tiempo_ms": r["tiempo_ms"],
    }


def escribir_jsonl(registros, ruta):
    with open(ruta, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def escribir_csv(registros, ruta):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS_CSV)
        escritor.writeheader()
        for r in registros:
            escritor.writerow(fila_csv(r))


def guardar_evidencias(imagenes, carpeta, frame_id):
    os.makedirs(carpeta, exist_ok=True)
    for nombre, imagen in imagenes.items():
        cv2.imwrite(os.path.join(carpeta, f"frame{frame_id:04d}_{nombre}.jpg"), imagen)
