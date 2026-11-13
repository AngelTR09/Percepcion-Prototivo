"""S13: CNN y transfer learning con Keras, en el entorno del laboratorio (`requirements-lab.txt`).

  python -m imagen.ejecutar_cnn --salida salidas/cnn --fashion
  python -m imagen.ejecutar_cnn --salida salidas/cnn --recortes32 salidas/recortes/recortes.npz
  python -m imagen.ejecutar_cnn --salida salidas/cnn --recortes96 salidas/recortes96/recortes.npz     # transfer learning

Los recortes se generan con `python -m imagen.ejecutar_recortes --lado 32|96`. Con recortes, cada modelo se evalúa también
con el conjunto de prueba rotado (transformación de S04) para medir la robustez, y se compara con la línea base mayoritaria.
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from . import cnn, config, mlp
from .ejecutar_mlp import NOMBRES_FASHION, guardar


def _fila_robustez(modelo, xp, yp, n):
    filas = []
    for g in config.ROBUSTEZ_GRADOS:
        m, _ = mlp.evaluar(modelo, cnn.rotar_lote(xp, g), yp, n)
        filas.append({"grados": g, "exactitud": round(m["exactitud"], 4), "recall": m["recall"]})
    return filas


def experimento(nombre, modelo, xe, ye, xp, yp, clases, salida, robustez=False, pesos_clase=None, epocas=None):
    n = len(clases)
    inicio = time.perf_counter()
    historia = mlp.entrenar(modelo, xe, ye, epocas=epocas, pesos_clase=pesos_clase)
    segundos = time.perf_counter() - inicio
    metricas, matriz = mlp.evaluar(modelo, xp, yp, n)
    entrenables, congelados = cnn.parametros(modelo)
    resumen = {
        "experimento": nombre, "clases": list(clases), "n_entrenamiento": int(len(xe)), "n_prueba": int(len(xp)),
        "parametros_entrenables": entrenables, "parametros_congelados": congelados,
        "epocas_ejecutadas": len(historia["loss"]), "segundos_entrenamiento": round(segundos, 1),
        "val_accuracy_final": round(historia["val_accuracy"][-1], 4),
        "linea_base_mayoritaria": round(mlp.linea_base_mayoritaria(ye, yp), 4), "prueba": metricas,
    }
    if robustez:
        resumen["robustez_rotacion"] = _fila_robustez(modelo, xp, yp, n)
    guardar(salida, nombre, historia, matriz, resumen)
    print(f"[{nombre}] exactitud={metricas['exactitud']:.3f} (base {resumen['linea_base_mayoritaria']:.3f}) "
          f"entrenables={entrenables:,} épocas={resumen['epocas_ejecutadas']} {segundos:.0f} s")
    for i, c in enumerate(clases):
        print(f"   {c:10s} soporte={metricas['soporte'][i]:5d} precisión={metricas['precision'][i]:.3f} "
              f"recall={metricas['recall'][i]:.3f} f1={metricas['f1'][i]:.3f}")
    for f in resumen.get("robustez_rotacion", []):
        print(f"   rotación {f['grados']:4.0f}°: exactitud={f['exactitud']:.3f} recall={f['recall']}")
    return resumen


def comparar_semillas(xe, ye, xp, yp, clases, n_semillas, salida):
    """MLP, CNN y CNN con aumento entrenados con `n_semillas` semillas distintas: media y desviación de la exactitud y del recall de
    la clase minoritaria (última clase) con el conjunto de prueba normal y rotado. Un solo entrenamiento no distingue un modelo mejor del ruido."""
    n, minoritaria = len(clases), len(clases) - 1
    modelos = {
        "mlp": lambda sem: mlp.construir_mlp(xe.shape[1:], n, semilla=sem),
        "cnn_sin_aumento": lambda sem: cnn.construir_cnn(xe.shape[1:], n, semilla=sem),
        "cnn_con_aumento": lambda sem: cnn.construir_cnn(xe.shape[1:], n, aumento=True, semilla=sem),
    }
    giradas = {g: cnn.rotar_lote(xp, g) for g in config.ROBUSTEZ_GRADOS}
    filas = []
    for nombre, crear in modelos.items():
        medidas = {(g, k): [] for g in config.ROBUSTEZ_GRADOS for k in ("exactitud", "recall")}
        for i in range(n_semillas):
            modelo = crear(config.CNN["semilla"] + i)
            mlp.entrenar(modelo, xe, ye)
            for g, x in giradas.items():
                m, _ = mlp.evaluar(modelo, x, yp, n)
                medidas[(g, "exactitud")].append(m["exactitud"])
                medidas[(g, "recall")].append(m["recall"][minoritaria])
        fila = {"modelo": nombre}
        for (g, k), v in medidas.items():
            fila[f"{k}_{g:.0f}grados"] = f"{np.mean(v):.3f} ± {np.std(v):.3f}"
        filas.append(fila)
        print(f"[{nombre}] " + "  ".join(f"{k}={v}" for k, v in fila.items() if k != "modelo"))
    with open(os.path.join(salida, "comparacion_semillas.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


def cargar_recortes(ruta, rgb=False):
    d = np.load(ruta)
    x = d["x"].astype(np.float32) / 255.0
    if rgb:
        x = x[..., ::-1].copy()      # OpenCV guarda BGR; MobileNetV2 espera RGB
    y, clases = d["y"], [str(c) for c in d["clases"]]
    prueba = mlp.dividir_por_tiempo(d["frame_id"], config.RECORTES["fraccion_prueba"])
    return x[~prueba], y[~prueba], x[prueba], y[prueba], clases


def main():
    p = argparse.ArgumentParser(description="CNN y transfer learning con Keras")
    p.add_argument("--salida", default="salidas/cnn")
    p.add_argument("--fashion", action="store_true")
    p.add_argument("--recortes32")
    p.add_argument("--recortes96")
    p.add_argument("--semillas", type=int, default=0, help="con --recortes32: compara MLP/CNN con N semillas (media ± desviación)")
    a = p.parse_args()
    os.makedirs(a.salida, exist_ok=True)

    if a.fashion:
        f = config.FASHION_REDUCIDO
        (xe, ye), (xp, yp) = mlp.cargar_fashion_mnist(f["entrenamiento"], f["prueba"], f["semilla"])
        xe, xp = xe[..., None], xp[..., None]
        experimento("cnn_fashion_mnist", cnn.construir_cnn(xe.shape[1:], 10), xe, ye, xp, yp, NOMBRES_FASHION, a.salida)

    if a.recortes32:
        xe, ye, xp, yp, clases = cargar_recortes(a.recortes32)
        if a.semillas:
            comparar_semillas(xe, ye, xp, yp, clases, a.semillas, a.salida)
        for nombre, aumento in (("cnn_recortes_sin_aumento", False), ("cnn_recortes_con_aumento", True)):
            experimento(nombre, cnn.construir_cnn(xe.shape[1:], len(clases), aumento=aumento),
                        xe, ye, xp, yp, clases, a.salida, robustez=True)

    if a.recortes96:
        xe, ye, xp, yp, clases = cargar_recortes(a.recortes96, rgb=True)
        modelo = cnn.construir_transfer(len(clases))
        experimento("transfer_cabeza", modelo, xe, ye, xp, yp, clases, a.salida, robustez=True)
        cnn.afinar(modelo)
        experimento("transfer_afinado", modelo, xe, ye, xp, yp, clases, a.salida, robustez=True,
                    epocas=config.TRANSFER["epocas_afinado"])


if __name__ == "__main__":
    main()
