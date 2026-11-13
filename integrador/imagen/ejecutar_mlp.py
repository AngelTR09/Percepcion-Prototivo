"""S12: MLP con Keras. Dos experimentos, en el entorno del laboratorio (`requirements-lab.txt`):

  python -m imagen.ejecutar_mlp --salida salidas/mlp                                   # Fashion-MNIST reducido
  python -m imagen.ejecutar_mlp --salida salidas/mlp --recortes salidas/recortes/recortes.npz   # recortes de VIDEO

Guarda historia de entrenamiento (CSV), matriz de confusión (CSV) y un resumen JSON con exactitud, precisión, recall y F1.
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from . import config, mlp

NOMBRES_FASHION = ["camiseta", "pantalon", "jersey", "vestido", "abrigo", "sandalia", "camisa", "zapatilla", "bolso", "botin"]


def guardar(salida, nombre, historia, matriz, resumen):
    with open(os.path.join(salida, f"{nombre}_historia.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        claves = list(historia)
        w.writerow(["epoca", *claves])
        for i in range(len(historia[claves[0]])):
            w.writerow([i + 1, *[round(historia[k][i], 5) for k in claves]])
    np.savetxt(os.path.join(salida, f"{nombre}_matriz_confusion.csv"), matriz, fmt="%d", delimiter=",")
    with open(os.path.join(salida, f"{nombre}_resumen.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)


def experimento(nombre, xe, ye, xp, yp, clases, salida, pesos_clase=None):
    n = len(clases)
    inicio = time.perf_counter()
    modelo = mlp.construir_mlp(xe.shape[1:], n)
    historia = mlp.entrenar(modelo, xe, ye, pesos_clase=pesos_clase)
    segundos = time.perf_counter() - inicio
    metricas, matriz = mlp.evaluar(modelo, xp, yp, n)
    resumen = {
        "experimento": nombre, "clases": list(clases), "n_entrenamiento": int(len(xe)), "n_prueba": int(len(xp)),
        "parametros": int(modelo.count_params()), "epocas_ejecutadas": len(historia["loss"]),
        "segundos_entrenamiento": round(segundos, 1), "val_accuracy_final": round(historia["val_accuracy"][-1], 4),
        "linea_base_mayoritaria": round(mlp.linea_base_mayoritaria(ye, yp), 4), "prueba": metricas,
        "configuracion": {k: (list(v) if isinstance(v, tuple) else v) for k, v in config.MLP.items()},
    }
    guardar(salida, nombre, historia, matriz, resumen)
    print(f"[{nombre}] exactitud prueba={metricas['exactitud']:.3f} (línea base {resumen['linea_base_mayoritaria']:.3f}) "
          f"épocas={resumen['epocas_ejecutadas']} {segundos:.0f} s")
    for i, c in enumerate(clases):
        print(f"   {c:10s} soporte={metricas['soporte'][i]:5d} precisión={metricas['precision'][i]:.3f} "
              f"recall={metricas['recall'][i]:.3f} f1={metricas['f1'][i]:.3f}")
    return resumen


def main():
    p = argparse.ArgumentParser(description="MLP con Keras")
    p.add_argument("--salida", default="salidas/mlp")
    p.add_argument("--recortes", help="ruta de recortes.npz (generado por imagen.ejecutar_recortes)")
    a = p.parse_args()
    os.makedirs(a.salida, exist_ok=True)

    if not a.recortes:
        f = config.FASHION_REDUCIDO
        (xe, ye), (xp, yp) = mlp.cargar_fashion_mnist(f["entrenamiento"], f["prueba"], f["semilla"])
        experimento("fashion_mnist", xe, ye, xp, yp, NOMBRES_FASHION, a.salida)
        return

    d = np.load(a.recortes)
    x = d["x"].astype(np.float32) / 255.0
    y, clases = d["y"], [str(c) for c in d["clases"]]
    prueba = mlp.dividir_por_tiempo(d["frame_id"], config.RECORTES["fraccion_prueba"])
    xe, ye, xp, yp = x[~prueba], y[~prueba], x[prueba], y[prueba]
    conteo = np.bincount(ye, minlength=len(clases))
    pesos = {i: float(len(ye) / (len(clases) * max(c, 1))) for i, c in enumerate(conteo)}   # compensa el desbalance
    experimento("recortes_sin_pesos", xe, ye, xp, yp, clases, a.salida)
    experimento("recortes_con_pesos", xe, ye, xp, yp, clases, a.salida, pesos_clase=pesos)


if __name__ == "__main__":
    main()
