"""S14: búsqueda de hiperparámetros y tabla comparativa de modelos, en el entorno del laboratorio (`requirements-lab.txt`).

  python -m imagen.ejecutar_hiperparametros --salida salidas/s14 \\
      --recortes32 salidas/recortes/recortes.npz --recortes96 salidas/recortes96/recortes.npz

Fase 1 (búsqueda): MLP y CNN, búsqueda aleatoria, validación por tiempo dentro del entrenamiento. El conjunto de prueba no participa.
Fase 2 (comparación): los mejores de cada familia y los de S12/S13, reentrenados con varias semillas y medidos UNA vez en la prueba,
sin rotar y rotada 25° (S04). Salidas: busqueda_*.csv, efectos_marginales.json y tabla_comparativa.csv/json.
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from . import cnn, config, hiperparametros as hp_, mlp


def _dividir(ruta):
    """(x_ajuste, y_ajuste, x_validacion, y_validacion, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba, clases).
    Prueba = últimos frames; validación = últimos frames del entrenamiento. Ambas por tiempo."""
    d = np.load(ruta)
    x = d["x"].astype(np.float32) / 255.0
    y, f, clases = d["y"], d["frame_id"], [str(c) for c in d["clases"]]
    prueba = mlp.dividir_por_tiempo(f, config.RECORTES["fraccion_prueba"])
    xe, ye, fe = x[~prueba], y[~prueba], f[~prueba]
    val = mlp.dividir_por_tiempo(fe, config.HIPERPARAMETROS["fraccion_validacion"])
    return xe[~val], ye[~val], xe[val], ye[val], xe, ye, x[prueba], y[prueba], clases


def _escribir_busqueda(ruta, resultados):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        claves = list(resultados[0]["hp"])
        w = csv.writer(f)
        w.writerow([*claves, "puntaje_validacion", "desviacion", "brecha_sobreajuste", "epocas", "segundos"])
        for r in resultados:
            w.writerow([*[str(r["hp"][k]).replace(",", ";") for k in claves], round(r["puntaje"], 4), round(r["desviacion"], 4),
                        round(r["brecha_sobreajuste"], 4), round(r["epocas"], 1), round(r["segundos"], 1)])


def fase_busqueda(xf, yf, xv, yv, n_clases, salida):
    datos = {"xf": xf, "yf": yf, "xv": xv, "yv": yv, "n_clases": n_clases}
    forma = xf.shape[1:]
    h = config.HIPERPARAMETROS
    mejores, efectos = {}, {}
    for nombre, espacio, constructor in (("mlp", h["espacio_mlp"], hp_.construir_mlp_hp), ("cnn", h["espacio_cnn"], hp_.construir_cnn_hp)):
        t = time.perf_counter()
        res = hp_.buscar(lambda hp, sem, c=constructor: c(hp, sem, forma, n_clases), espacio, datos)
        _escribir_busqueda(os.path.join(salida, f"busqueda_{nombre}.csv"), res)
        mejores[nombre] = res[0]["hp"]
        efectos[nombre] = {k: hp_.efecto_marginal(res, k) for k in espacio}
        print(f"[búsqueda {nombre}] {len(res)} configuraciones, {time.perf_counter() - t:.0f} s")
        print(f"   mejor: puntaje={res[0]['puntaje']:.3f} ± {res[0]['desviacion']:.3f}  {res[0]['hp']}")
        print(f"   peor : puntaje={res[-1]['puntaje']:.3f}  {res[-1]['hp']}")
    with open(os.path.join(salida, "efectos_marginales.json"), "w", encoding="utf-8") as f:
        json.dump({"mejores": {k: {a: (list(b) if isinstance(b, tuple) else b) for a, b in v.items()} for k, v in mejores.items()},
                   "efectos": efectos}, f, indent=2, ensure_ascii=False)
    return mejores


def _medir(modelo, xp, yp, n, grados):
    fila = {}
    minoritaria = n - 1
    for g in grados:
        m, _ = mlp.evaluar(modelo, cnn.rotar_lote(xp, g), yp, n)
        fila[f"exactitud_{g:.0f}"] = m["exactitud"]
        fila[f"recall_{g:.0f}"] = m["recall"][minoritaria]
        fila[f"f1_{g:.0f}"] = m["f1"][minoritaria]
    return fila


def fase_comparacion(modelos, xe, ye, xp, yp, n, salida):
    """`modelos` = {nombre: (constructor(semilla), x_entrenamiento, x_prueba, parada_temprana)}."""
    grados = config.HIPERPARAMETROS["grados_seleccion"]
    base = config.CNN["semilla"]
    filas = [{"modelo": "linea_base_mayoritaria", "parametros": 0, "exactitud_0": round(mlp.linea_base_mayoritaria(ye, yp), 3)}]
    for nombre, (crear, x_e, x_p, parada) in modelos.items():
        medidas = []
        for i in range(config.HIPERPARAMETROS["semillas_final"]):
            modelo = crear(base + i)
            t = time.perf_counter()
            mlp.entrenar(modelo, x_e, ye, parada_temprana=parada)
            seg = time.perf_counter() - t
            m = _medir(modelo, x_p, yp, n, grados)
            m["segundos"], m["parametros"] = seg, modelo.count_params()
            medidas.append(m)
        fila = {"modelo": nombre, "parametros": medidas[0]["parametros"]}
        for k in medidas[0]:
            if k != "parametros":
                fila[k] = f"{np.mean([m[k] for m in medidas]):.3f} ± {np.std([m[k] for m in medidas]):.3f}"
        filas.append(fila)
        print(f"[{nombre}] " + "  ".join(f"{k}={v}" for k, v in fila.items() if k != "modelo"))
    claves = sorted({k for f in filas for k in f}, key=lambda k: (k != "modelo", k != "parametros", k))
    with open(os.path.join(salida, "tabla_comparativa.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=claves, restval="")
        w.writeheader()
        w.writerows(filas)


def main():
    p = argparse.ArgumentParser(description="Hiperparámetros, regularización y comparación de modelos")
    p.add_argument("--salida", default="salidas/s14")
    p.add_argument("--recortes32", required=True)
    p.add_argument("--recortes96", help="con transfer learning en la comparación")
    a = p.parse_args()
    os.makedirs(a.salida, exist_ok=True)

    xf, yf, xv, yv, xe, ye, xp, yp, clases = _dividir(a.recortes32)
    n, forma = len(clases), xe.shape[1:]
    print(f"ajuste={len(xf)} validación={len(xv)} (clase minoritaria {int((yv == n - 1).sum())}) prueba={len(xp)} — la prueba no se toca en la búsqueda")
    mejores = fase_busqueda(xf, yf, xv, yv, n, a.salida)

    modelos = {
        "mlp_s12": (lambda s: mlp.construir_mlp(forma, n, semilla=s), xe, xp, True),
        "mlp_ajustado": (lambda s: hp_.construir_mlp_hp(mejores["mlp"], s, forma, n), xe, xp, True),
        "cnn_s13_con_aumento": (lambda s: cnn.construir_cnn(forma, n, aumento=True, semilla=s), xe, xp, True),
        "cnn_ajustada": (lambda s: hp_.construir_cnn_hp(mejores["cnn"], s, forma, n), xe, xp, True),
        "cnn_ajustada_sin_parada_temprana": (lambda s: hp_.construir_cnn_hp(mejores["cnn"], s, forma, n), xe, xp, False),
    }
    if a.recortes96:
        _, _, _, _, xe96, ye96, xp96, yp96, _ = _dividir(a.recortes96)
        assert (ye96 == ye).all() and (yp96 == yp).all(), "los recortes de 32 y de 96 px deben salir del mismo video y los mismos frames"
        rgb = lambda x: x[..., ::-1].copy()      # OpenCV guarda BGR; MobileNetV2 espera RGB
        modelos["transfer_cabeza"] = (lambda s: cnn.construir_transfer(n, semilla=s), rgb(xe96), rgb(xp96), True)
    fase_comparacion(modelos, xe, ye, xp, yp, n, a.salida)


if __name__ == "__main__":
    main()
