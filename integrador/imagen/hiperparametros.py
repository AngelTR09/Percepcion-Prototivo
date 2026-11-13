"""S14: búsqueda de hiperparámetros, regularización y evaluación comparativa.

Protocolo (para no engañarse): los hiperparámetros se eligen SOLO con una validación tomada del final del entrenamiento
(por tiempo, sin mezclar frames). Ninguna función de este módulo recibe el conjunto de prueba. Cada configuración se entrena con
varias semillas y se puntúa por el F1 de la clase minoritaria, promediado entre la validación normal y la rotada (S04),
porque la robustez a la rotación fue el punto débil medido en S13. El conjunto de prueba se usa una sola vez, en
`ejecutar_hiperparametros`, con la configuración ya elegida.
"""
import itertools
import time

import numpy as np

from . import cnn, config, mlp


def muestrear(espacio, n, semilla):
    """`n` configuraciones distintas tomadas al azar (reproducible) del espacio; si el espacio es menor, devuelve todas."""
    claves = list(espacio)
    todas = [dict(zip(claves, combo)) for combo in itertools.product(*(espacio[k] for k in claves))]
    if n >= len(todas):
        return todas
    idx = np.random.default_rng(semilla).choice(len(todas), size=n, replace=False)
    return [todas[i] for i in sorted(idx)]


def _f1_minoritaria(modelo, x, y, n_clases):
    metricas, _ = mlp.evaluar(modelo, x, y, n_clases)
    return metricas["f1"][n_clases - 1], metricas["exactitud"]


def entrenar_validando(modelo, xf, yf, xv, yv, epocas=None, paciencia=None):
    """Entrena con `validation_data` explícita (la validación por tiempo) y parada temprana sobre su pérdida."""
    p = config.MLP
    k = mlp._keras()
    parada = k.callbacks.EarlyStopping(monitor="val_loss", patience=p["paciencia"] if paciencia is None else paciencia,
                                       restore_best_weights=True)
    h = modelo.fit(xf, yf, validation_data=(xv, yv), epochs=p["epocas"] if epocas is None else epocas,
                   batch_size=p["lote"], callbacks=[parada], verbose=0, shuffle=True)
    return {k_: [float(v) for v in vs] for k_, vs in h.history.items()}


def evaluar_configuracion(construir, hp, datos, semillas, grados=None):
    """Entrena `construir(hp, semilla)` con cada semilla y devuelve media y desviación del puntaje de validación.
    `datos` = {xf, yf, xv, yv, n_clases} (ajuste y validación). Puntaje = F1 de la clase minoritaria promediado entre rotaciones."""
    grados = config.HIPERPARAMETROS["grados_seleccion"] if grados is None else grados
    giradas = {g: cnn.rotar_lote(datos["xv"], g) for g in grados}
    puntajes, huecos, epocas, tiempos = [], [], [], []
    for semilla in semillas:
        modelo = construir(hp, semilla)
        inicio = time.perf_counter()
        h = entrenar_validando(modelo, datos["xf"], datos["yf"], datos["xv"], datos["yv"])
        tiempos.append(time.perf_counter() - inicio)
        f1s = [_f1_minoritaria(modelo, x, datos["yv"], datos["n_clases"])[0] for x in giradas.values()]
        puntajes.append(float(np.mean(f1s)))
        huecos.append(h["accuracy"][-1] - h["val_accuracy"][-1])   # sobreajuste: exactitud de entrenamiento menos la de validación
        epocas.append(len(h["loss"]))
    return {"puntaje": float(np.mean(puntajes)), "desviacion": float(np.std(puntajes)), "brecha_sobreajuste": float(np.mean(huecos)),
            "epocas": float(np.mean(epocas)), "segundos": float(np.mean(tiempos)), "hp": hp}


def buscar(construir, espacio, datos, n_configuraciones=None, semillas=None, semilla_busqueda=None):
    """Búsqueda aleatoria. Devuelve los resultados ordenados de mejor a peor puntaje de validación."""
    h = config.HIPERPARAMETROS
    n = h["n_configuraciones"] if n_configuraciones is None else n_configuraciones
    semillas = range(h["semillas_busqueda"]) if semillas is None else semillas
    base = config.CNN["semilla"]
    resultados = [evaluar_configuracion(construir, hp, datos, [base + s for s in semillas])
                  for hp in muestrear(espacio, n, h["semilla_busqueda"] if semilla_busqueda is None else semilla_busqueda)]
    return sorted(resultados, key=lambda r: -r["puntaje"])


def efecto_marginal(resultados, clave):
    """Puntaje medio de validación por valor de un hiperparámetro (promedia sobre las demás elecciones).
    Orientativo: con pocas configuraciones, los demás hiperparámetros no están balanceados entre valores."""
    grupos = {}
    for r in resultados:
        grupos.setdefault(str(r["hp"][clave]), []).append(r["puntaje"])
    return {v: round(float(np.mean(p)), 4) for v, p in grupos.items()}


def construir_mlp_hp(hp, semilla, forma, n_clases):
    return mlp.construir_mlp(forma, n_clases, capas=hp["capas"], dropout=hp["dropout"], l2=hp["l2"], tasa=hp["tasa"], semilla=semilla)


def construir_cnn_hp(hp, semilla, forma, n_clases):
    return cnn.construir_cnn(forma, n_clases, filtros=hp["filtros"], denso=hp["denso"], dropout=hp["dropout"], l2=hp["l2"],
                             tasa=hp["tasa"], aumento=hp["aumento"], semilla=semilla)
