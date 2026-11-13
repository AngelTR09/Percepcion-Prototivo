"""Análisis de sesgo por condición (S15, RNF-09, ISO/IEC TR 24027).

Con los datos que hay (un video, sin etiquetas demográficas) no se puede auditar el sesgo hacia personas.
Sí se puede medir si el detector de VIDEO, y la mejora de IMAGEN, rinden parejo entre grupos de objetos:
por clase, por tamaño en el encuadre y por posición (centro o borde), bajo poca luz.
Un grupo cuya tasa de recuperación es mucho menor que la de otros es un sesgo de desempeño a documentar.
"""
from . import config, metricas


def grupo_clase(deteccion, forma):
    return deteccion["clase"]


def grupo_tamano(deteccion, forma):
    x1, y1, x2, y2 = deteccion["bbox"]
    fraccion = max(x2 - x1, 0) * max(y2 - y1, 0) / float(forma[0] * forma[1])
    pequeno, mediano = config.SESGO["tamano_cortes"]
    return "pequeno" if fraccion < pequeno else ("mediano" if fraccion < mediano else "grande")


def grupo_region(deteccion, forma):
    x1, y1, x2, y2 = deteccion["bbox"]
    cx, cy = (x1 + x2) / 2.0 / forma[1], (y1 + y2) / 2.0 / forma[0]
    bajo, alto = config.SESGO["region_central"]
    return "centro" if bajo <= cx <= alto and bajo <= cy <= alto else "borde"


CRITERIOS = {"clase": grupo_clase, "tamano": grupo_tamano, "region": grupo_region}


def recuperadas_por_referencia(referencia, candidatas, iou_minimo):
    """Una bandera por detección de referencia: ¿reaparece (misma clase, IoU >= mínimo, sin reutilizar candidatas)?"""
    usadas, banderas = set(), []
    for r in referencia:
        encontrada = False
        for i, c in enumerate(candidatas):
            if i not in usadas and c["clase"] == r["clase"] and metricas.iou(r["bbox"], c["bbox"]) >= iou_minimo:
                usadas.add(i)
                encontrada = True
                break
        banderas.append(encontrada)
    return banderas


def acumular(tabla, referencia, resultados, forma, iou_minimo):
    """Suma a `tabla[criterio][grupo]` el conteo de referencias y, por cada variante en `resultados`
    ({nombre: detecciones}), cuántas se recuperaron. La referencia define el grupo."""
    banderas = {n: recuperadas_por_referencia(referencia, dets, iou_minimo) for n, dets in resultados.items()}
    for criterio, funcion in CRITERIOS.items():
        for i, r in enumerate(referencia):
            fila = tabla.setdefault(criterio, {}).setdefault(funcion(r, forma), {"n": 0, **{v: 0 for v in resultados}})
            fila["n"] += 1
            for variante in resultados:
                fila[variante] += int(banderas[variante][i])
    return tabla


def tasas(tabla_criterio, variante, minimo=None):
    """{grupo: tasa} de un criterio para una variante, solo con grupos de al menos `minimo` referencias."""
    minimo = config.SESGO["min_por_grupo"] if minimo is None else minimo
    return {g: f[variante] / f["n"] for g, f in tabla_criterio.items() if f["n"] >= minimo}


def brecha(tabla_criterio, variante, minimo=None):
    """Diferencia entre la mejor y la peor tasa de grupo (0 = parejo). None si hay menos de 2 grupos comparables."""
    t = tasas(tabla_criterio, variante, minimo)
    if len(t) < 2:
        return None
    return round(max(t.values()) - min(t.values()), 3)
