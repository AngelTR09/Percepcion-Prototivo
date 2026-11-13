"""Rendimiento por etapa y FPS de punta a punta (S15, RNF-01).

Cronometra cada etapa del modo producción (normalizar, gamma, filtro, CLAHE, segmentar, características, detector de VIDEO)
para saber dónde se va el tiempo y si el pipeline alcanza tiempo real. No cambia el pipeline: replica sus pasos.
"""
from . import caracteristicas, config, mejoramiento, metricas, preprocesamiento, segmentacion

ETAPAS = ("normalizar", "gamma", "filtro", "clahe", "segmentar", "caracteristicas", "detector")


def medir_etapas(frame, detector, ancho, analisis=True, filtro="gaussiano"):
    """Procesa un frame como `pipeline.procesar_frame_produccion` y devuelve (ms por etapa, detecciones, tamaño procesado).
    Con `analisis=False` se omiten segmentar y características (solo mejora + detección)."""
    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError("frame vacío o corrupto")
    ms = {}

    def etapa(nombre, funcion, *args):
        resultado, tiempo = metricas.cronometrar(funcion, *args)
        ms[nombre] = tiempo
        return resultado

    f = etapa("normalizar", preprocesamiento.normalizar_tamano, frame, ancho)
    iluminado, _ = etapa("gamma", mejoramiento.correccion_gamma, f, config.BRILLO_OBJETIVO)
    filtrado = etapa("filtro", preprocesamiento.aplicar_filtro, iluminado, filtro, config.FILTROS[filtro])
    mejorado = etapa("clahe", mejoramiento.clahe, filtrado, config.CLAHE["clip"], config.CLAHE["grilla"])
    if analisis:
        objetos, _ = etapa("segmentar", segmentacion.segmentar, mejorado, config.AREA_MINIMA,
                           config.MAX_OBJETOS, config.KERNEL_MORFOLOGIA)
        etapa("caracteristicas", caracteristicas.extraer, mejorado, len(objetos), config)
    detecciones = etapa("detector", detector, mejorado)
    return ms, detecciones, mejorado.shape[:2]


def resumir(lista_ms):
    """Promedio por etapa, total y FPS a partir de varias salidas de `medir_etapas`."""
    if not lista_ms:
        raise ValueError("no hay mediciones")
    n = len(lista_ms)
    promedio = {e: sum(m.get(e, 0.0) for m in lista_ms) / n for e in ETAPAS}
    total = sum(promedio.values())
    return {
        "ms_por_etapa": {e: round(v, 2) for e, v in promedio.items()},
        "ms_total": round(total, 2),
        "fps": round(1000.0 / total, 2) if total > 0 else float("inf"),
    }


def es_tiempo_real(fps, umbral=None):
    return fps >= (config.FPS_TIEMPO_REAL if umbral is None else umbral)
