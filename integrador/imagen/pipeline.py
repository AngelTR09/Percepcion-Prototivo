"""Pipeline por frame: entrada del contrato (sección 4) -> registro de salida."""
import time

from . import caracteristicas, config, mejoramiento, metricas, preprocesamiento, segmentacion


def procesar_frame(entrada, detector, degradar=True, degradacion=None):
    """Procesa un frame. Devuelve (registro_json, imagenes) donde imagenes sirve de evidencia (RF-11).

    Con degradar=True se oscurece y ensucia el frame para tener una referencia limpia
    con la cual medir la mejora; con False se mejora el frame tal cual llega.
    """
    inicio = time.perf_counter()
    if entrada.get("frame") is None or entrada["frame"].size == 0:
        raise ValueError("frame vacío o corrupto")

    referencia = preprocesamiento.normalizar_tamano(entrada["frame"], config.ANCHO_PROCESO)
    if degradar:
        d = degradacion or config.DEGRADACION
        entrada_procesar = mejoramiento.degradar(
            referencia, d["factor_luz"], d["sigma_ruido"], d["semilla"] + entrada["frame_id"]
        )
    else:
        entrada_procesar = referencia

    # S04: primero se corrige la luz (gamma), así el filtro se compara contra una referencia a igual brillo
    iluminado, gamma = mejoramiento.correccion_gamma(entrada_procesar, config.BRILLO_OBJETIVO)
    # S03: elegir el filtro con métricas (PSNR/SSIM/tiempo)
    filtro, filtrado, tabla_filtros = preprocesamiento.comparar_filtros(
        iluminado, referencia, config.FILTROS
    )
    # S04: CLAHE al final para no amplificar el ruido antes de filtrarlo
    mejorado = mejoramiento.clahe(filtrado, config.CLAHE["clip"], config.CLAHE["grilla"])

    # S05: segmentación y S06: características sobre el frame mejorado
    objetos, mascara = segmentacion.segmentar(
        mejorado, config.AREA_MINIMA, config.MAX_OBJETOS, config.KERNEL_MORFOLOGIA
    )
    rasgos = caracteristicas.extraer(mejorado, len(objetos), config)

    # RF-09: ¿cuánto ayuda la mejora al detector de VIDEO?
    det_ref = detector(referencia)
    det_deg = detector(entrada_procesar)
    det_mej = detector(mejorado)
    solo_filtro = preprocesamiento.aplicar_filtro(entrada_procesar, filtro, config.FILTROS[filtro])
    det_fil = detector(solo_filtro)
    n_ref = len(det_ref)
    rec_deg = metricas.recuperadas(det_ref, det_deg, config.IOU_MINIMO)
    rec_mej = metricas.recuperadas(det_ref, det_mej, config.IOU_MINIMO)
    rec_fil = metricas.recuperadas(det_ref, det_fil, config.IOU_MINIMO)

    registro = {
        "version_contrato": config.VERSION_CONTRATO,
        "frame_id": entrada["frame_id"],
        "timestamp": round(entrada["timestamp"], 3),
        "calidad": {
            "psnr": round(metricas.psnr(referencia, mejorado), 3),
            "ssim": round(metricas.ssim(referencia, mejorado), 4),
            "psnr_degradado": round(metricas.psnr(referencia, entrada_procesar), 3),
            "ssim_degradado": round(metricas.ssim(referencia, entrada_procesar), 4),
            "contraste": round(metricas.contraste(mejorado), 3),
            "filtro_usado": filtro,
            "gamma": round(gamma, 3),
        },
        "objetos": objetos,
        "caracteristicas": rasgos,
        "deteccion_video": {  # campo adicional a la v0.1 del contrato
            "referencia": n_ref,
            "sobre_degradado": rec_deg,
            "sobre_mejorado": rec_mej,
            "sobre_solo_filtro": rec_fil,
            "clases_mejorado": sorted({d["clase"] for d in det_mej}),
        },
        "tiempo_ms": 0.0,
    }
    registro["tiempo_ms"] = round((time.perf_counter() - inicio) * 1000, 1)
    imagenes = {
        "1_referencia": referencia,
        "2_degradado": entrada_procesar,
        f"3_filtrado_{filtro}": filtrado,
        "4_mejorado": mejorado,
        "5_mascara": mascara,
    }
    return registro, imagenes, tabla_filtros
