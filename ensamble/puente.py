"""Traduce lo que producen VIDEO e IMAGEN al JSON que consume PREDICCION (`POST /api/prediccion/integrar`).

Contrato de salida (ver prediccion/docs/CONTRATO_MODULOS.md):
  video.frames[].detecciones[]   {clase, confianza}                       <- detector de VIDEO sobre el frame original
  imagen.frames[].objetos[]      {clase, confianza_video, confianza_procesada, estado}
                                                                           <- el mismo detector sobre el frame mejorado
Estados: `confirmada` = VIDEO ya lo veia y la mejora lo sostiene; `recuperada` = solo aparece tras mejorar.
Lo que VIDEO veia y la mejora pierde se descarta (PREDICCION lo ignora como `no_confirmada`).
"""
import ensamble  # noqa: F401  (prepara sys.path)
from imagen import config, metricas

MODELO_VIDEO = "SistemaSupervisionExamenes/best.pt"
CONTRASTE_REFERENCIA = 64.0  # desviacion estandar de luminancia que se considera calidad plena


def calidad_normalizada(contraste):
    """Contraste de luminancia (0-255) -> calidad en [0, 1]."""
    return max(0.0, min(1.0, float(contraste) / CONTRASTE_REFERENCIA))


def _emparejar(det_video, det_mejorada):
    """Empareja por clase e IoU (voraz, uno a uno). Devuelve [(det_mejorada, det_video | None)]."""
    usadas = set()
    pares = []
    for m in sorted(det_mejorada, key=lambda d: -d["confianza"]):
        mejor, mejor_iou = None, 0.0
        for i, v in enumerate(det_video):
            if i in usadas or v["clase"] != m["clase"]:
                continue
            valor = metricas.iou(v["bbox"], m["bbox"])
            if valor >= config.IOU_MINIMO and valor > mejor_iou:
                mejor, mejor_iou = i, valor
        if mejor is not None:
            usadas.add(mejor)
        pares.append((m, det_video[mejor] if mejor is not None else None))
    return pares


def construir_payload(frame_id, timestamp, contraste, det_video, det_mejorada, objetos_permitidos=None):
    """Arma el cuerpo de `integrar_fuentes` para un frame."""
    calidad = round(calidad_normalizada(contraste), 3)
    objetos = []
    for indice, (m, v) in enumerate(_emparejar(det_video, det_mejorada)):
        objetos.append({
            "id": f"objeto-{indice + 1}",
            "clase": m["clase"],
            "confianza_video": round(float(v["confianza"]), 3) if v else 0.0,
            "confianza_procesada": round(float(m["confianza"]), 3),
            "estado": "confirmada" if v else "recuperada",
        })
    ts = round(float(timestamp), 3)
    return {
        "segundo_actual": ts,
        "ventana_segundos": 60,
        "objetos_permitidos": list(objetos_permitidos or []),
        "video": {
            "modelo": MODELO_VIDEO,
            "frames": [{
                "frame_id": frame_id, "timestamp": ts, "calidad": calidad,
                "detecciones": [{"clase": d["clase"], "confianza": round(float(d["confianza"]), 3)} for d in det_video],
            }],
        },
        "imagen": {
            "modulo": "IMAGEN",
            "modelo": "SistemaSupervisionExamenes-Imagen",
            "frames": [{"frame_id": frame_id, "timestamp": ts, "calidad_normalizada": calidad, "objetos": objetos}],
        },
    }
