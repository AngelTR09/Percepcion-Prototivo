"""Orquesta un frame completo: VIDEO (detecta) -> IMAGEN (mejora) -> PREDICCION (evalua)."""
import base64
import os
import threading
import time

import cv2
import numpy as np

import ensamble  # noqa: F401  (prepara sys.path)
from imagen import config, pipeline, preprocesamiento

from . import puente

_detector = None
_candado = threading.Lock()


class DetectorNoDisponible(RuntimeError):
    """VIDEO no se puede ejecutar en este entorno (falta ultralytics/torch o el modelo)."""


class ImagenInvalida(ValueError):
    pass


def _desactivado():
    return os.getenv("INTEGRADO_DETECTOR", "video").strip().lower() in ("off", "ninguno")


def establecer_detector(detector):
    """Inyecta un detector (pruebas). `None` fuerza a recargar el de VIDEO."""
    global _detector
    _detector = detector


def obtener_detector():
    """Carga perezosa del detector de VIDEO: arrancar el servidor no paga el costo de YOLO."""
    global _detector
    if _detector is None:
        with _candado:
            if _detector is None:
                if _desactivado():
                    raise DetectorNoDisponible("El detector de VIDEO esta desactivado (INTEGRADO_DETECTOR=off).")
                try:
                    from imagen import adaptador_video
                    _detector = adaptador_video.cargar_detector()
                except Exception as error:  # ultralytics ausente, pesos ausentes...
                    raise DetectorNoDisponible(
                        f"No se pudo cargar el detector de VIDEO: {type(error).__name__}") from error
    return _detector


def estado_detector():
    if _detector is not None:
        return "cargado"
    if _desactivado():
        return "desactivado"
    try:
        import importlib.util
        if importlib.util.find_spec("ultralytics") is None:
            return "no_instalado"
    except Exception:
        return "no_instalado"
    return "disponible_sin_cargar"


def decodificar(imagen_b64):
    if not isinstance(imagen_b64, str) or not imagen_b64.strip():
        raise ImagenInvalida("imagen_b64 es obligatorio")
    if imagen_b64.startswith("data:") and "," in imagen_b64[:100]:  # data URL
        imagen_b64 = imagen_b64.split(",", 1)[1]
    try:
        crudo = base64.b64decode(imagen_b64)
    except Exception:
        raise ImagenInvalida("imagen_b64 no es base64 valido") from None
    frame = cv2.imdecode(np.frombuffer(crudo, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ImagenInvalida("No se pudo decodificar la imagen")
    return frame


def codificar_jpg(frame, calidad=85):
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, calidad])
    return base64.b64encode(buffer.tobytes()).decode("ascii") if ok else None


def analizar_frame(frame, *, timestamp=0.0, frame_id=0, objetos_permitidos=None, devolver_imagen=False):
    """Ejecuta la cadena completa sobre un frame BGR. Devuelve un dict serializable."""
    from backend.modulos.prediccion.integracion import integrar_fuentes

    detector = obtener_detector()
    referencia = preprocesamiento.normalizar_tamano(frame, config.ANCHO_PROCESO)
    det_video = detector(referencia)                         # VIDEO sobre el frame original
    entrada = {"frame_id": frame_id, "timestamp": timestamp, "frame": referencia}
    registro, mejorado, det_mejorada = pipeline.procesar_frame_produccion(entrada, detector)  # IMAGEN
    payload = puente.construir_payload(
        frame_id, timestamp, registro["calidad"]["contraste"], det_video, det_mejorada, objetos_permitidos
    )
    prediccion = integrar_fuentes(payload)                   # PREDICCION
    salida = {
        "video": {"detecciones": [{k: d[k] for k in ("clase", "confianza", "bbox")} for d in det_video]},
        "imagen": {
            "calidad": registro["calidad"],
            "caracteristicas": registro["caracteristicas"],
            "n_objetos_segmentados": len(registro["objetos"]),
            "objetos_verificados": payload["imagen"]["frames"][0]["objetos"],
            "tiempo_ms": registro["tiempo_ms"],
        },
        "prediccion": prediccion,
        "contrato_enviado_a_prediccion": payload,
    }
    if devolver_imagen:
        salida["imagen"]["mejorada_b64"] = codificar_jpg(mejorado)
    return salida


_candado_vivo = threading.Lock()

# Lo mismo que dibuja VIDEO (modulos/deteccion.py): azul = permitido, rojo = el resto.
PERMITIDOS_VIDEO = {"persona", "mochila"}
NOMBRES_VIDEO = {"persona": "Persona", "mochila": "Mochila", "telefono": "Telefono", "audifonos": "Audifonos",
                 "reloj": "Reloj", "cuaderno": "Cuaderno/Libro", "libro": "Cuaderno/Libro", "laptop": "Laptop"}


def detectar_en_vivo(jpeg_bytes):
    """Deteccion de VIDEO sobre un fotograma, para dibujar sus recuadros en vivo.

    Usa el detector del grupo VIDEO tal cual (sus umbrales por clase). Cajas en pixeles del frame recibido.
    """
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ImagenInvalida("No se pudo decodificar la imagen")
    detector = obtener_detector()
    alto, ancho = frame.shape[:2]
    inicio = time.perf_counter()
    with _candado_vivo:  # un solo hilo a la vez sobre el modelo
        detecciones = detector(frame)
    return {
        "ancho": ancho, "alto": alto, "latencia_ms": round((time.perf_counter() - inicio) * 1000, 1),
        "objetos": [{
            "clase": d["clase"], "etiqueta": NOMBRES_VIDEO.get(d["clase"], d["clase"]),
            "confianza": round(float(d["confianza"]), 4), "permitido": d["clase"] in PERMITIDOS_VIDEO,
            "caja": {"x1": d["bbox"][0], "y1": d["bbox"][1], "x2": d["bbox"][2], "y2": d["bbox"][3]},
        } for d in detecciones],
    }
