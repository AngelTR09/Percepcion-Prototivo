"""ÚNICO punto de contacto con el código de VIDEO (solo lectura, copia en video_base/).

Traduce lo que VIDEO entrega a nuestro contrato de entrada. Ver docs/SOLICITUDES_A_VIDEO.md.
"""
import os
import sys

import cv2

RUTA_VIDEO_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "video_base"
)


def leer_frames(ruta_video, cada_n=1):
    """Genera dicts del contrato de entrada. VIDEO solo lee cámara, así que abrimos el archivo aquí
    (solicitud #2) y calculamos frame_id/timestamp (solicitud #1)."""
    captura = cv2.VideoCapture(ruta_video)
    if not captura.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {ruta_video}")
    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
    fuente = os.path.basename(ruta_video)
    frame_id = 0
    try:
        while True:
            correcto, frame = captura.read()
            if not correcto:
                break
            if frame_id % cada_n == 0:
                yield {
                    "frame_id": frame_id,
                    "timestamp": frame_id / fps,
                    "fps": fps,
                    "fuente": fuente,
                    "frame": frame,
                }
            frame_id += 1
    finally:
        captura.release()


def cargar_detector():
    """Devuelve detectar_objetos de VIDEO. Importación diferida porque carga el modelo
    al importar (solicitud #4). Convierte su salida a {clase, confianza, bbox=[x1,y1,x2,y2]}."""
    if RUTA_VIDEO_BASE not in sys.path:
        sys.path.insert(0, RUTA_VIDEO_BASE)
    from modulos.deteccion import detectar_objetos  # noqa: E402  (código de VIDEO)

    def detectar(frame):
        return [
            {
                "clase": d["clase"],
                "confianza": d["confianza"],
                "bbox": [d["x1"], d["y1"], d["x2"], d["y2"]],
            }
            for d in detectar_objetos(frame)
        ]

    return detectar


def cargar_dibujante():
    """Devuelve una función que dibuja con dibujar_detecciones de VIDEO (cajas azul/rojo)."""
    if RUTA_VIDEO_BASE not in sys.path:
        sys.path.insert(0, RUTA_VIDEO_BASE)
    from modulos.deteccion import dibujar_detecciones  # noqa: E402  (código de VIDEO)

    def dibujar(frame, detecciones):
        originales = [
            {"clase": d["clase"], "confianza": d["confianza"],
             "x1": d["bbox"][0], "y1": d["bbox"][1], "x2": d["bbox"][2], "y2": d["bbox"][3]}
            for d in detecciones
        ]
        return dibujar_detecciones(frame.copy(), originales)

    return dibujar


def leer_fuente(fuente, cada_n=1):
    """Fuente en vivo o archivo. Si `fuente` es un número, usa la cámara a través de las funciones
    de VIDEO (iniciar_camara/obtener_frame); si es una ruta, lee el archivo con leer_frames."""
    if not str(fuente).isdigit():
        yield from leer_frames(fuente, cada_n)
        return
    import time
    if RUTA_VIDEO_BASE not in sys.path:
        sys.path.insert(0, RUTA_VIDEO_BASE)
    from configuracion import config as config_video  # noqa: E402  (código de VIDEO)
    from modulos.camara import cerrar_camara, iniciar_camara, obtener_frame  # noqa: E402

    config_video.CAMARA = int(fuente)  # VIDEO lee este valor al abrir la cámara
    camara = iniciar_camara()
    if camara is None:
        raise RuntimeError(f"No se pudo abrir la cámara {fuente}")
    inicio, frame_id = time.perf_counter(), 0
    try:
        while True:
            frame = obtener_frame(camara)
            if frame is None:
                break
            if frame_id % cada_n == 0:
                yield {"frame_id": frame_id, "timestamp": time.perf_counter() - inicio,
                       "fps": 0.0, "fuente": f"camara{fuente}", "frame": frame}
            frame_id += 1
    finally:
        cerrar_camara(camara)


def cargar_alerta():
    """Devuelve las piezas de la alerta de VIDEO, sin modificarlas:
    (es_no_permitido, tiempo_alerta, guardar_alerta). guardar_alerta es la suya: guarda el JPG en
    video_base/alertas/ y suena el aviso; devuelve la ruta del archivo."""
    if RUTA_VIDEO_BASE not in sys.path:
        sys.path.insert(0, RUTA_VIDEO_BASE)
    from configuracion.config import TIEMPO_ALERTA  # noqa: E402  (código de VIDEO)
    from modulos.registro import guardar_alerta  # noqa: E402
    from principal import es_no_permitido  # noqa: E402  (solo importa; su bucle no se ejecuta)

    return es_no_permitido, TIEMPO_ALERTA, guardar_alerta
