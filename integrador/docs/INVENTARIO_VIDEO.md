# Inventario del módulo VIDEO

Fuente: copia de solo lectura en `video_base/` del repo `cesarcabanillas1921-bit/SistemaSupervisionExamenes`, commit `265366d`.
Estado: **auditoría por lectura de código**. Aún no se ejecutó (requiere cámara en vivo).

| Pregunta | Respuesta |
|---|---|
| Problemática | Supervisión de exámenes: detectar objetos no permitidos (teléfono, audífonos, reloj, cuaderno/libro) y alertar si permanecen 5 s |
| Ejecución | `python principal.py` (Python 3.12 según su README); dependencias en `requisitos.txt`: opencv-python, ultralytics, numpy, mediapipe, matplotlib, pygame |
| Entradas | Solo cámara en vivo, índice `CAMARA = 0` (`configuracion/config.py`). No lee archivos de video |
| Entrega | Frame `ndarray` BGR `uint8` por llamada a `obtener_frame(camara)`; lista de detecciones por `detectar_objetos(frame)` |
| Metadatos | **Ninguno**: no hay `frame_id`, `timestamp`, FPS ni nombre de fuente |
| Muestreo | Ninguno: procesa todos los frames a la velocidad de la cámara |
| Audio | **No lo separa.** Solo reproduce `sonido/alerta.mp3` con pygame al alertar |
| Reutilizable | `modulos.camara.iniciar_camara/obtener_frame/cerrar_camara`; `modulos.deteccion.detectar_objetos(frame) -> [{clase, confianza, x1, y1, x2, y2}]`; `dibujar_detecciones(frame, dets)` |
| Procesamiento de imagen | Ninguno propio. Solo YOLO (`modelo_ia/best.pt`, 8 clases) con umbral por clase, y dibujo de cajas |
| Brechas | Rutas y bucle acoplados a `principal.py` (estado global, `imshow`); `deteccion.py` carga el modelo al importar; `modulos/comportamiento.py` vacío; `config.py` tiene valores que el código no usa (`RUTA_MODELO`, `CONFIANZA_MINIMA`) |

## Lo que VIDEO nos puede entregar
- Frames BGR `uint8` (cámara, o video si el adaptador lo abre con `cv2.VideoCapture`).
- Detecciones por frame: clase, confianza y bbox (`detectar_objetos`).
- Modelo YOLO entrenado (`best.pt`) con 8 clases.

## Lo que nos falta y debemos generar nosotros
- `frame_id`, `timestamp` y `fps` (el adaptador los calcula).
- Lectura desde archivo de video para pruebas repetibles.
- Todo el procesamiento de imagen: filtros, mejora, segmentación, características y métricas.
- Salida estructurada (`.jsonl` / `.csv`) para PREDICCIÓN.
- Audio separado: es tarea de VIDEO/AUDIO, no nuestra.
