# Resultados de la ejecución (se copian a `resultados/`)

Generados con `datos/padang.webm` (aula, 1080p, 10 s, CC BY-SA 4.0). Para regenerarlos: `./correr_todo.sh`, `python -m imagen.ejecutar_integrado ...` y luego `python publicar_resultados.py` (los videos se reducen a 640 px).

| Carpeta | Qué contiene | Quién lo usa |
|---|---|---|
| `resultados/imagen_integrado/` | **Salida principal de IMAGEN** sobre el video completo (620 frames): `imagen.jsonl` e `imagen.csv` (una fila por frame con `frame_id`, `timestamp`, contraste, nº de objetos segmentados, densidad de bordes, esquinas, puntos ORB, media HSV, `tiempo_ms`), `video_integrado.mp4` (frame mejorado + cajas de VIDEO) y `resumen_integrado.json` (FPS medidos) | PREDICCIÓN, AUDIO (sincronizan por `timestamp`, en segundos desde el inicio del video) |
| `resultados/video/` | Lo que detecta VIDEO: `detecciones.jsonl` (clase, confianza, bbox por frame), `video_anotado.mp4`, `resumen_video.json` | PREDICCIÓN |
| `resultados/imagen_evaluacion/` | Evaluación con video degradado (poca luz y ruido): PSNR/SSIM, comparación de filtros, detecciones recuperadas y `evidencias/` antes/después | Informe y exposición |
| `resultados/alertas_mejoradas/` | Alertas de VIDEO con el frame y el objeto mejorados (`*_frame_mejorado.jpg`, `*_objeto_comparacion.jpg`) y `alertas_mejoradas.jsonl` | Informe y exposición |
| `resultados/salidas_mejoradas/` | Ejemplo de mejora adaptativa sobre imágenes de alerta, con `calidad_antes_despues.csv` | Informe y exposición |

Las alertas de ejemplo se generaron tratando la mochila como objeto no permitido (el video de prueba no tiene teléfonos ni cuadernos).
