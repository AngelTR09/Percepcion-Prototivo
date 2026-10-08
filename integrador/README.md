# Integrador: VIDEO + IMAGEN

> **Atribución — grupo VIDEO.** `video_base/` es una copia sin modificar del proyecto *Sistema de Supervisión Inteligente* del grupo VIDEO: https://github.com/cesarcabanillas1921-bit/SistemaSupervisionExamenes (commit `265366d`). Autoría y derechos son de su grupo; el repositorio de origen no declara licencia. Se incluye solo para ejecutar la secuencia VIDEO → IMAGEN del curso, y solo `imagen/adaptador_video.py` lo importa.

- `video_base/`: copia de solo lectura del código del grupo VIDEO (repo `SistemaSupervisionExamenes`, commit `265366d`). No se modifica.
- `imagen/`: módulo del grupo IMAGEN. Solo `adaptador_video.py` conoce el código de VIDEO.
- `docs/`: inventario de VIDEO, solicitudes, requisitos y [AVANCE_S07.md](docs/AVANCE_S07.md).

## Ejecutar (Debian)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-imagen.txt pygame
./correr_todo.sh                      # VIDEO y luego IMAGEN sobre datos/padang.webm
```
Secuencia completa VIDEO → IMAGEN con salida de video (modo producción):
```bash
python -m imagen.ejecutar_integrado --fuente datos/padang.webm --salida salidas/integrado
python -m imagen.ejecutar_integrado --fuente 0 --mostrar      # cámara en vivo, q para salir
```
Genera `video_integrado.mp4` (frame mejorado + cajas de VIDEO), `imagen.jsonl`, `imagen.csv` y `resumen_integrado.json` con los FPS.

Mejorar las **salidas de VIDEO** (capturas de alerta, imágenes o videos ya generados), sin tocar su código:
```bash
python -m imagen.mejorar_salidas --entrada video_base/alertas --salida salidas/mejoradas
python -m imagen.mejorar_salidas --entrada resultado.mp4 --salida salidas/mejoradas
python -m imagen.mejorar_salidas --entrada video_base/alertas --salida salidas/mejoradas --vigilar   # atento a alertas nuevas
```
Por cada archivo decide qué aplicar (gamma si está oscuro, filtro si hay ruido, CLAHE si falta contraste) y guarda la versión mejorada, una comparación lado a lado y `calidad_antes_despues.csv`. Si la imagen ya está bien, no la toca.

**Alerta de VIDEO + mejora instantánea.** Aplica la misma regla de VIDEO (objeto no permitido visible `TIEMPO_ALERTA` segundos) con su detector, su lista de objetos no permitidos y su `guardar_alerta`. Al dispararse, IMAGEN mejora el frame completo y recorta y mejora el objeto que la activó:
```bash
python -m imagen.alerta_mejorada --fuente datos/examen.mp4 --salida salidas/alertas
python -m imagen.alerta_mejorada --fuente 0 --mostrar        # cámara en vivo
```
Para probarlo con un video sin objetos prohibidos: `--tambien-prohibir mochila --tiempo 1 --degradar`.
Guarda `*_frame_mejorado.jpg`, `*_objeto_comparacion.jpg` y `alertas_mejoradas.jsonl` (calidad antes/después y ms). Requiere `pygame`, dependencia de VIDEO.

Por separado:
```bash
python -m imagen.ejecutar_video --video datos/padang.webm --salida salidas/video --cada-n 6
python -m imagen.ejecutar --video datos/padang.webm --salida salidas/imagen --cada-n 60 --luz 0.15
python -m pytest -q
```

## Resultados (carpeta `salidas/`, no se sube a git)
- `video/detecciones.jsonl`: detecciones de VIDEO por frame (clase, confianza, bbox, `timestamp`).
- `video/video_anotado.mp4`, `video/resumen_video.json`.
- `imagen/imagen.jsonl` e `imagen.csv`: calidad, objetos y características por frame para PREDICCIÓN.
- `imagen/resumen.json` y `imagen/evidencias/`: métricas y antes/después.

`datos/padang.webm`: Wikimedia Commons, CC BY-SA 4.0.
