# Integrador: VIDEO + IMAGEN

- `video_base/`: copia de solo lectura del código del grupo VIDEO (repo `SistemaSupervisionExamenes`, commit `265366d`). No se modifica.
- `imagen/`: módulo del grupo IMAGEN. Solo `adaptador_video.py` conoce el código de VIDEO.
- `docs/`: inventario de VIDEO, solicitudes, requisitos y [AVANCE_S07.md](docs/AVANCE_S07.md).

## Ejecutar (Debian)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-imagen.txt pygame
./correr_todo.sh                      # VIDEO y luego IMAGEN sobre datos/padang.webm
```
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
