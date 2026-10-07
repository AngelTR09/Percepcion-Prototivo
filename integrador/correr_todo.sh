#!/usr/bin/env bash
# Ejecuta VIDEO y luego IMAGEN sobre el mismo video. Uso: ./correr_todo.sh [video] [salida]
set -e
cd "$(dirname "$0")"
VIDEO="${1:-datos/padang.webm}"
SALIDA="${2:-salidas}"
python -m imagen.ejecutar_video --video "$VIDEO" --salida "$SALIDA/video" --cada-n 6
python -m imagen.ejecutar --video "$VIDEO" --salida "$SALIDA/imagen" --cada-n 60 --luz 0.15
