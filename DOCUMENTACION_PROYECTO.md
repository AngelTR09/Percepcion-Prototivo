# Documentación del proyecto — Grupo IMAGEN

Percepción Computacional (ISIA 111), UPAO 2026-20. Este documento describe lo que existe hoy, cómo ejecutarlo y qué falta. Última actualización: 2026-10-07.

> El `README.md` de la raíz describe el panel anterior (Render y Docker del avance S02). Todo lo nuevo está en [`integrador/`](integrador/) y en la rama `imagen-integracion`.

## 1. Qué es

Sistema único hecho por cuatro grupos: **VIDEO → (IMAGEN ‖ AUDIO) → PREDICCIÓN**. Este repo contiene el módulo de **IMAGEN** conectado al código de VIDEO.

- **Problemática:** en exámenes supervisados con cámara, la poca luz y el ruido hacen que el detector de VIDEO pierda objetos. IMAGEN mejora el frame antes de detectar, mide si eso ayuda y entrega calidad, objetos y características con `timestamp` a PREDICCIÓN.
- **Regla:** el código de VIDEO es de solo lectura. Está en `integrador/video_base/` (commit `265366d` de su repo) y solo `integrador/imagen/adaptador_video.py` lo importa.

## 2. Estructura

```
integrador/
├── video_base/        código de VIDEO (no se modifica) + modelo best.pt
├── imagen/            módulo de IMAGEN
│   ├── adaptador_video.py    único punto de contacto con VIDEO
│   ├── preprocesamiento.py   grises/HSV, redimensión, 4 filtros de ruido      (S02-S03)
│   ├── mejoramiento.py       gamma, CLAHE, ecualización, degradación sintética (S04)
│   ├── segmentacion.py       Otsu + morfología + contornos                    (S05)
│   ├── caracteristicas.py    Canny, Harris, ORB, media HSV                    (S06)
│   ├── metricas.py           PSNR, SSIM, MSE, contraste, IoU, tiempos
│   ├── pipeline.py           modo evaluación y modo producción
│   ├── exportar.py           .jsonl, .csv y evidencias
│   ├── ejecutar.py           evaluación con degradación y comparación de filtros
│   ├── ejecutar_video.py     solo VIDEO: guarda sus detecciones
│   └── ejecutar_integrado.py secuencia completa VIDEO → IMAGEN con salida de video
├── docs/              inventario de VIDEO, requisitos, solicitudes, avance S07
├── tests/             8 pruebas (pytest)
├── datos/             videos de prueba
└── correr_todo.sh     VIDEO y luego IMAGEN sobre un video
```

## 3. Cómo se ejecuta (Debian)

```bash
cd integrador
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-imagen.txt pygame

# Secuencia completa con salida de video (modo producción)
python -m imagen.ejecutar_integrado --fuente datos/padang.webm --salida salidas/integrado
python -m imagen.ejecutar_integrado --fuente 0 --mostrar          # cámara en vivo, q para salir

# Evaluación: degrada el video y mide cuánto ayuda la mejora al detector
python -m imagen.ejecutar --video datos/padang.webm --salida salidas/imagen --cada-n 60 --luz 0.15

# Solo el detector de VIDEO / todo junto / pruebas
python -m imagen.ejecutar_video --video datos/padang.webm --salida salidas/video --cada-n 6
./correr_todo.sh
python -m pytest -q
```

La carpeta `salidas/` no se sube a git; se genera al ejecutar.

## 4. Qué hace el prototipo de VIDEO

Abre la cámara 0, pasa cada frame por YOLO (`best.pt`, 8 clases: persona, mochila, teléfono, cuaderno, libro, audífonos, reloj, laptop), dibuja cajas (azul permitido, rojo no permitido) y, si un objeto no permitido permanece 5 s, guarda una captura y suena una alerta. No entrega `frame_id` ni `timestamp`, no lee archivos de video y no separa audio. Detalle en [docs/INVENTARIO_VIDEO.md](integrador/docs/INVENTARIO_VIDEO.md); lo que pedimos a su grupo, en [docs/SOLICITUDES_A_VIDEO.md](integrador/docs/SOLICITUDES_A_VIDEO.md).

## 5. Qué hace IMAGEN

Dos modos:

- **Producción** (`ejecutar_integrado`): gamma → filtro Gaussiano → CLAHE → segmentación → características → una pasada del detector de VIDEO sobre el frame mejorado. No necesita imagen de referencia.
- **Evaluación** (`ejecutar`): oscurece y ensucia el video (simulación), compara Media, Gaussiano, Mediana y Bilateral con PSNR/SSIM/tiempo, y cuenta cuántas detecciones de VIDEO reaparecen respecto al video limpio.

**Mejora de las salidas de VIDEO** (`mejorar_salidas`): lee las capturas de alerta o videos que VIDEO ya generó, mide brillo, contraste, nitidez y ruido sin necesitar referencia, y aplica solo lo necesario (gamma, filtro, CLAHE). Con `--vigilar` procesa las alertas nuevas conforme aparecen. Probado con una alerta oscura simulada: brillo 0.10 → 0.49, contraste 12.7 → 40.0, ruido 4.1 → 0.9; una imagen normal no se modificó y un archivo corrupto se omitió sin detener el lote.

**Alerta + mejora instantánea** (`alerta_mejorada`): usa el detector, la lista de objetos no permitidos, el tiempo (`TIEMPO_ALERTA`) y `guardar_alerta` de VIDEO sin modificarlos. Cuando una alerta se dispara, IMAGEN mejora el frame y el recorte del objeto (limpia, aclara y sube contraste solo si hace falta) y registra calidad antes/después. Se probó con `padang.webm` oscurecido tratando la mochila como no permitida (VIDEO no la prohíbe): 3 alertas, mejora en ~110-130 ms por alerta (la primera ~500 ms por arranque en frío). El recorte pasa de casi negro a visible, con algo de grano en las zonas negras. El contador usa el `timestamp` del video, por eso funciona igual con archivo que con cámara; no se probó con cámara en vivo.

**Salidas por frame** (`imagen.jsonl` e `imagen.csv`, contrato v0.1): `frame_id`, `timestamp`, calidad (contraste, filtro, gamma; PSNR/SSIM solo en evaluación), objetos segmentados (bbox, área, perímetro, centroide) y características (densidad de bordes, esquinas, puntos ORB, media HSV). `video_integrado.mp4` muestra el frame mejorado con las cajas de VIDEO.

## 6. Resultados medidos

Video `padang.webm` (aula, 1080p, 10 s, CC BY-SA 4.0, Wikimedia Commons).

| Prueba | Resultado |
|---|---|
| Producción, 620 frames | 26 FPS de punta a punta; 33 ms por frame en IMAGEN; 0 frames omitidos |
| Evaluación, luz 0.15, 11 frames | Detecciones de VIDEO recuperadas (de 154): degradado 47, solo filtro 51, mejora completa **93**. PSNR 6.8 → 17.6 dB, SSIM 0.15 → 0.67 |
| Evaluación, luz 0.35 | Degradado 94, solo filtro 116, mejora completa 117 |
| Filtros | Gaussiano con mejor SSIM en ambos niveles y casi sin costo (<1 ms); Bilateral unas 30 veces más lento |
| Tiempos por etapa (ms) | gamma 0.7, filtro 0.5, CLAHE 9.3, segmentación 1.0, características 15.2, YOLO una pasada 4.9 |
| Pruebas | 10 en verde |

**Hallazgo en contra:** con `aula_demo.mp4` (baja resolución, comprimido, 2 fps) la calidad subió, pero las detecciones recuperadas bajaron (90 → 76 de 295). La mejora ayuda con video de buena resolución y degradación fuerte. Los ~710 ms por frame que se vieron en la evaluación son el costo del experimento (cuatro filtros, métricas y cuatro pasadas de YOLO), no del modo producción.

## 7. Limitaciones y pendientes

- Ningún video libre probado contiene teléfonos, cuadernos ni audífonos: solo se midieron personas y mochilas.
- La degradación es sintética; falta un video real de examen con mala luz. Muestra pequeña: 11 frames en la evaluación.
- La cámara en vivo está implementada con las funciones de VIDEO pero **no se probó**.
- La alerta de VIDEO está replicada en `alerta_mejorada.py` con sus propias funciones; su bucle (`principal.py`) no se ejecuta ni se modifica. Falta probarla con teléfonos o cuadernos reales.
- Falta entrenar el modelo con las etapas de mejora (carpeta `entrenamiento/`, fuera de este repo): dataset, clases y configuración por revisar, y decidir si se mejora el modelo de comportamientos o el de objetos de VIDEO.
- Sin acordar con AUDIO y PREDICCIÓN el contrato (`timestamp`, frecuencia de muestreo) ni quién clasifica.
- Normas ISO propuestas, pendientes de validar con el docente. Análisis de sesgo y ética (RNF-09) incompleto.

## 8. Referencias

- [Requisitos y matriz de trazabilidad](integrador/docs/REQUISITOS.md)
- [Avance S07](integrador/docs/AVANCE_S07.md)
- Licencia del video de prueba: CC BY-SA 4.0, requiere atribución al citarlo.
