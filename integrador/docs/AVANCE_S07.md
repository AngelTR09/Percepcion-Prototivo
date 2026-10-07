# Avance S07 — Módulo IMAGEN (Percepción Computacional, UPAO 2026-20)

## 1. Problemática y relación con VIDEO
Exámenes presenciales supervisados con cámara. VIDEO detecta con YOLO objetos no permitidos, pero con poca luz o ruido del sensor el detector pierde objetos. IMAGEN mejora la calidad del frame antes de detectar, **mide si la mejora ayuda al detector de VIDEO**, y entrega a PREDICCIÓN calidad, objetos y características con `timestamp`. Detalle en [REQUISITOS.md](REQUISITOS.md); lo que se consume de VIDEO en [INVENTARIO_VIDEO.md](INVENTARIO_VIDEO.md).

## 2. Diagrama de integración
```
video (archivo) ─► adaptador_video ─► degradar* ─► gamma ─► filtro (elegido por métricas) ─► CLAHE
                   (frame_id, ts)                                              │
        detectar_objetos de VIDEO (referencia / degradado / solo filtro / mejorado)
                                    + segmentación + características ─► .jsonl + .csv + evidencias
```
\* La degradación (poca luz + ruido) es sintética: da una referencia limpia para medir. El código de VIDEO no se modificó; solo `imagen/adaptador_video.py` lo importa.

## 3. Técnicas por semana
| Semana | Técnica | Dónde |
|---|---|---|
| S02 | Lectura de frames, BGR↔grises/HSV | `adaptador_video.py`, `preprocesamiento.py` |
| S03 | Filtros Media, Gaussiano, Mediana, Bilateral comparados con PSNR, SSIM y tiempo | `preprocesamiento.comparar_filtros` |
| S04 | Corrección gamma automática, CLAHE, redimensión | `mejoramiento.py` |
| S05 | Otsu + apertura/cierre + contornos (bbox, área, perímetro, centroide) | `segmentacion.py` |
| S06 | Canny, Harris, ORB, media HSV | `caracteristicas.py` |

## 4. Resultados
Video: `padang.webm` (aula, 1080p, 10 s; CC BY-SA 4.0, Wikimedia Commons). 11 frames (1 de cada 60). Detecciones de VIDEO que reaparecen (clase igual, IoU ≥ 0.5) respecto al frame limpio, de 154:

| Luz | Degradado | Solo filtro | Filtro + gamma + CLAHE |
|---|---|---|---|
| 0.35 | 94 | 116 | 117 |
| 0.15 | 47 | 51 | **93** |

Calidad (promedio): a luz 0.15, PSNR 6.79 → 17.61 dB y SSIM 0.149 → 0.666. A luz 0.35, PSNR 9.08 → 18.98 dB y SSIM 0.333 → 0.728.

Filtros (promedio por frame):

| Filtro | PSNR 0.35 | SSIM 0.35 | PSNR 0.15 | SSIM 0.15 | ms/frame |
|---|---|---|---|---|---|
| Media | 16.60 | 0.617 | 15.41 | 0.562 | 0.4 |
| Gaussiano | 16.82 | **0.664** | 15.55 | **0.593** | 0.6 |
| Mediana | 16.60 | 0.616 | 15.15 | 0.545 | 1.1 |
| Bilateral | 16.92 | 0.657 | 15.35 | 0.531 | ~17 |

El Gaussiano obtiene el mejor SSIM en ambos casos y cuesta casi nada; el Bilateral es ~30 veces más lento. Tiempo total ≈ 710 ms por frame, casi todo en las cuatro pasadas de YOLO.

**Hallazgo con un video anterior.** En `aula_demo.mp4` (baja resolución, comprimido, 2 fps) la calidad también subió (PSNR 11.1 → 15.8 dB) pero las detecciones recuperadas **bajaron** (90 → 76 de 295). Con poca resolución de origen, el CLAHE y el gamma amplifican artefactos y el detector no se beneficia. La mejora ayuda con video de buena resolución y degradación fuerte.

Evidencias antes/después: `salidas/padang_luz0.15/evidencias/`. Salidas para PREDICCIÓN: `imagen.jsonl` e `imagen.csv` en cada carpeta de salidas. Pruebas: `pytest`, 7 en verde.

## 5. Limitaciones
- Ningún video libre disponible contiene teléfonos, cuadernos ni audífonos; solo se midieron personas y mochilas.
- La degradación es sintética y la muestra es chica (11 frames, un video).
- El tiempo por frame no permite tiempo real; el pipeline trabaja en diferido.
- Con video de baja calidad la mejora no ayuda al detector (ver hallazgo).
- Pendiente de ética y sesgo (RNF-09): condiciones de falla documentadas aquí, falta el análisis de sesgo del dataset.

## 6. Solicitudes a VIDEO
Ver [SOLICITUDES_A_VIDEO.md](SOLICITUDES_A_VIDEO.md): `frame_id`/`timestamp`/`fps`, lectura de archivo de video, audio en `.wav`, carga diferida del modelo y limpieza de `config.py`.

## 7. Próximos pasos
- Probar con un video real de examen con los objetos de VIDEO.
- Reducir el tiempo por frame (menos pasadas de YOLO, frame más chico).
- RF-12 y RF-13: filtrado en frecuencia y restauración (S09–S10).
- RF-14: CSV listo para Scikit-learn (S11); acordar con PREDICCIÓN el contrato y quién clasifica.
- Opcionales: CNN (S13), tiempo real (S15).
