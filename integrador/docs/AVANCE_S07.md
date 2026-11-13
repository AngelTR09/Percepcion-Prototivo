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
| S04 | Rotación, escalamiento, traslación y volteo con `warpAffine`; robustez del detector de VIDEO | `transformaciones.py`, `ejecutar_geometricas.py` |
| S05 | Otsu + apertura/cierre + contornos (bbox, área, perímetro, centroide) | `segmentacion.py` |
| S05 | Umbral adaptativo (`adaptiveThreshold`) y segmentación por regiones (watershed), comparados con Otsu por IoU de máscaras | `segmentacion.py`, `ejecutar_segmentacion.py` |
| S06 | Canny, Harris, ORB, media HSV | `caracteristicas.py` |
| S06 | Sobel comparado con Canny; ORB con descriptores y `BFMatcher` + homografía RANSAC; `matchTemplate` como alternativa | `caracteristicas.py`, `ejecutar_patrones.py` |
| S15 | FPS por etapa y por configuración; análisis de sesgo de desempeño por clase, tamaño y posición; ética y uso responsable | `rendimiento.py`, `sesgo.py`, `ejecutar_rendimiento.py`, `docs/ANALISIS_SESGO_ETICA.md` |
| S12 | MLP con Keras (Flatten + Dense relu + softmax), parada temprana, matriz de confusión, precisión/recall/F1, línea base mayoritaria | `mlp.py`, `ejecutar_mlp.py`, `ejecutar_recortes.py` |
| S13 | CNN (Conv + MaxPool) con aumento geométrico; transfer learning con MobileNetV2 (cabeza y afinado); robustez a rotaciones de S04 | `cnn.py`, `ejecutar_cnn.py` |

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
- Tiempo real: el modo producción alcanza 8.8 FPS a 960 px y 15.6 FPS a 640 px (sección 8); el modo de evaluación (4 pasadas de YOLO con métricas) sigue siendo diferido.
- Con video de baja calidad la mejora no ayuda al detector (ver hallazgo).
- Ética y sesgo (RNF-09): se midió el desempeño por clase, tamaño y posición (sección 8 y [ANALISIS_SESGO_ETICA.md](ANALISIS_SESGO_ETICA.md)). El sesgo demográfico y el balance del dataset de entrenamiento no se pueden auditar con los datos disponibles.

## 6. Solicitudes a VIDEO
Ver [SOLICITUDES_A_VIDEO.md](SOLICITUDES_A_VIDEO.md): `frame_id`/`timestamp`/`fps`, lectura de archivo de video, audio en `.wav`, carga diferida del modelo y limpieza de `config.py`.

## 7. Próximos pasos
- Probar con un video real de examen con los objetos de VIDEO.
- Reducir el tiempo por frame (menos pasadas de YOLO, frame más chico).
- RF-12 y RF-13: filtrado en frecuencia y restauración (S09–S10).
- RF-14: CSV listo para Scikit-learn (S11); acordar con PREDICCIÓN el contrato y quién clasifica.
- Opcionales: CNN (S13), tiempo real (S15).

## 5. S04 — Transformaciones geométricas como aumento de datos
Se transforma cada frame (rotación, escalamiento, traslación, volteo) y se transforman también las cajas con la misma matriz afín. Se cuenta cuántas detecciones de VIDEO del frame original reaparecen (clase igual, IoU ≥ 0.5) en el frame transformado. Solo cuentan como referencia las cajas que siguen al menos 50 % dentro del encuadre. Mismos 11 frames de `padang.webm` que en la sección 4.

`python -m imagen.ejecutar_geometricas --video datos/padang.webm --salida salidas/geometricas --cada-n 60`

| Transformación | Referencia | Recuperadas | Tasa |
|---|---|---|---|
| Rotar +10° | 154 | 111 | 0.72 |
| Rotar −10° | 154 | 88 | 0.57 |
| Rotar +25° | 154 | 46 | 0.30 |
| Escalar 0.7 | 154 | 128 | 0.83 |
| Escalar 1.3 | 120 | 102 | 0.85 |
| Trasladar 10 % / 5 % | 134 | 118 | 0.88 |
| Voltear horizontal | 154 | 128 | 0.83 |

Lectura: el detector de VIDEO es sensible a la rotación (cae a 0.30 con 25°) y razonablemente estable ante escala y traslación. Ni el volteo, que no cambia el contenido, llega a 1.0: parte de las detecciones son marginales (confianza baja) y aparecen o desaparecen con cualquier cambio de píxeles, así que la tasa mide la estabilidad del detector y no solo la geometría. Las cajas transformadas se verifican en `tests/test_transformaciones.py` con un detector de prueba que sí es exacto (IoU > 0.7 en todas las transformaciones).
Pendiente (no hecho): reentrenar o afinar VIDEO con estos aumentos; esto solo mide.

## 6. S05 — Umbral adaptativo y segmentación por regiones (watershed)
`python -m imagen.ejecutar_segmentacion --video datos/padang.webm --salida salidas/segmentacion --cada-n 60` (mismos 11 frames; VIDEO detecta 14.0 objetos por frame).

| Método | Objetos/frame | IoU vs Otsu | IoU vs cajas de VIDEO | ms/frame |
|---|---|---|---|---|
| Otsu + morfología + contornos | 7.5 | 1.000 | 0.217 | 3.9 |
| Umbral adaptativo (bloque 101, c = 3) | 17.4 | 0.601 | **0.256** | 9.7 |
| Watershed (semillas al 60 % de la distancia) | 14.5 | 0.512 | 0.153 | 153.0 |

Cómo leerlo:
- **La referencia es débil.** Una caja de VIDEO incluye fondo y las máscaras separan por intensidad, no por objeto; por eso todos los IoU son bajos. Sirve para comparar métodos entre sí, no como exactitud absoluta.
- **Umbral adaptativo.** Mejora un poco a Otsu y es el único que funciona con iluminación despareja: en una escena sintética con verdad conocida (fondo que sube de 30 a 200) Otsu obtiene IoU 0.09 y el adaptativo 0.998 (`tests/test_segmentacion_regiones.py`). Con un bloque chico fragmenta (23.8 objetos por frame con bloque 51); la ventana debe ser bastante mayor que el objeto.
- **Watershed.** Es el único que separa objetos que se tocan (los contornos de una máscara binaria los funden en uno). Su número de objetos (14.5) se acerca al de VIDEO (14.0), pero su máscara cubre menos y es 40 veces más lento que Otsu: no se usó en el pipeline.
- **Defecto corregido en el proceso.** La primera versión calculaba las semillas con un umbral global y el IoU contra VIDEO cayó a 0.06; con el umbral por componente subió a 0.15. Los parámetros (bloque 101, c = 3, fracción 0.6) se eligieron barriendo `padang.webm`, no la imagen de prueba.
- Con un solo video de aula y 11 frames, estos números no se generalizan a otras escenas.
Pendiente: comparación sobre más videos, y decidir si watershed entra al pipeline (hoy no: `pipeline.py` sigue usando Otsu).

## 7. S06 — Sobel, ORB completo y reconocimiento de patrones
`python -m imagen.ejecutar_patrones --video datos/padang.webm --salida salidas/patrones --cada-n 60` (mismos 11 frames). `caracteristicas.extraer` no cambió: sigue siendo el contrato de columnas con PREDICCIÓN.

**Sobel vs Canny** (promedio por frame, tolerancia de 1 px):

| | Densidad de bordes | ms | Cubierto por el otro |
|---|---|---|---|
| Canny | 0.118 | 2.2 | 76 % de sus píxeles coinciden con un borde de Sobel |
| Sobel (umbral 20 % de la magnitud máxima) | 0.130 | 2.6 | 99.7 % de sus píxeles coinciden con un borde de Canny |

Sobel solo marca los gradientes fuertes y Canny los confirma casi todos; Canny además recupera bordes finos que Sobel deja bajo el umbral. Su costo es parecido.

**Reconocimiento de patrones.** Se recorta la detección de VIDEO más segura de cada frame (11 recortes) y se busca en el mismo frame transformado con las transformaciones de S04. Acierto = IoU ≥ 0.5 con la caja transformada.

| Transformación | ORB + BFMatcher + RANSAC | matchTemplate |
|---|---|---|
| Rotar +10° / −10° / +25° | 10/11 en las tres | 0/11 en las tres |
| Escalar 0.7 | 10/11 | 0/11 |
| Escalar 1.3 | 8/11 | 2/11 |
| Trasladar 10 % | 10/11 | 11/11 |
| Voltear horizontal | 0/11 | 0/11 |

Tiempo por búsqueda: ORB 15–30 ms, matchTemplate 10 ms.

Cómo leerlo:
- `matchTemplate` solo sirve cuando el patrón no cambia de orientación ni de tamaño (traslación). ORB tolera rotación y escala, a cambio de algo más de tiempo.
- Ninguno reconoce el patrón volteado: los descriptores ORB no son invariantes al espejo.
- **Limitaciones medidas en las pruebas sintéticas.** (1) ORB ignora los puntos a menos de 31 px del borde; un recorte de 120 px quedaba con 4 puntos y no se podía localizar nada. Con `borde = 15` queda con ~50. (2) RANSAC aceptaba homografías degeneradas (una caja de 19×58 px con 10 inliers); se descartan si el cuadrilátero no es convexo o su área está fuera de 0.25–4× la de la plantilla. (3) Reducir la imagen borra el detalle fino: en la escena sintética ORB localiza con ampliación y reducción 0.9×, pero no con 0.8×; en los frames reales sí llega a 0.7× porque tienen más textura.
- Reconocimiento básico de patrones aquí significa "encontrar un recorte conocido"; no clasifica objetos. Falta usarlo dentro del pipeline (hoy no entra: `pipeline.py` no cambió).
- 11 recortes de un solo video: sirven de comparación, no para generalizar.

## 8. S15 — Tiempo real y sesgo por condición
**FPS reales** (`python -m imagen.ejecutar_rendimiento --video datos/padang.webm --salida salidas/rendimiento --cada-n 30`, 21 frames, CPU de la PC de desarrollo con Windows, sin GPU). Modo producción: una sola pasada de YOLO por frame. Tiempo real = al menos 15 FPS (`config.FPS_TIEMPO_REAL`).

| Configuración | ms/frame | FPS | Tiempo real | Detecciones de 960 px conservadas |
|---|---|---|---|---|
| Completo, 960 px | 113.7 | 8.8 | no | 246/292 (84 %) |
| Completo, 640 px | 64.2 | **15.6** | sí | 214/292 (73 %) |
| Solo mejora + detección, 640 px | 44.0 | 22.8 | sí | 214/292 (73 %) |
| Solo mejora + detección, 480 px | 39.9 | 25.1 | sí | 174/292 (60 %) |

Dónde se va el tiempo a 960 px: características 51 ms (ORB y Harris), detector de VIDEO 33 ms, CLAHE 21 ms; el resto suma menos de 9 ms. Hallazgos:
- **El detector cuesta ~33 ms con cualquier ancho** (YOLO reescala a 640 internamente). Achicar el frame no lo acelera; ahorra en CLAHE y en características.
- La columna de detecciones conservadas se compara contra el frame original a 960 px; ya a 960 px la mejora cambia algunas cajas (84 %), así que achicar a 640 cuesta unos 11 puntos más y a 480 unos 24.
- La cifra anterior de ~710 ms/frame correspondía al modo de evaluación (4 pasadas de YOLO, referencia limpia y métricas), no al modo producción.
- En Render o con otra CPU los FPS serán distintos; hay que volver a medir ahí. Con la cadena integrada (`/api/integrado/analizar`) hay dos pasadas de YOLO por imagen.
- MobileNet-SSD (alternativa opcional del sílabo) no se probó; YOLO alcanza.

**Sesgo por condición** (`... --cada-n 15 --sesgo --luz 0.15`, 42 frames, luz al 15 %). Detecciones del frame limpio que reaparecen sin y con la mejora:

| Criterio | Brecha sin mejora | Brecha con mejora | Detalle |
|---|---|---|---|
| Clase | 0.24 | **0.47** | persona 0.36 → 0.64, mochila 0.12 → 0.16 |
| Tamaño | 0.53 | 0.44 | grande 0.70 → 0.90, pequeño 0.17 → 0.46 |
| Posición | 0.22 | 0.18 | centro 0.46 → 0.68, borde 0.24 → 0.50 |

Hallazgo principal: la mejora ayuda mucho a las personas y casi nada a las mochilas, así que la brecha entre clases empeora. Análisis completo, límites (referencia = detecciones de VIDEO, un solo video, sin etiquetas demográficas) y uso responsable en [ANALISIS_SESGO_ETICA.md](ANALISIS_SESGO_ETICA.md).

## 9. S12 — Perceptrón multicapa con Keras
Entorno aparte (`requirements-lab.txt`: TensorFlow 2.21.0, Keras 3.15.1); no entra al Dockerfile ni a Render, y las pruebas de Keras se omiten si no está instalado. Arquitectura: `Flatten → Dense(128, relu) → Dense(64, relu) → Dense(n, softmax)`, Adam (tasa 1e-3), lote 64, 20 % de validación y parada temprana con paciencia 4 (parámetros en `config.MLP`, semilla fija).

**Experimento A: Fashion-MNIST reducido** (10 000 de entrenamiento, 2 000 de prueba; `python -m imagen.ejecutar_mlp --salida salidas/mlp`). Exactitud en prueba **0.851** (azar: 0.10); paró en la época 9; 4 s de entrenamiento. Por clase (F1): pantalón 0.96, bolso 0.96, sandalia 0.94, botín 0.94, zapatilla 0.93, camiseta 0.85, vestido 0.83, abrigo 0.73, jersey 0.73, **camisa 0.62**. Se confunden las prendas de torso entre sí; un MLP ve cada píxel por separado y no capta la forma local, que es lo que la CNN de S13 debería mejorar.

**Experimento B: recortes de VIDEO** (2 902 recortes de 32×32 de `padang.webm`: 2 573 personas y 329 mochilas, 1 de cada 3 frames; `ejecutar_recortes` y luego `ejecutar_mlp --recortes`). División por tiempo: los últimos 30 % de los frames para prueba (851 recortes: 783 y 68), para que frames casi idénticos no caigan a ambos lados.

| Modelo | Exactitud | Recall mochila | Precisión mochila | F1 mochila |
|---|---|---|---|---|
| Línea base (siempre "persona") | 0.920 | 0.00 | — | — |
| MLP sin pesos | **0.995** | 0.941 | 1.000 | 0.970 |
| MLP con pesos de clase | 0.979 | **0.971** | 0.805 | 0.880 |

- Compensar el desbalance (pesos inversos a la frecuencia) sube el recall de mochila (64 → 66 de 68) pero baja la precisión a 0.805: aparecen falsas mochilas. Es un compromiso, no una mejora; qué conviene depende del costo de una falsa alerta.
- La exactitud sola engaña: una línea base que siempre dice "persona" ya acierta 0.92.
- **Límites.** (1) Las etiquetas son las clases que dio VIDEO, no anotaciones manuales: el MLP aprende a imitar a VIDEO. (2) Persona y mochila se distinguen fácil por forma y tamaño. (3) Un solo video: las mismas personas y la misma escena están en entrenamiento y prueba, así que 0.995 no se generalizará a otra aula. (4) Solo 68 mochilas en prueba: un error cambia el recall en 1.5 puntos.
Pendiente: S13 (CNN y transfer learning) sobre estos mismos recortes y Fashion-MNIST, y S14 (hiperparámetros y regularización con tabla comparativa).

## 10. S13 — CNN y transfer learning
Mismo entorno de laboratorio que S12. CNN: `[aumento] → (Conv3×3 relu → MaxPool 2×2) × 2 (32 y 64 filtros) → Flatten → Dense(64) → Dense(n, softmax)`. El aumento (rotación ±0.07 vuelta ≈ ±25°, traslación y zoom 10 %) equivale a las transformaciones de S04 y solo actúa al entrenar. Transfer learning: MobileNetV2 con pesos de ImageNet, congelada, más una cabeza nueva (pooling global, dropout 0.2, Dense); después se afinan sus últimas 30 capas con tasa 1e-5 (BatchNorm sigue congelada). Parámetros en `config.CNN` y `config.TRANSFER`.

`python -m imagen.ejecutar_cnn --salida salidas/cnn --fashion` · `--recortes32 salidas/recortes/recortes.npz --semillas 5` · `--recortes96 salidas/recortes96/recortes.npz` (los recortes de 96 px se generan con `ejecutar_recortes --lado 96`).

**Fashion-MNIST reducido (MLP de S12 contra CNN, una semilla cada uno):**

| Modelo | Parámetros | Exactitud | F1 camisa | F1 jersey | F1 abrigo |
|---|---|---|---|---|---|
| MLP | 109 386 | 0.851 | 0.62 | 0.73 | 0.73 |
| CNN | 220 234 | **0.876** | 0.69 | 0.79 | 0.76 |

La CNN gana 2.5 puntos y mejora justo las prendas de torso que el MLP confundía (la convolución capta la forma local), al doble de parámetros. Con una semilla, el margen no está medido con su variabilidad.

**Recortes de VIDEO, 5 semillas (media ± desviación; 851 recortes de prueba, 68 mochilas; recall = el de mochila).** La prueba también se rota con la transformación de S04.

| Modelo | Exactitud 0° | Recall 0° | Exactitud 25° | Recall 25° |
|---|---|---|---|---|
| MLP | 0.992 ± 0.004 | 0.932 ± 0.007 | 0.920 ± 0.000 | **0.000** |
| CNN sin aumento | 0.983 ± 0.019 | 0.941 ± 0.013 | 0.941 ± 0.028 | 0.268 ± 0.347 |
| CNN con aumento | **0.997 ± 0.003** | **0.982 ± 0.014** | 0.981 ± 0.030 | 0.759 ± 0.372 |

**Transfer learning (96 px, una semilla):** cabeza entrenada: exactitud 0.999 con solo **2 562 parámetros entrenables** (2.26 M congelados), recall de mochila 0.985, y a 25° exactitud 0.988 y recall 0.912. Afinando las últimas capas (1.5 M entrenables) no mejora en prueba (0.999) y a 25° baja (0.979, recall 0.853). Entrena en 43 s contra 2–4 s de la CNN pequeña.

Lectura:
- **El MLP no sirve fuera de la orientación de entrenamiento:** acierta 0.99 sin rotar y con 10° ya no detecta mochilas (recall 0.012). La convolución más el aumento es lo que da robustez; el aumento mejora recall de mochila de 0.941 a 0.982 sin rotar y de 0.268 a 0.759 a 25°.
- **La varianza entre semillas es enorme a 25°** (± 0.35 en recall). Una sola ejecución puede dar 0.15 o 0.96: la primera corrida de la CNN sin aumento dio 0.946 y la de con aumento 0.9965 a 25°, y ambas eran extremos de su distribución. Sin repetir con semillas, esta comparación habría sido ruido.
- **Transfer learning es la opción más robusta aquí** y con casi nada que entrenar, pero se midió con una sola semilla y sin probar rotación en entrenamiento; no se debe leer como "mejor que el aumento".
- **Límites** (los mismos de S12): etiquetas de VIDEO y no manuales, una sola escena con las mismas personas en entrenamiento y prueba, 68 mochilas en prueba (un error = 1.5 puntos), y MobileNetV2 ya vio ImageNet, que contiene personas y mochilas.
Pendiente: S14 (hiperparámetros, regularización y tabla comparativa con intervalos).
