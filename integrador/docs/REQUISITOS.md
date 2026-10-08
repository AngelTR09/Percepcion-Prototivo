# Requisitos del módulo IMAGEN

> Las normas ISO son propuesta del grupo, pendientes de validar con el docente. Se cita solo nombre y objetivo general.

## 1. Problemática

**Contexto.** Exámenes presenciales supervisados con una cámara. El sistema de VIDEO detecta con YOLO objetos no permitidos (teléfono, audífonos, reloj, cuaderno/libro).

**Problema.** Con poca luz, contraluz o ruido del sensor, el detector pierde objetos o falla en las alertas. Hoy nadie mide ni corrige la calidad del frame antes de detectar.

**Por qué multimodal.** La imagen da qué objeto hay y dónde. El audio (AUDIO) aporta conversación o sonidos del teléfono. PREDICCIÓN las cruza por `timestamp` para decidir si hay una irregularidad.

**Objetivo general del sistema.** Apoyar la supervisión de exámenes combinando video, imagen y audio para estimar la probabilidad de irregularidad.

**Objetivo del módulo IMAGEN.** Mejorar la calidad de cada frame (ruido, contraste, iluminación), segmentar los objetos de interés y extraer características, midiendo cuánto mejora la detección de VIDEO, y entregarlo con `timestamp` a PREDICCIÓN.

**Alcance y limitaciones.** Cámara fija, un estudiante por cuadro, procesamiento diferido sobre video grabado (tiempo real como extensión). Fuera de alcance: la predicción final y el audio.

## 2. RNF

| ID | Característica | Requisito | Medida |
|---|---|---|---|
| RNF-01 | Eficiencia de desempeño (ISO 25010) | Procesar cada frame en tiempo compatible con el muestreo de VIDEO | ms/frame, FPS |
| RNF-02 | Adecuación funcional | La mejora aumenta la detección de VIDEO en frames degradados | Detecciones recuperadas, recall, IoU |
| RNF-03 | Calidad de datos (ISO 25012) | Mejora medible de la calidad del frame | PSNR, SSIM, MSE, contraste |
| RNF-04 | Interoperabilidad | Entrada y salida con contrato versionado | Validación del esquema JSON |
| RNF-05 | Fiabilidad | Un frame corrupto o vacío no detiene el pipeline | Pruebas con frames inválidos |
| RNF-06 | Mantenibilidad | Una etapa = una función; parámetros en `config.py` | Revisión de código |
| RNF-07 | Portabilidad | Funciona en Debian con dependencias fijadas | Instalación limpia en `venv` |
| RNF-08 | Seguridad y privacidad | No se guardan frames con personas salvo evidencia; `salidas/` fuera de git | Revisión de `.gitignore` |
| RNF-09 | Ética y sesgo | Documentar condiciones de falla (luz, escena) y sesgos del dataset | Sección de ética del informe |

## 3. Normas ISO

| Norma | Aporte | RNF |
|---|---|---|
| ISO/IEC 25010 | Modelo de calidad del producto: vocabulario para redactar y medir los RNF | 01, 02, 04–08 |
| ISO/IEC 25012 | Calidad de datos: exactitud, completitud, consistencia | 03 |
| ISO/IEC/IEEE 29148 | Redacción de requisitos verificables y trazables | Todos |
| ISO/IEC 23053 | Marco del pipeline de ML: datos, modelo, evaluación | 02, 03, 06 |
| ISO/IEC 23894 | Gestión de riesgos en IA | 05, 09 |
| ISO/IEC TR 24027 | Sesgo en sistemas de IA | 09 |
| ISO/IEC 42001 | Gestión de IA responsable | 08, 09 |

## 4. RF

Los RF de integración citan lo que se consume de VIDEO.

| ID | Requisito funcional | Semana | RNF | Consume de VIDEO |
|---|---|---|---|---|
| RF-01 | Leer frames y asignarles `frame_id`, `timestamp`, `fps` | S02 | 04 | `cv2.VideoCapture` (solicitud 1 y 2); `camara.obtener_frame` |
| RF-02 | Convertir a grises y HSV según la etapa | S02 | 03 | — |
| RF-03 | Reducir ruido con Media, Gauss, Mediana y Bilateral y elegir con PSNR/SSIM/tiempo | S03 | 01, 03 | — |
| RF-04 | Mejorar contraste e iluminación (ecualización, CLAHE, gamma) | S04 | 03 | — |
| RF-05 | Normalizar tamaño del frame | S04 | 01 | — |
| RF-06 | Segmentar objetos (umbral/HSV + morfología + contornos) | S05 | 02 | — |
| RF-07 | Por objeto: bbox, área, perímetro, centroide | S05 | 02 | `deteccion.detectar_objetos` (bbox de YOLO para delimitar regiones) |
| RF-08 | Extraer bordes (Sobel/Canny), esquinas (Harris), ORB, histograma de color | S06 | 02 | — |
| RF-09 | Comparar detecciones de VIDEO sobre el frame original y el mejorado | S05–S06 | 02 | `deteccion.detectar_objetos` |
| RF-10 | Emitir un registro JSON por frame según el contrato | Integración | 04 | — |
| RF-11 | Guardar evidencias antes/después de cada etapa | Informe | 06 | — |

Extensión (S14): RF-12 filtrado en frecuencia (S09–S10), RF-13 restauración (S10), RF-14 CSV para Scikit-learn (S11), RF-15 clasificación con CNN (S13, opcional), RF-16 tiempo real (S15, opcional).

## 5. Matriz de trazabilidad

| Problemática (elemento) | RNF | ISO | RF |
|---|---|---|---|
| Fallo con poca luz y contraluz | RNF-02, RNF-03 | 25010, 25012 | RF-04, RF-09 |
| Ruido del sensor | RNF-03, RNF-01 | 25012, 25010 | RF-03 |
| Medir si la mejora ayuda al detector | RNF-02 | 25010, 23053 | RF-06, RF-07, RF-08, RF-09 |
| Sincronía con AUDIO y PREDICCIÓN | RNF-04 | 25010, 29148 | RF-01, RF-10 |
| Robustez ante frames malos | RNF-05 | 25010, 23894 | RF-01, RF-10 |
| Reproducibilidad y mantenimiento | RNF-06, RNF-07 | 25010 | RF-11 |
| Privacidad y sesgo | RNF-08, RNF-09 | 42001, 24027, 23894 | RF-11 |
