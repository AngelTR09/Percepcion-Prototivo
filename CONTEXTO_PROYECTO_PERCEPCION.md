# CONTEXTO — Proyecto integrador de Percepción Computacional (Grupo IMAGEN)

> **Para Claude Code.** En este repositorio ya está el proyecto del grupo **VIDEO**. Tu tarea es acoplarle el módulo del grupo **IMAGEN** (el nuestro), de modo que todo funcione como un solo proyecto junto con los grupos AUDIO y PREDICCIÓN.
> Todo (código, comentarios, documentación y respuestas) va en **español**. Los comandos de terminal son para **Linux Debian (bash)** y deben poder copiarse y pegarse.

---

## 0. Resumen en 30 segundos

- **Curso:** Percepción Computacional (ISIA 111), UPAO, semestre 2026-20, docente Carlos Alfredo Mendoza Corpus.
- **Sistema único hecho por 4 grupos:** VIDEO → (IMAGEN ‖ AUDIO) → PREDICCIÓN.
- **Nuestro módulo (IMAGEN):** recibe frames del módulo VIDEO, los limpia, mejora, segmenta y extrae características. Entrega un resultado estructurado y sincronizado por tiempo a PREDICCIÓN.
- **Metodología obligatoria:** Problemática → RNF → Normas técnicas ISO → RF.
- **Regla de oro:** el código del grupo VIDEO es de **solo lectura**. No se modifica, renombra ni reformatea ningún archivo suyo. Nosotros **consumimos lo que VIDEO entrega** (funciones, clases, archivos de salida) a través de un **adaptador propio**, y construimos nuestro módulo encima.
- **Nuestra problemática puede ser distinta a la de VIDEO.** No estamos obligados a continuar su caso de uso. Lo que sí debemos respetar es lo que VIDEO puede entregarnos (frames, metadatos, audio) para que la cadena VIDEO → IMAGEN → PREDICCIÓN funcione.
- **Fecha urgente:** la **semana 7 (12–16 oct 2026)** es la defensa del **avance del proyecto**, y el sílabo pide un *mini proyecto integrador: detección y segmentación de objetos reales con las técnicas de las semanas 1 a 6*. Prioriza lo que entra en ese avance.

---

## 1. Qué dice el sílabo (fuente oficial)

### 1.1 Datos generales
| Campo | Valor |
|---|---|
| Programa | Ing. de Sistemas e Inteligencia Artificial, Facultad de Ingeniería, UPAO |
| Asignatura | Percepción Computacional, ISIA 111, ciclo 07, 3 créditos, presencial |
| Duración | 16 semanas, del 31/08/2026 al 23/12/2026 (4 h/semana) |
| Prerrequisito | Inteligencia Artificial: Principios y Técnicas |
| Competencia | **CE3:** despliega soluciones de IA **con estándares éticos** |
| Criterio | CE3_CD1: aplica técnicas de IA para desarrollar y complementar sistemas |
| RA Unidad 1 | RA1.1: aplica técnicas de procesamiento y limpieza de datos y señales con herramientas actuales |
| RA Unidad 2 | RA2.1: analiza la información obtenida con técnicas avanzadas de validación |
| Herramientas | Python, NumPy, SciPy, OpenCV (U1); TensorFlow, Keras, Scikit-learn (U2) |

### 1.2 Contenido por semana y para qué grupo sirve cada tema

| Sem. | Tema | Laboratorio (lo que se practica) | Grupo que más lo usa |
|---|---|---|---|
| S01 | Introducción, flujo de procesamiento de datos visuales; **formación de grupos para el producto final** | Ensayo/control de lectura | Todos |
| S02 | Representación digital: muestreo, cuantificación, espacios de color, señales analógicas/digitales | Imágenes como matrices; RGB↔HSV↔grises; señales con NumPy | **IMAGEN**, AUDIO |
| S03 | Preprocesamiento y limpieza: calidad de datos, eliminación de ruido, suavizado, normalización | Filtros Media, Gaussiano, Mediana, Bilateral + comparación con **métricas visuales y cuantitativas** | **IMAGEN**, AUDIO |
| S04 | Transformaciones y mejoramiento | Rotación, escalado, traslación; ecualización de histograma; mejora de contraste en imágenes de baja calidad; corrección de iluminación | **IMAGEN** |
| S05 | Segmentación | Umbralización, regiones, contornos, operaciones morfológicas para **identificar objetos de interés** | **IMAGEN** |
| S06 | Extracción de características y patrones | Sobel, Canny, Harris, ORB/SIFT; reconocimiento básico de patrones | **IMAGEN** |
| **S07** | **Reforzamiento** | **Defensa del avance del proyecto + mini proyecto integrador: detección y segmentación de objetos reales (S01–S06)** | **IMAGEN (entregable clave)** |
| S08 | Examen parcial | — | — |
| S09 | Análisis espectral, Transformada de Fourier | FFT de señales y patrones periódicos | AUDIO, IMAGEN (opcional) |
| S10 | Filtrado digital y restauración de imágenes | Pasa-bajos / pasa-altos con SciPy y OpenCV, análisis comparativo de rendimiento | AUDIO, IMAGEN |
| S11 | ML para percepción | Clasificador supervisado con Scikit-learn **sobre patrones visuales previamente extraídos** | PREDICCIÓN (usa nuestras características) |
| S12 | Redes neuronales | MLP con TensorFlow/Keras | PREDICCIÓN |
| S13 | CNN y transfer learning | CNN sobre MNIST / Fashion-MNIST / CIFAR-10 | PREDICCIÓN / IMAGEN |
| **S14** | **Reforzamiento** | **Defensa del proyecto final** + optimización de hiperparámetros y regularización | Todos |
| S15 | Sistemas inteligentes | Detección con modelos preentrenados (YOLO, MobileNet-SSD); **visión en tiempo real**; **ética, sesgos y responsabilidad en IA** | IMAGEN, VIDEO |
| S16 | Examen final | — | — |

> **Nota:** S15 llega *después* de la defensa final (S14). Si queremos usar YOLO/MobileNet-SSD en el proyecto final, hay que adelantarlo por cuenta propia. Si no, el proyecto debe poder defenderse solo con técnicas clásicas (S02–S06, S09–S10).

### 1.3 Evaluación (lo que pesa el proyecto)
- **EP1 (20%)** = 40% prácticas de laboratorio + 20% control + **40% presentación y exposición del avance del proyecto (S07)**
- **EP2 (20%)** = 40% prácticas + 20% control + **40% presentación, exposición y defensa del proyecto final (S14)**
- **Examen parcial (EP) 30%**, **examen final (EF) 30%**
- **PROMO = 0.20·EP1 + 0.20·EP2 + 0.30·EP + 0.30·EF** (escala vigesimal)
- Los laboratorios semanales son **individuales** (informe .PDF + código .PY). El **proyecto es grupal**. Las técnicas de ABP (aprendizaje basado en proyectos) se evalúan con rúbrica analítica, lista de cotejo y rúbrica de exposición.

### 1.4 Lo que el sílabo NO dice (viene de indicaciones del docente/grupo)
- La división en grupos **video / imagen / audio / predicción** y la secuencia del pipeline.
- La metodología **Problemática → RNF → ISO → RF**.
- La **problemática concreta** del proyecto.

No inventes nada de esto. Lo que no esté en el código del grupo VIDEO, **pregúntalo**.

---

## 2. Arquitectura del sistema integrado

```
                     frames + metadatos        características visuales
                  ┌───────────────► IMAGEN ───────────────┐
  fuente ─► VIDEO ┤                                       ├─► PREDICCIÓN ─► resultado
 (cámara/  (decodifica,                                    │   (fusiona y predice)
  archivo)  muestrea)  pista de audio + metadatos          │
                  └───────────────► AUDIO  ───────────────┘
                                         características sonoras
```

**Responsabilidades (límites claros para no pisarnos):**

| Módulo | Hace | NO hace |
|---|---|---|
| VIDEO | Lee la fuente, decodifica, muestrea frames (cada N frames o a X FPS), separa el audio, marca `timestamp` | Analizar el contenido de los frames |
| **IMAGEN** | Preprocesa, mejora, segmenta, detecta objetos y extrae características **por frame** | La predicción final del sistema |
| AUDIO | Limpia y filtra la señal, FFT/espectro, extrae características sonoras **por ventana de tiempo** | Analizar imágenes |
| PREDICCIÓN | Sincroniza imagen y audio por `timestamp`, entrena y evalúa el modelo (Scikit-learn / MLP / CNN) y da la salida final | Preprocesar frames o audio crudo |

> **Punto a confirmar con el grupo PREDICCIÓN:** si la clasificación de objetos (por ejemplo, con CNN) la hacemos nosotros dentro de IMAGEN o la hacen ellos. Por defecto, IMAGEN entrega **detecciones + vector de características** y PREDICCIÓN decide.

---

## 3. Metodología: Problemática → RNF → ISO → RF

Cada eslabón se **deriva** del anterior, y al final debe existir una **matriz de trazabilidad** (cada RF enlazado a su RNF y este a su norma ISO).

### 3.1 Problemática — ⚠️ PENDIENTE

La problemática del grupo IMAGEN **no tiene que ser la misma que la del grupo VIDEO**. Se define a partir de **lo que VIDEO nos puede entregar** (el inventario del Paso 2, sección 5) y de las técnicas de imagen del curso (S02–S06, S09–S10).

**Claude Code:**
1. Identifica la problemática que plantea VIDEO (si existe) y resúmela.
2. Evalúa si nuestro módulo puede ser **continuación directa** de esa problemática (**Modo A**) o si conviene una **problemática propia** que solo reutilice lo que VIDEO entrega (**Modo B**). Ver Paso 3 de la sección 5.
3. Propón **2 o 3 problemáticas candidatas** para IMAGEN, cada una con: qué usa de VIDEO, qué técnicas del curso aplica, qué entregaría a PREDICCIÓN y si es viable para el avance de S07.
4. **Espera mi elección** antes de seguir.

Estructura de la problemática elegida:

- **Contexto:** dónde ocurre y a quién afecta.
- **Problema:** qué se hace hoy de forma manual, lenta o poco fiable.
- **Por qué hace falta percepción multimodal:** qué aporta la imagen y qué aporta el audio a la predicción.
- **Objetivo general del sistema.**
- **Objetivo específico del módulo IMAGEN.**
- **Alcance y limitaciones:** tipo de video, iluminación, cámara fija o móvil, tiempo real o diferido.

### 3.2 RNF — derivados de la problemática (borrador)

| ID | Característica (ISO/IEC 25010) | Requisito no funcional del módulo IMAGEN | Cómo se mide |
|---|---|---|---|
| RNF-01 | Eficiencia de desempeño | Procesar cada frame en un tiempo compatible con la tasa de muestreo que entrega VIDEO | ms/frame, FPS efectivos |
| RNF-02 | Adecuación funcional (exactitud) | La segmentación/detección alcanza un umbral mínimo acordado | IoU, precisión, recall, F1 |
| RNF-03 | Calidad de datos (ISO/IEC 25012) | Mejorar de forma medible la calidad del frame antes del análisis | PSNR, SSIM, MSE, contraste, nivel de ruido (S03) |
| RNF-04 | Compatibilidad / interoperabilidad | Entrada y salida siguen un contrato fijo y versionado con VIDEO y PREDICCIÓN | Validación del esquema JSON |
| RNF-05 | Fiabilidad (tolerancia a fallos) | Un frame corrupto o vacío no detiene el pipeline: se registra y se omite | Pruebas con frames inválidos |
| RNF-06 | Mantenibilidad (modularidad) | Una etapa = una función; los parámetros van en un archivo de configuración | Revisión de código |
| RNF-07 | Portabilidad | Funciona en Debian con dependencias fijadas en `requirements.txt` | Instalación limpia en un `venv` |
| RNF-08 | Seguridad / privacidad | No se guardan frames con personas identificables salvo que sea necesario; las salidas de depuración van en una carpeta ignorada por git | Revisión de `salidas/` y `.gitignore` |
| RNF-09 | Ética y sesgo (CE3, S15) | Documentar condiciones donde el módulo falla (iluminación, tipo de escena) y posibles sesgos del dataset | Sección de ética en el informe |

### 3.3 Normas técnicas ISO que respaldan los RNF (propuesta)

| Norma | Qué aporta al proyecto | RNF |
|---|---|---|
| **ISO/IEC 25010** (modelo de calidad del producto, SQuaRE) | Vocabulario y características de calidad para redactar y medir los RNF | 01, 02, 04, 05, 06, 07, 08 |
| **ISO/IEC 25012** (calidad de datos) | Exactitud, completitud y consistencia de los datos de entrada y salida | 03 |
| **ISO/IEC/IEEE 29148** (ingeniería de requisitos) | Cómo redactar RF/RNF verificables, sin ambigüedad y trazables | Todos |
| **ISO/IEC 23053** (marco para sistemas de IA basados en ML) | Estructura del pipeline datos → modelo → evaluación | 02, 03, 06 |
| **ISO/IEC 23894** (gestión de riesgos en IA) | Identificar riesgos del módulo (fallos con poca luz, falsos positivos) | 05, 09 |
| **ISO/IEC TR 24027** (sesgo en sistemas de IA) | Analizar sesgos del dataset y del modelo | 09 |
| **ISO/IEC 42001** (sistema de gestión de IA) | Marco de uso responsable, ligado a la competencia CE3 | 08, 09 |

> ⚠️ Estas normas son **propuesta nuestra**, no del sílabo. Confírmalas con el docente. Claude Code: cita solo nombre y objetivo general de cada norma; **no inventes números de cláusula ni texto literal**.

### 3.4 RF del módulo IMAGEN (cada uno ligado a un tema del curso)

**Núcleo — entra en el avance de S07:**

| ID | Requisito funcional | Técnica / semana | RNF |
|---|---|---|---|
| RF-01 | Recibir frames de VIDEO (`ndarray` BGR `uint8`) con su `frame_id` y `timestamp` | S02 | 04 |
| RF-02 | Convertir el frame a escala de grises y HSV según la etapa | S02 | 03 |
| RF-03 | Reducir el ruido con Media, Gaussiano, Mediana y Bilateral, y **elegir el filtro con métricas** (PSNR/SSIM/tiempo) | S03 | 01, 03 |
| RF-04 | Mejorar el contraste e iluminación (ecualización de histograma / CLAHE, corrección gamma) | S04 | 03 |
| RF-05 | Normalizar el tamaño/orientación del frame (redimensión; rotación si la cámara lo exige) | S04 | 01 |
| RF-06 | Segmentar objetos de interés (Otsu/umbral adaptativo o HSV + morfología apertura/cierre + contornos) | S05 | 02 |
| RF-07 | Obtener por cada objeto: bounding box, área, perímetro, centroide | S05 | 02 |
| RF-08 | Extraer características: bordes (Sobel/Canny), esquinas (Harris), descriptores ORB, histograma de color | S06 | 02 |
| RF-09 | Emitir por frame un registro JSON según el contrato de la sección 4 | Integración | 04 |
| RF-10 | Guardar evidencias visuales (antes/después de cada etapa) para el informe y la exposición | Informe | 06 |

**Extensión — para el proyecto final de S14:**

| ID | Requisito funcional | Técnica / semana |
|---|---|---|
| RF-11 | Filtrado en el dominio frecuencial (FFT, pasa-bajos/pasa-altos) y comparación con el filtrado espacial | S09–S10 |
| RF-12 | Restauración de frames degradados (desenfoque, ruido periódico) | S10 |
| RF-13 | Exportar un dataset de características (CSV) para que PREDICCIÓN entrene en Scikit-learn | S11 |
| RF-14 | (Opcional) Clasificar recortes con una CNN / transfer learning | S13 |
| RF-15 | (Opcional) Detección con modelo preentrenado (YOLO / MobileNet-SSD) en tiempo real | S15 (adelantado) |

---

## 4. Contrato de integración (propuesta v0.1, acordar con los otros grupos)

**Entrada (VIDEO → IMAGEN), por frame:**
```python
{
    "frame_id": int,          # índice del frame en el video original
    "timestamp": float,       # segundos desde el inicio (clave de sincronización con AUDIO)
    "fps": float,             # FPS del video original
    "fuente": str,            # nombre del archivo o "camara"
    "frame": numpy.ndarray    # H x W x 3, uint8, BGR (formato OpenCV)
}
```

**Salida (IMAGEN → PREDICCIÓN), por frame:**
```json
{
  "version_contrato": "0.1",
  "frame_id": 120,
  "timestamp": 4.0,
  "calidad": {"psnr": 0.0, "ssim": 0.0, "contraste": 0.0, "filtro_usado": "bilateral"},
  "objetos": [
    {"id": 0, "bbox": [x, y, w, h], "area": 0.0, "perimetro": 0.0, "centroide": [cx, cy]}
  ],
  "caracteristicas": {
    "n_objetos": 0,
    "densidad_bordes": 0.0,
    "n_esquinas": 0,
    "n_keypoints_orb": 0,
    "hist_hsv": [0.0, 0.0, 0.0]
  },
  "tiempo_ms": 0.0
}
```
- Formato de intercambio: un archivo **JSON Lines** (`.jsonl`, una línea por frame) y/o un **CSV** de características para PREDICCIÓN.
- `timestamp` es obligatorio: con él PREDICCIÓN alinea los frames con las ventanas de audio.
- **El formato de VIDEO no se cambia.** Si VIDEO entrega otra cosa (imágenes en disco, otro orden de canales, sin `timestamp`, otra estructura), el **adaptador** (`imagen/adaptador_video.py`) lo convierte a este contrato de entrada. Si a VIDEO le falta un dato (por ejemplo `timestamp`), el adaptador lo calcula (`frame_id / fps`) y se registra como solicitud en `docs/SOLICITUDES_A_VIDEO.md`.

```
 código VIDEO (intacto)  ──►  imagen/adaptador_video.py  ──►  imagen/pipeline.py  ──►  salida para PREDICCIÓN
  (solo se importa o se      (traduce lo que VIDEO        (nuestro procesamiento)
   leen sus salidas)          entrega a nuestro contrato)
```

---

## 5. Instrucciones de trabajo para Claude Code

Haz los pasos **en orden** y **detente al final de cada uno** para mostrarme el resultado y esperar mi confirmación.

### Regla de oro (aplica a todos los pasos)
- El código del grupo VIDEO es de **SOLO LECTURA**. No edites, muevas, renombres, reformatees ni borres ningún archivo suyo, ni siquiera para "arreglar" algo.
- Si para integrarnos falta algo en VIDEO, **no lo cambies**: resuélvelo en nuestro adaptador y anótalo en `docs/SOLICITUDES_A_VIDEO.md` para que ellos decidan.
- Todo lo nuestro va en carpetas propias: `imagen/`, `docs/`, `tests/`, `salidas/`.

---

**Paso 1 — Ubicar el repositorio de VIDEO y congelar su estado.**
- Encuentra dónde está clonado el repositorio del grupo VIDEO (raíz actual o subcarpeta). Si no lo encuentras, **pregúntame la ruta**.
- Registra su estado de partida para demostrar después que no se tocó:
```bash
cd <ruta_repo_video>
git status
git log --oneline -5
git rev-parse HEAD > /tmp/video_commit_inicial.txt
find . -path ./.git -prune -o -type f -print0 | sort -z | xargs -0 sha256sum > /tmp/video_hash_inicial.txt
```
- Crea una rama propia para nuestro trabajo (`git checkout -b imagen/integracion`). **No hagas commits ni push sin mostrarme el diff.**

**Paso 2 — Auditar el proyecto de VIDEO (solo lectura) y hacer el inventario.**
Revisa todo el repositorio: README, estructura, dependencias, punto de entrada, scripts, notebooks, datos de prueba y salidas generadas. **Ejecútalo** con un video de prueba (en un `venv` aparte, sin modificar archivos) para confirmar que funciona y ver qué produce de verdad. Crea `docs/INVENTARIO_VIDEO.md` con:

| Pregunta | Qué anotar |
|---|---|
| ¿Qué problemática plantea VIDEO? | Resumen en 3–5 líneas (o "no está definida") |
| ¿Cómo se ejecuta? | Comando exacto, dependencias, versión de Python |
| ¿Qué entradas acepta? | Archivo de video, cámara, formatos, resolución |
| ¿Qué **entrega**? | Frames en memoria / imágenes en disco / generador / clase; formato (BGR/RGB, dtype, tamaño) |
| ¿Qué metadatos da? | `frame_id`, `timestamp`, FPS, duración, nombre de la fuente |
| ¿Cada cuántos frames muestrea? | FPS de salida, salto de frames |
| ¿Separa el audio? | Sí/No, formato (wav), ruta, frecuencia de muestreo (importante para el grupo AUDIO) |
| ¿Qué funciones/clases son **reutilizables** por importación? | Nombre, archivo, firma, qué devuelven |
| ¿Hace ya algún procesamiento de imagen? | Por ejemplo redimensionar, pasar a grises, detección. Así evitamos duplicar o nos apoyamos en ello |
| ¿Qué falta o qué está mal para integrarse? | Brechas, errores, rutas fijas, dependencias raras |

Termina el inventario con una lista **"Lo que VIDEO nos puede entregar"** y otra **"Lo que nos falta y debemos generar nosotros"**.

**Paso 3 — Evaluar la secuencia: ¿continuación (Modo A) o problemática propia (Modo B)?**
Con el inventario, evalúa y explícame:
- **Modo A — continuación:** la salida de VIDEO encaja con un análisis de imagen coherente con su problemática. Construimos IMAGEN como siguiente etapa del mismo caso.
- **Modo B — problemática propia:** su problemática no encaja con lo que el curso pide para imagen (detección y segmentación de objetos reales) o con lo que necesita PREDICCIÓN. En ese caso **no seguimos su problemática**, pero sí usamos lo que entregan (lectura de video, frames, metadatos, audio separado) como entrada. Ellos siguen siendo el primer eslabón de la cadena.

Para tu recomendación, compara los dos modos en una tabla con estos criterios:
- Coherencia con el pipeline VIDEO → (IMAGEN ‖ AUDIO) → PREDICCIÓN.
- Uso de las técnicas S02–S06 (avance de S07).
- Qué podríamos entregar a PREDICCIÓN.
- Esfuerzo.
- Riesgo.

Después propón las 2–3 problemáticas candidatas de la sección 3.1. **Espera mi elección.**

**Paso 4 — Documentar los requisitos.** Con la problemática elegida, ajusta los RNF, las normas y los RF de la sección 3 y crea `docs/REQUISITOS.md` con:
- Problemática
- Tabla de RNF
- Tabla de normas ISO
- Tabla de RF
- **Matriz de trazabilidad Problemática → RNF → ISO → RF**

Los RF de integración deben citar la función o archivo concreto de VIDEO que se consume, según el inventario.

**Paso 5 — Diseñar la integración sin tocar VIDEO.** Muéstrame el plan **antes** de escribir código. Elige el mecanismo según el inventario, de menor a mayor acoplamiento:
1. **Importar** sus funciones o clases desde nuestro adaptador (`from <modulo_video> import ...`), sin editar su código. Si hace falta, agrega la ruta con `sys.path` o instala su paquete en modo editable **solo si ya tiene `setup.py`/`pyproject.toml`**.
2. **Leer sus salidas** (frames en disco, CSV o JSON de metadatos, wav) desde una carpeta de intercambio.
3. **Ejecutarlo como proceso externo** (`subprocess`) y leer lo que genera.
4. Si nada de lo anterior sirve, **reimplementar en nuestro adaptador** solo la parte mínima (por ejemplo, leer el video con `cv2.VideoCapture`) y documentarlo como solicitud a VIDEO.

El plan debe incluir:
- El diagrama del flujo final.
- El mecanismo elegido y su justificación.
- Qué datos se transforman en el adaptador.
- Cómo se ejecuta todo el pipeline con un solo comando, por ejemplo:
```bash
python -m imagen.ejecutar --video datos/prueba.mp4 --salida salidas/
```

**Paso 6 — Implementar lo que falta** (núcleo RF-01 a RF-10):
```
imagen/
├── __init__.py
├── config.py              # parámetros (tamaño, filtros, umbrales)
├── adaptador_video.py     # ÚNICO punto de contacto con el código de VIDEO
├── preprocesamiento.py    # color, ruido, normalización     (S02–S03)
├── mejoramiento.py        # histograma, CLAHE, gamma, geom.  (S04)
├── segmentacion.py        # umbrales, morfología, contornos (S05)
├── caracteristicas.py     # Sobel, Canny, Harris, ORB        (S06)
├── metricas.py            # PSNR, SSIM, MSE, tiempos         (S03)
├── pipeline.py            # procesar_frame(dict) -> dict     (contrato sección 4)
├── exportar.py            # .jsonl y .csv para PREDICCIÓN
└── ejecutar.py            # punto de entrada: VIDEO → adaptador → pipeline → exportar
tests/test_imagen.py
tests/test_integracion_video.py   # prueba con un video real a través del adaptador
salidas/                          # evidencias (en .gitignore)
docs/INVENTARIO_VIDEO.md
docs/SOLICITUDES_A_VIDEO.md
docs/REQUISITOS.md
```
- Solo `adaptador_video.py` conoce el código de VIDEO. Si VIDEO cambia, solo cambia ese archivo.
- Código comentado en español, funciones cortas y sin variables globales.

**Paso 7 — Verificar la integración y que VIDEO quedó intacto.**
```bash
cd <ruta_repo_video>
git diff --stat -- <archivos_de_video>      # debe salir vacío
find . -path ./.git -prune -o -type f -print0 | sort -z | xargs -0 sha256sum > /tmp/video_hash_final.txt
diff /tmp/video_hash_inicial.txt /tmp/video_hash_final.txt && echo "VIDEO intacto"
```
Excluye de la comparación los archivos nuevos que estén **solo** en nuestras carpetas (`imagen/`, `docs/`, `tests/`, `salidas/`).
Luego corre el pipeline completo con al menos un video real de VIDEO y reporta:
- Tiempo por frame.
- Tabla comparativa de filtros (PSNR, SSIM, tiempo).
- Imágenes antes/después en `salidas/`.
- Un `.jsonl` y un `.csv` de ejemplo para PREDICCIÓN.
- `pytest` en verde.

**Paso 8 — Material para el avance (S07).** Genera `docs/AVANCE_S07.md` con:
- Problemática elegida y su relación con VIDEO.
- Diagrama de integración.
- Técnicas usadas por semana.
- Resultados y métricas.
- Limitaciones.
- Solicitudes a VIDEO.
- Próximos pasos (RF-11 a RF-15).

Debe servir de base para el informe PDF y la exposición.

### Comandos base (Debian)
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r <ruta_repo_video>/requirements.txt   # si existe: primero lo de VIDEO
pip install numpy scipy opencv-python scikit-image scikit-learn matplotlib pytest
pip freeze > requirements-imagen.txt
```
> No sobrescribas el `requirements.txt` de VIDEO; el nuestro es `requirements-imagen.txt`.

### Reglas
- Solo las librerías del curso (NumPy, SciPy, OpenCV, Scikit-learn, TensorFlow/Keras); `scikit-image` solo para las métricas SSIM/PSNR.
- Si algo del proyecto de VIDEO no está claro, **pregunta, no asumas**.
- Antes de hacer commits, muéstrame el diff.

---

## 6. Pendientes que debe resolver el grupo (no Claude Code)

1. Elegir la problemática del grupo IMAGEN entre las candidatas del Paso 3 (Modo A o Modo B) y comunicárselo a AUDIO y PREDICCIÓN.
1b. Enviar al grupo VIDEO las solicitudes de `docs/SOLICITUDES_A_VIDEO.md`. Si ellos las aplican, solo se ajusta nuestro adaptador.
2. Validar con el docente las normas ISO que se usarán.
3. Acordar el contrato JSON con AUDIO y PREDICCIÓN (sobre todo `timestamp` y la frecuencia de muestreo).
4. Decidir quién clasifica: IMAGEN (CNN/YOLO) o PREDICCIÓN.
5. Definir qué se muestra en el avance de **S07 (12–16 oct)** y qué queda para la defensa final de **S14**.
