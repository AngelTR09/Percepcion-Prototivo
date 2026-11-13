# Análisis de sesgo y ética (RNF-09, S15)

Marco: estructura de ISO/IEC TR 24027 (sesgo en sistemas de IA): origen en los datos, en el algoritmo y en el uso. No se declara conformidad con la norma; se usa para ordenar el análisis.

## 1. Qué se puede y qué no se puede auditar con lo que hay

| Pregunta | ¿Se puede? | Por qué |
|---|---|---|
| ¿El detector rinde peor con ciertas personas (edad, género, tono de piel)? | **No** | El video no tiene etiquetas demográficas y no se infieren: hacerlo sería un riesgo mayor que el sesgo que se quiere medir. |
| ¿El dataset con que se entrenó `best.pt` está balanceado? | **No** | El dataset (~4 GB) no está en el repositorio; solo se versionan el script `filtrar_dataset.py` y el modelo. Hay que pedírselo al grupo VIDEO. |
| ¿El detector y la mejora rinden parejo entre clases, tamaños y posiciones? | **Sí** | Se mide en `imagen/sesgo.py` con los frames del proyecto. Es un sesgo de desempeño, no demográfico. |

## 2. Resultado medido: desempeño por condición bajo poca luz
`python -m imagen.ejecutar_rendimiento --video datos/padang.webm --salida salidas/rendimiento --cada-n 15 --sesgo --luz 0.15`
42 frames de `padang.webm`, luz al 15 % más ruido. Se mide qué fracción de las detecciones del frame limpio reaparece (misma clase, IoU ≥ 0.5) en el frame degradado, sin y con la mejora de IMAGEN. La brecha es la diferencia entre el mejor y el peor grupo (0 = parejo).

| Criterio | Grupo | Referencias | Sin mejora | Con mejora |
|---|---|---|---|---|
| Clase | persona | 521 | 0.36 | 0.64 |
| Clase | mochila | 67 | 0.12 | 0.16 |
| | *brecha* | | 0.24 | **0.47** |
| Tamaño | grande (≥ 5 % del frame) | 151 | 0.70 | 0.90 |
| Tamaño | mediano (1–5 %) | 293 | 0.23 | 0.48 |
| Tamaño | pequeño (< 1 %) | 144 | 0.17 | 0.46 |
| | *brecha* | | 0.53 | 0.44 |
| Posición | centro | 261 | 0.46 | 0.68 |
| Posición | borde | 327 | 0.24 | 0.50 |
| | *brecha* | | 0.22 | 0.18 |

Lectura:
- **La mejora no beneficia a todos por igual.** Duplica casi la recuperación de personas (0.36 → 0.64) pero apenas mueve la de mochilas (0.12 → 0.16): la brecha entre clases **empeora** (0.24 → 0.47). Una alerta por mochila sería mucho menos confiable que una por persona con poca luz.
- **Objetos pequeños y de los bordes** se recuperan peor siempre; la mejora reduce algo esa brecha (0.53 → 0.44 por tamaño, 0.22 → 0.18 por posición), pero no la cierra: con poca luz, un objeto pequeño o al borde se pierde con mucha más frecuencia (0.46 con mejora frente a 0.90 de los grandes).
- Esto puede deberse a que la mochila es la clase minoritaria (67 frente a 521 referencias) y a cómo se entrenó el detector, algo que este análisis no puede separar.

## 3. Condiciones de falla documentadas
| Condición | Efecto medido | Dónde |
|---|---|---|
| Poca luz (15 %) | Se pierde más de la mitad de las detecciones sin mejora; la mejora recupera hasta 0.64 en personas | `AVANCE_S07.md` §4, esta sección |
| Video de baja resolución y comprimido | La mejora **empeora** el resultado (90 → 76 de 295 detecciones en `aula_demo.mp4`) | `AVANCE_S07.md` §4 |
| Cámara inclinada o rotada | A 25° el detector conserva 0.30 de sus detecciones | `AVANCE_S07.md` §5 |
| Objetos pequeños o en el borde del encuadre | Recuperación 0.17–0.24 sin mejora | §2 |
| Clase mochila | Casi no se recupera con poca luz (0.16) | §2 |
| Frame volteado (espejo) | Ni ORB ni `matchTemplate` reconocen el patrón | `AVANCE_S07.md` §7 |

## 4. Límites de este análisis
- La "verdad" de referencia son las detecciones de VIDEO sobre el frame limpio, no etiquetas hechas a mano: si el detector se equivoca en el frame limpio, ese error pasa a la referencia.
- Un solo video de aula (Wikimedia Commons, CC BY-SA 4.0), 42 frames, solo personas y mochilas. No hay teléfonos, cuadernos ni audífonos, que son justamente los objetos que disparan las alertas.
- La degradación es sintética. Las brechas pueden cambiar con una cámara y un aula reales.
- Frames consecutivos del mismo video no son independientes: los números dan una idea, no un intervalo de confianza.

## 5. Uso responsable (lo que el sistema no debe hacer)
- **Una alerta no es una prueba.** El sistema marca objetos y PREDICCIÓN calcula un índice de revisión; la decisión sobre un estudiante la toma una persona que revisa la evidencia. Con poca luz o con mochilas, el sistema falla más.
- **No inferir** emociones, identidad, rasgos o intención a partir de los frames. IMAGEN solo mejora calidad y extrae características de bajo nivel (bordes, esquinas, color).
- **Privacidad (RNF-08).** El servidor integrado procesa cada imagen en memoria y no la guarda. Las salidas locales van a `salidas/`, que no se versiona. Las carpetas `integrador/resultados/` del repositorio contienen frames de los videos de prueba del proyecto (`padang.webm` es un video público), no de estudiantes de la universidad, salvo que alguien suba otros; con datos propios hay que pedir consentimiento informado, anonimizar rostros (existe la técnica en `documentos_S02`) y fijar un tiempo de retención.
- **Antes de usarlo con estudiantes reales:** medir la tasa de falsas alertas por clase y por condición de luz con datos del aula real, y revisar que ningún grupo de personas reciba más alertas por razones ajenas a su conducta (por ejemplo, ubicación del asiento en el borde del encuadre, que se mide peor).

## 6. Qué haría falta
1. El dataset de entrenamiento del grupo VIDEO, para auditar el balance por clase.
2. Un video de aula con las clases que importan (teléfono, cuaderno, audífonos) y etiquetas hechas a mano.
3. Repetir §2 con esos datos y fijar un umbral de brecha aceptable (por ejemplo, ≤ 0.15 entre clases).
