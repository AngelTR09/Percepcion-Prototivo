# Smart Classroom Vision — panel y entrega de avance

Contenido, con un propósito para cada parte:

| Carpeta | Qué es | Qué hacer con ella |
|---|---|---|
| *(raíz del repositorio)* | Panel web (Flask) con métricas del modelo, calidad del dataset y una sesión de demostración: `app.py`, `pagina.html`, `render.yaml`, `datos/`, `static/` | **Desplegar en Render** (Blueprint) |
| `2_docker/` | Sistema completo (cámara → detección → anonimización → indicadores) + modelo entrenado + video de demostración | **Ejecutar en tu computador con Docker** |
| `3_metricas/` | Evidencia del entrenamiento: curvas, matriz de confusión, ejemplos de predicción, CSV de resultados | **Adjuntar al informe** |

## 1. Ejecutar el sistema en Docker (evidencia del código corriendo)

Requisitos: Docker y Docker Compose. Cierra cualquier programa que use la cámara (el navegador, por ejemplo) si vas a usarla.

**Demostración sin cámara** (usa `demo/aula_demo.mp4`, imágenes reales del dataset):
```bash
cd 2_docker
docker compose -f docker-compose.demo.yml up --build
```
Abre **http://localhost:8000** · usuario `docente` · contraseña `docente123`.
En ~10 s aparece el primer indicador (en la demostración se genera uno cada 10 s; en el aula real, cada 60 s).

**Con la cámara USB del aula:**
```bash
cd 2_docker
docker compose up -d --build
```

Para detenerlo: `docker compose -f docker-compose.demo.yml down`.

Lo que se ve en el panel: personas, laptops, celulares, libros y los 5 comportamientos (levantar la mano, leer, escribir, cabeza agachada, cabeza girada), iluminación, calidad de imagen, FPS y un gráfico por minuto. El reporte CSV queda en `2_docker/datos/`.

**Vista previa de la cámara (opcional, solo en el panel local).** Muestra la imagen del aula con los **rostros difuminados** y las detecciones dibujadas. En la demostración está activada (`VISTA_PREVIA=1`): baja a la sección «Vista previa» del panel y pulsa **Ver vista previa** (o abre `http://localhost:8000/#vista`). En el aula real viene apagada; se activa con `VISTA_PREVIA=1 docker compose up -d --build`.
- La imagen se genera solo mientras alguien la mira, no se guarda en disco y **no se envía a ninguna parte**: solo se ve en el panel local.
- El difuminado puede fallar con rostros pequeños, de perfil o parcialmente tapados (los del fondo del aula, sobre todo). No la muestres fuera del aula.
- En la demostración, la etiqueta «cámara: reconectando» aparece un instante cada vez que el video de 30 s vuelve a empezar.

## 2. Publicar el panel en Render

1. Este repositorio ya tiene la estructura que Render necesita en la raíz.
2. En Render: **New + → Blueprint** y elige ese repositorio. Render lee `render.yaml`.
3. En el servicio creado, pestaña **Environment**, copia `PANEL_CLAVE` (contraseña del panel).
4. Abre la URL `https://<tu-servicio>.onrender.com` · usuario `docente` · contraseña = `PANEL_CLAVE`.

El panel de Render **no está conectado a ninguna cámara ni al Docker**: muestra las métricas reales del modelo, la calidad del dataset y una **sesión de demostración** de indicadores (etiquetada como tal), todo con datos fijos del repositorio.

**Limitaciones del plan gratuito de Render**
- El servicio se «duerme» tras ~15 min sin uso; la primera visita tarda ~1 minuto.
- El panel usa contraseña básica; Render sirve la página por HTTPS, así que viaja cifrada.

## 3. Qué decir sobre los resultados (con honestidad)

- Modelo YOLO11n a 640 px, 40 épocas: **mAP@0.5 = 0.434** en validación (2122 imágenes). **La meta del RNF06 (≥ 0.70) todavía no se cumple**: está en curso.
- Mejores clases: levantar la mano (0.578) y escribir (0.554). Peores: cabeza agachada (0.335) y cabeza girada (0.248).
- `BowHead` es el 6.1 % de las anotaciones de entrenamiento (el RNF09 pide ≥ 10 % por clase).
- Siguiente paso: entrenar YOLO11s a 960 px en Colab (`entrenar_colab.ipynb`, que no está en este repositorio).
- El prototipo aplica en tiempo real solo el control de nitidez; el control de calidad completo (RF03) está en el cuaderno de mejora de imagen.

## Checklist de evidencias (capturas)

- [ ] Terminal con `docker compose ... up` mostrando «Panel del docente: …».
- [ ] Panel del aula (`localhost:8000`) con cifras en «ahora» y al menos 2 minutos en la tabla.
- [ ] Intento de entrar sin contraseña (pantalla de acceso restringido).
- [ ] Panel de Render con la tabla de métricas por clase y la curva de entrenamiento.
- [ ] `3_metricas/entrenamiento/results.png`, `confusion_matrix_normalized.png` y `val_batch*_pred.jpg` (comportamientos detectados) y `mejora_imagen/fig5_anonimizacion.png` (rostros difuminados).
- [ ] `2_docker/datos/indicadores_aula.csv` (reporte exportado).
