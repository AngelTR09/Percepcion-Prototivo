# Smart Classroom Vision — panel y entrega de avance

Contenido, con un propósito para cada parte:

| Carpeta | Qué es | Qué hacer con ella |
|---|---|---|
| *(raíz del repositorio)* | Panel web (Flask) con métricas del modelo, calidad del dataset e indicadores por minuto: `app.py`, `pagina.html`, `render.yaml`, `datos/`, `static/` | **Desplegar en Render** (Blueprint) |
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
- La imagen se genera solo mientras alguien la mira, no se guarda en disco y **nunca se envía a Render**.
- El difuminado puede fallar con rostros pequeños, de perfil o parcialmente tapados (los del fondo del aula, sobre todo). No la muestres fuera del aula.
- En la demostración, la etiqueta «cámara: reconectando» aparece un instante cada vez que el video de 30 s vuelve a empezar.

## Cámara real en la página de Render (opcional)

Con `NUBE_IMAGEN=1` en el `.env` de `2_docker/`, la página de Render muestra la cámara del aula (sección **«Cámara del aula en vivo»**, botón *Ver cámara en vivo*, o `.../#camara`), y a la vez el sistema sigue generando las estadísticas.
```bash
cd 2_docker          # .env con NUBE_URL, NUBE_CLAVE y NUBE_IMAGEN=1
docker compose up -d --build     # usa la cámara USB (cierra antes el navegador u otro programa que la use)
```
Cómo se limita el riesgo para la privacidad:
- Solo se envía la imagen **ya difuminada**, y **únicamente mientras alguien tiene abierta esa sección** en Render. Si nadie mira, el aula solo pregunta cada 3 s y no envía nada.
- Render la guarda **solo en memoria** (un único cuadro), nunca en disco, y la descarta en ~8 s si deja de llegar.
- Requiere contraseña, y el aula se identifica con la API_KEY. Solo acepta JPEG de hasta 400 KB.
- **Esto se aparta del RNF02** («no se almacenará video ni imágenes con rostros» y procesamiento local): hay que declararlo en el informe, avisar en el aula y contar con el consentimiento de quienes aparecen. El difuminado falla con rostros pequeños, de perfil o tapados.

## 2. Publicar el panel en Render

1. Este repositorio ya tiene la estructura que Render necesita en la raíz.
2. En Render: **New + → Blueprint** y elige ese repositorio. Render lee `render.yaml`.
3. En el servicio creado, pestaña **Environment**, copia `PANEL_CLAVE` (contraseña del panel) y `API_KEY` (clave para que el aula envíe datos).
4. Abre la URL `https://<tu-servicio>.onrender.com` · usuario `docente` · contraseña = `PANEL_CLAVE`.

Sin datos del aula, el panel muestra una **sesión de demostración** (etiquetada como tal) y las métricas reales del modelo.

### Conectar el Docker con Render (opcional)
```bash
cd 2_docker
NUBE_URL=https://<tu-servicio>.onrender.com NUBE_CLAVE=<API_KEY> \
  docker compose -f docker-compose.demo.yml up --build
```
Cada indicador (solo números) se envía a Render y el panel pasa a decir «datos del aula en vivo».

**Limitaciones del plan gratuito de Render**
- El servicio se «duerme» tras ~15 min sin uso; la primera visita tarda ~1 minuto.
- El disco es efímero: los indicadores enviados se pierden al reiniciar. Para conservarlos habría que añadir una base de datos.
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
