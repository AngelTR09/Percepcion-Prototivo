# Sistema de Supervisión Inteligente

Sistema de supervisión de exámenes del curso de Percepción. Integra en una sola página web el trabajo de los grupos **VIDEO**, **IMAGEN** y **PREDICCIÓN**, conectados por eventos:

```
VIDEO  ──alerta──▶  IMAGEN  ──imagen mejorada + JSON──▶  PREDICCIÓN
cámara, detección     preprocesamiento, mejora,          alumnos, emociones,
y alarma (5 s)        segmentación, características,     causas, consecuencias
                      verificación                        y riesgo
```

## Estructura

| Carpeta | Grupo | Contenido |
|---|---|---|
| `video/` | VIDEO | Su sistema original: `principal.py`, `modulos/`, modelos y sonido de alarma |
| `imagen/` | IMAGEN | `modulo_imagen/` (procesamiento), calibración y vigilante de alertas |
| `prediccion/` | PREDICCIÓN | Su `backend/` (motor de reglas y modelos entrenados) y el modelo de pose |
| `integracion/` | Integración | Cámara de VIDEO dentro de la página y eventos entre módulos |
| `app.py` | Integración | Página web (Streamlit) con las secciones Video, Imágenes y Predicciones |

Cada grupo trabaja **solo en su carpeta**. El código de VIDEO y de PREDICCIÓN se mantiene sin modificar; la integración se adapta a él desde `integracion/`.

Repositorios de origen:

- VIDEO: https://github.com/cesarcabanillas1921-bit/SistemaSupervisionExamenes
- IMAGEN: https://github.com/Zudo45/SistemaSupervisionExamenes-Imagen
- PREDICCIÓN: https://github.com/alessandrorr1007-debug/Prediccion_GD

## Ramas

| Rama | Uso |
|---|---|
| `main` | Versión estable, la que se presenta. Siempre debe funcionar |
| `video` | Cambios del grupo VIDEO en prueba |
| `imagen` | Cambios del grupo IMAGEN en prueba |
| `prediccion` | Cambios del grupo PREDICCIÓN en prueba |

Forma de trabajo:

1. Trabajar en la rama del grupo: `git switch imagen` (por ejemplo).
2. Modificar solo la carpeta del grupo y subir los cambios: `git push`.
3. Cuando funcione, abrir un **Pull Request** de la rama hacia `main` en GitHub para que se revise e integre.

## Inicio rápido (Windows)

1. Instalar **Python 3.12** marcando "Add Python to PATH".
2. Descargar el repositorio (`git clone` o **Code → Download ZIP**).
3. Doble clic en **`INICIAR.bat`** (es el único arranque). Hace todo solo:
   - la primera vez crea el entorno, instala las librerías y descarga `cloudflared` (necesita internet y tarda varios minutos);
   - cierra restos de un inicio anterior;
   - abre la página en esta laptop, **enciende la cámara** y crea un link público `https://…trycloudflare.com` que funciona desde cualquier red.
4. Compartir el **link** y la **clave** que aparecen en la ventana negra (también en la barra lateral de la página).

- Quien entra por el link pone la clave y ve la cámara de la laptop que abrió `INICIAR.bat` (solo puede mirar).
- Cada inicio da un link y una clave **nuevos**. Al cerrar la ventana negra el link viejo deja de funcionar (error 1033 de Cloudflare): hay que pasar el nuevo.
- Desde la misma red también sirve `http://<IP>:8501` (barra lateral). El video usa además el puerto `8502`.

## Datos y privacidad

Las capturas de las alarmas (`video/alertas/`) y los resultados generados (`resultados/`) **no se suben** al repositorio (ver `.gitignore`).
