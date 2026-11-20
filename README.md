# Sistema de Supervisión Inteligente: VIDEO + IMAGEN + PREDICCIÓN

Tres servicios vivos e independientes (cada uno con su propio contenedor, interfaz, API y código) y un portal que los reúne en una barra lateral. Se comunican por HTTP:

```
VIDEO ──hallazgo──▶ IMAGEN ──(solo si verifica)──▶ PREDICCIÓN
```

IMAGEN tiene la última palabra: lo que no verifica se descarta como falso positivo y no llega a PREDICCIÓN.

| Servicio | Puerto | Qué hace | Código |
|---|---|---|---|
| Portal | 8000 | Barra lateral con los tres apartados | `servicios/portal/` |
| VIDEO | 8001 | Detección en vivo, alerta a los 2,5 s | `servicios/video/`, `integrador/video_base/` |
| IMAGEN | 8002 | Verificación con las técnicas del curso (S03 a S06) | `servicios/imagen/`, `integrador/imagen/` |
| PREDICCIÓN | 8003 | Análisis facial y motor de evaluación | `servicios/prediccion/`, `prediccion/` |

## Desplegar

**Docker (local)**
```
docker compose up --build        # portal en http://localhost:8000
```
Cada servicio se abre también por su cuenta en su puerto.

**Render (plan gratis):** New + Blueprint, elegir este repositorio (`render.yaml` crea los 4 servicios). Si cambias los nombres, actualiza las URLs del archivo. VIDEO y PREDICCIÓN cargan YOLO y en el plan gratis caben justo.

## Salud y API
- Todos: `GET /salud`.
- VIDEO: `POST /api/detectar` (JPEG), `POST /api/hallazgo`.
- IMAGEN: `POST /api/recibir`, `GET /api/ultimo`.
- PREDICCIÓN: `POST /api/recibir`, `GET /api/ultimo` y su API original en `/api/*`.

Prototipo académico: la salida es evidencia para revisión humana; usar solo datos ficticios.
