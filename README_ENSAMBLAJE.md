# Ensamblaje: VIDEO + IMAGEN + PREDICCION

Un solo servicio, un solo Dockerfile, desplegable en Render.

```
VIDEO (best.pt)  ->  IMAGEN (mejora, verifica)  ->  PREDICCION (indice y evidencia)
integrador/video_base   integrador/imagen             prediccion/
```

| Carpeta | Contenido | Se modifica |
|---|---|---|
| `integrador/` | VIDEO (copia de solo lectura) + IMAGEN, tal como estaban | No |
| `prediccion/` | AulaPredict completo (backend FastAPI, frontend, modelos, 154 pruebas) | No |
| `ensamble/` | El puente: arma el JSON de PREDICCION a partir de lo que producen VIDEO e IMAGEN | Aqui |
| `servidor.py` | Servidor unico: `/salud`, `/api/integrado/*`, `/integrado` y PREDICCION montada en `/` | Aqui |

## Como fluye una imagen (`POST /api/integrado/analizar`)

1. **VIDEO** detecta objetos en el frame original (su `best.pt`, via `integrador/imagen/adaptador_video.py`).
2. **IMAGEN** corrige luz (gamma), filtra ruido, aplica CLAHE, segmenta, extrae rasgos y vuelve a pasar el detector de VIDEO sobre el frame mejorado.
3. `ensamble/puente.py` compara ambas detecciones (misma clase, IoU >= 0.5):
   `confirmada` si VIDEO ya lo veia, `recuperada` si solo aparece tras mejorar. Lo que la mejora pierde se descarta.
4. **PREDICCION** recibe `video.frames` + `imagen.frames` por su propia funcion `integrar_fuentes` y devuelve el indice explicable.

Cuerpo: `{"imagen_b64": "...", "timestamp": 0, "objetos_permitidos": ["libro"], "devolver_imagen": true}`.
Si el detector de VIDEO no esta disponible responde `503 detector_no_disponible`; entrada invalida, `422`; mas de 8 MB, `413`.

## Ejecutar

```bash
docker compose up --build            # http://localhost:8000  (panel de PREDICCION)  y  /integrado  (la cadena)
docker build --build-arg CON_VIDEO=0 -t supervision-ligera .   # sin torch/YOLO: imagen liviana, la cadena responde 503
```
Sin Docker (Python 3.12):
```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt -r requirements-video.txt
uvicorn servidor:app --port 8000
```

## Desplegar gratis en la nube

Memoria medida dentro del contenedor Linux con tope duro de 512 MiB: **384 MiB (75 %) tras 10 imagenes seguidas con la cadena completa, sin OOM**. (Un proceso suelto en Windows llega a ~590 MB; el contenedor usa menos.)

| Opcion | Costo | RAM | Cadena VIDEO -> IMAGEN -> PREDICCION |
|---|---|---|---|
| **Hugging Face Spaces** (Docker) | gratis | 16 GB, 2 vCPU | **completa, con holgura** |
| Render, plan `free` (`render.yaml`) | gratis | 512 MB, CPU compartida | completa, pero justa de memoria y lenta al despertar |
| Render, plan `standard` | de pago | 2 GB | completa |

**Hugging Face Spaces (recomendado):**
```bash
pip install huggingface_hub
hf auth login                                                    # una vez; token de escritura creado en huggingface.co/settings/tokens
python deploy/publicar_huggingface.py --space TU_USUARIO/supervision-inteligente
```
Sube solo lo commiteado y lo que necesita el Dockerfile (unos 43 MB). La primera construccion tarda varios minutos; luego abre `https://TU_USUARIO-supervision-inteligente.hf.space/integrado`.
Es publico salvo que pases `--privado`; no tiene inicio de sesion propio, usa solo datos ficticios. Un Space gratuito se duerme tras un tiempo sin visitas y despierta al entrar. Para apagarlo del todo: Settings > Delete this Space.
Para probar la imagen en local antes de subir: `python deploy/publicar_huggingface.py --preparar salidas/space` y `docker build -t space-prueba salidas/space`.

**Render gratis:** New + > **Blueprint** y elegir este repositorio (`render.yaml` ya esta en plan `free`).
El detector se carga en la primera peticion, no al arrancar; con CPU compartida esa primera peticion puede tardar decenas de segundos, y el servicio se duerme tras ~15 min sin visitas.
Si se reinicia por memoria: variable `INTEGRADO_DETECTOR=off` (la cadena responde 503; el panel y PREDICCION siguen).

Variables: `INTEGRADO_DETECTOR` (`video`|`off`), `FRONTEND_ORIGINS`, y las `AULAPREDICT_*` de PREDICCION (ver `prediccion/.env.example`).

Privacidad: el seguimiento academico de PREDICCION rechaza conexiones remotas por diseno de su grupo, asi que en Render esos endpoints responden como rechazados. Los demas funcionan.

## Pruebas

```bash
python -m pytest tests -q                      # ensamblaje: puente, servidor, cadena real con best.pt
(cd integrador && python -m pytest tests -q)   # VIDEO + IMAGEN
(cd prediccion && python -m pytest tests -q)   # PREDICCION
```
CI (`.github/workflows/ci.yml`) ejecuta las tres suites y construye y arranca la imagen Docker en cada Pull Request.

## Politica de ramas

Ver [CONTRIBUIR.md](CONTRIBUIR.md): a `main` solo se llega por Pull Request aprobado por el dueno, con CI en verde y la rama al dia con `main`.
