# Como subir cambios

`main` es la version que se despliega en Render. Esta protegida:

- **Nadie empuja directo a `main`**, ni el dueno ni los colaboradores: solo entra por Pull Request.
- El Pull Request necesita la **aprobacion del dueno** (`.github/CODEOWNERS`) y que **CI este en verde** (`pruebas` y `docker`).
- La rama debe estar **al dia con `main`** antes de poder fusionarse.
- Los colaboradores si pueden **crear y subir su propia rama** y abrir el Pull Request.
- Sin `force push` ni borrado de `main`.

## Flujo

```bash
git switch main && git pull
git switch -c mi-cambio                 # una rama por cambio
# ... trabajar ...
git merge main                          # interactuar con main: traer lo ultimo y resolver conflictos en tu rama
python -m pytest tests -q               # probar en la rama
(cd integrador && python -m pytest tests -q)
(cd prediccion && python -m pytest tests -q)
git push -u origin mi-cambio            # CI corre solo
```
Luego abre el Pull Request hacia `main`. Si `main` avanza mientras esperas, vuelve a hacer `git merge main`, prueba y sube.

Cada grupo toca solo lo suyo: `integrador/video_base/` es copia de solo lectura de VIDEO, `prediccion/` es de PREDICCION, `integrador/imagen/` de IMAGEN, `ensamble/` y `servidor.py` son del ensamblaje.
