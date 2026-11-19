"""Publica el sistema integrado (VIDEO + IMAGEN + PREDICCION) en Hugging Face Spaces: gratis, Docker, CPU con 16 GB de RAM.

Un Space gratuito da 16 GB de RAM y 2 vCPU: la cadena completa corre con holgura (en Render gratis, 512 MB, cabe justo: 384 MiB medidos).

  hf auth login                                         # una vez; token con permiso de escritura (tú lo pegas, no se comparte con nadie)
  python deploy/publicar_huggingface.py --space TU_USUARIO/supervision-inteligente

  python deploy/publicar_huggingface.py --preparar salidas/space      # solo arma la carpeta, sin subir (para probar el Docker en local)

Sube el contenido COMMITEADO (HEAD) y solo lo que el Dockerfile necesita, con el mismo recorte que `.dockerignore`.
El Space es público salvo que uses --privado. No tiene inicio de sesión propio: usar solo datos ficticios.
"""
import argparse
import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
RUTAS = ["ensamble", "integrador", "prediccion", "servidor.py", "requirements.txt", "requirements-video.txt", "Dockerfile"]
EXCLUIR = ("integrador/resultados/", "integrador/docs/", "integrador/datos/", "integrador/salidas/", "integrador/tests/",
           "prediccion/docs/", "prediccion/ejemplos/", "prediccion/tests/")
PUERTO = "7860"
LIMITE_MB = 500

README = """---
title: Supervision Inteligente
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: {puerto}
pinned: false
short_description: VIDEO, IMAGEN y PREDICCION sobre una imagen del aula
---

# Supervisión inteligente: VIDEO → IMAGEN → PREDICCIÓN

Prototipo académico de Percepción Computacional. VIDEO detecta objetos, IMAGEN mejora y verifica, PREDICCIÓN calcula un índice de revisión.
La salida es evidencia para revisión humana, no una prueba. Usar solo imágenes ficticias o de consentimiento: el servicio no tiene inicio de sesión.

- `/integrado`: sube una imagen y mira la cadena completa.
- `/`: panel de PREDICCIÓN. `/api/integrado/estado`: estado del detector.
- La primera petición tarda en cargar el detector de VIDEO.

Código: https://github.com/AngelTR09/Percepcion-Prototivo
"""


def extraer_commiteado(destino: Path):
    """Copia a `destino` las rutas del Dockerfile tal como están en HEAD (nada sin commitear, nada ignorado)."""
    crudo = subprocess.run(["git", "archive", "--format=tar", "HEAD", *RUTAS], cwd=RAIZ, check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(crudo)) as tar:
        miembros = [m for m in tar.getmembers() if not (m.name + "/").startswith(EXCLUIR)]
        tar.extractall(destino, members=miembros, filter="data")


def parchar_dockerfile(destino: Path):
    """Un Space corre como usuario 1000 y escucha en el puerto declarado en `app_port`."""
    ruta = destino / "Dockerfile"
    texto = ruta.read_text(encoding="utf-8")
    for viejo, nuevo in (("--uid 10001", "--uid 1000"), ("PORT=8000", f"PORT={PUERTO}"), ("EXPOSE 8000", f"EXPOSE {PUERTO}"),
                         ("environ.get('PORT','8000')", f"environ.get('PORT','{PUERTO}')"),
                         ("${PORT:-8000}", "${PORT:-" + PUERTO + "}")):
        if viejo not in texto:
            sys.exit(f"El Dockerfile cambió y no contiene '{viejo}': actualiza deploy/publicar_huggingface.py")
        texto = texto.replace(viejo, nuevo)
    ruta.write_text(texto, encoding="utf-8", newline="\n")


def preparar(destino: Path) -> float:
    destino.mkdir(parents=True, exist_ok=True)
    extraer_commiteado(destino)
    parchar_dockerfile(destino)
    (destino / "README.md").write_text(README.format(puerto=PUERTO), encoding="utf-8", newline="\n")
    mb = sum(f.stat().st_size for f in destino.rglob("*") if f.is_file()) / 1e6
    if mb > LIMITE_MB:
        sys.exit(f"La carpeta pesa {mb:.0f} MB (límite de seguridad {LIMITE_MB} MB): revisa EXCLUIR.")
    return mb


def main():
    p = argparse.ArgumentParser(description="Publica en Hugging Face Spaces (gratis)")
    p.add_argument("--space", help="USUARIO/NOMBRE del Space, por ejemplo AngelTR09/supervision-inteligente")
    p.add_argument("--privado", action="store_true", help="crea el Space privado")
    p.add_argument("--preparar", metavar="CARPETA", help="solo arma la carpeta del Space, sin subir nada")
    a = p.parse_args()

    if a.preparar:
        mb = preparar(Path(a.preparar))
        print(f"Carpeta lista en {a.preparar} ({mb:.0f} MB). Probar: docker build -t space-prueba {a.preparar}")
        return
    if not a.space or "/" not in a.space:
        sys.exit("Indica --space USUARIO/NOMBRE")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("Falta huggingface_hub: pip install huggingface_hub")
    api = HfApi()
    try:
        quien = api.whoami()["name"]
    except Exception:
        sys.exit("No hay sesión de Hugging Face. Ejecuta: hf auth login")
    print(f"Sesión: {quien}")

    with tempfile.TemporaryDirectory() as tmp:
        mb = preparar(Path(tmp))
        print(f"Subiendo {mb:.0f} MB a https://huggingface.co/spaces/{a.space} ({'privado' if a.privado else 'público'})")
        api.create_repo(a.space, repo_type="space", space_sdk="docker", private=a.privado, exist_ok=True)
        api.upload_folder(folder_path=tmp, repo_id=a.space, repo_type="space", commit_message="Despliegue del sistema integrado")
    usuario, nombre = a.space.split("/", 1)
    print(f"\nListo. El Space construye la imagen (varios minutos la primera vez):\n  panel: https://huggingface.co/spaces/{a.space}")
    print(f"  app  : https://{usuario.lower()}-{nombre.lower()}.hf.space/integrado")


if __name__ == "__main__":
    main()
