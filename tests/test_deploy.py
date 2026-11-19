"""deploy/publicar_huggingface.py: la carpeta del Space debe salir completa, sin lo que sobra y con el Dockerfile ajustado."""
import importlib.util
import shutil
import subprocess

import pytest

from conftest import RAIZ

RUTA = RAIZ / "deploy" / "publicar_huggingface.py"
hay_git = shutil.which("git") is not None and subprocess.run(
    ["git", "rev-parse", "--git-dir"], cwd=RAIZ, capture_output=True).returncode == 0


@pytest.fixture(scope="module")
def space(tmp_path_factory):
    if not hay_git:
        pytest.skip("no hay un repositorio git")
    espec = importlib.util.spec_from_file_location("publicar_huggingface", RUTA)
    modulo = importlib.util.module_from_spec(espec)
    espec.loader.exec_module(modulo)
    destino = tmp_path_factory.mktemp("space")
    modulo.preparar(destino)
    return modulo, destino


def test_contiene_lo_que_necesita_el_dockerfile(space):
    _, d = space
    for ruta in ("Dockerfile", "README.md", "servidor.py", "requirements.txt", "requirements-video.txt",
                 "ensamble/integrado.html", "integrador/imagen/pipeline.py", "integrador/video_base/modelo_ia/best.pt",
                 "prediccion/backend/app.py", "prediccion/backend/entrenamiento_prediccion/modelo_sucesos_academicos.pkl"):
        assert (d / ruta).is_file(), ruta


def test_no_sube_lo_que_sobra(space):
    _, d = space
    for ruta in ("integrador/resultados", "integrador/datos", "integrador/tests", "prediccion/tests", "prediccion/docs", "tests"):
        assert not (d / ruta).exists(), ruta


def test_dockerfile_ajustado_para_un_space(space):
    _, d = space
    texto = (d / "Dockerfile").read_text(encoding="utf-8")
    assert "--uid 1000" in texto and "10001" not in texto
    assert "PORT=7860" in texto and "EXPOSE 7860" in texto
    codigo = "\n".join(l for l in texto.splitlines() if not l.lstrip().startswith("#"))   # los comentarios no se ejecutan
    assert "8000" not in codigo


def test_readme_declara_docker_y_el_mismo_puerto(space):
    _, d = space
    cabecera = (d / "README.md").read_text(encoding="utf-8").split("---")[1]
    assert "sdk: docker" in cabecera and "app_port: 7860" in cabecera


def test_el_dockerfile_original_no_se_toca():
    assert "--uid 10001" in (RAIZ / "Dockerfile").read_text(encoding="utf-8")
