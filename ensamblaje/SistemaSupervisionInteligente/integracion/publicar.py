# ============================================================
# PAGINA PUBLICA (TUNEL DE CLOUDFLARE)
# Inicia la pagina y dos tuneles de Cloudflare (sin cuenta):
#   - pagina: puerto 8501
#   - video de la camara: puerto 8502
# Los links publicos y una clave para las visitas se guardan en
# resultados/tunel.json, que la pagina lee para mostrarlos.
#
# Uso (desde la raiz del proyecto): INICIAR.bat
# ============================================================

import os
import re
import sys
import json
import time
import secrets
import threading
import subprocess
import webbrowser

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOUDFLARED = os.path.join(RAIZ, "herramientas", "cloudflared.exe")
RUTA_TUNEL = os.path.join(RAIZ, "resultados", "tunel.json")

PUERTO_PAGINA = 8501
PUERTO_VIDEO = 8502

def atar_procesos_hijos():
    """
    Windows: mete este proceso en un Job que mata a todos sus hijos
    (cloudflared y streamlit) cuando se cierra la ventana con la X.
    Sin esto los tuneles quedaban abiertos y ocupaban los puertos.
    """

    if os.name != "nt":
        return

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

    class LIMITES(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class LIMITES_EXT(ctypes.Structure):
        _fields_ = [
            ("Basic", LIMITES),
            ("IoInfo", ctypes.c_uint64 * 6),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    job = kernel32.CreateJobObjectW(None, None)
    info = LIMITES_EXT()
    info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess())
    atar_procesos_hijos.job = job  # mantener el handle abierto


def abrir_navegador_cuando_este_lista():

    def esperar():
        import urllib.request
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://localhost:{PUERTO_PAGINA}", timeout=2)
                webbrowser.open(f"http://localhost:{PUERTO_PAGINA}")
                return
            except OSError:
                time.sleep(1)

    threading.Thread(target=esperar, daemon=True).start()


PATRON_LINK = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def abrir_tunel(puerto, nombre, links):
    """
    Lanza cloudflared y guarda el link publico cuando aparece en su salida.
    """

    proceso = subprocess.Popen(
        [CLOUDFLARED, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{puerto}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    def leer():
        for linea in proceso.stdout:
            encontrado = PATRON_LINK.search(linea)
            if encontrado and nombre not in links:
                links[nombre] = encontrado.group(0)

    threading.Thread(target=leer, daemon=True).start()

    return proceso


def main():

    if not os.path.exists(CLOUDFLARED):
        print("No se encontro herramientas/cloudflared.exe")
        sys.exit(1)

    atar_procesos_hijos()

    os.makedirs(os.path.dirname(RUTA_TUNEL), exist_ok=True)

    if os.path.exists(RUTA_TUNEL):
        os.remove(RUTA_TUNEL)

    links = {}
    procesos = [
        abrir_tunel(PUERTO_PAGINA, "pagina", links),
        abrir_tunel(PUERTO_VIDEO, "video", links)
    ]

    print("Creando los links publicos...")

    inicio = time.time()

    while len(links) < 2 and time.time() - inicio < 60:
        time.sleep(0.5)

    if len(links) < 2:
        print("No se pudieron crear los links publicos (revisa la conexion a internet).")
        for p in procesos:
            p.terminate()
        sys.exit(1)

    clave = secrets.token_hex(3).upper()

    with open(RUTA_TUNEL, "w", encoding="utf-8") as f:
        json.dump({**links, "clave": clave, "pid": os.getpid()}, f, indent=2)

    print()
    print("=" * 60)
    print(f"  Link publico de la pagina: {links['pagina']}")
    print(f"  Clave para las visitas:    {clave}")
    print("=" * 60)
    print("  Comparte el link y la clave. Cierra esta ventana para terminar.")
    print()

    print("  La camara de esta laptop se enciende sola al abrir la pagina.")
    print()

    abrir_navegador_cuando_este_lista()

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", os.path.join(RAIZ, "app.py")],
            cwd=RAIZ,
            env={**os.environ, "SSI_AUTOCAMARA": "1"}
        )
    finally:
        for p in procesos:
            p.terminate()
        if os.path.exists(RUTA_TUNEL):
            os.remove(RUTA_TUNEL)


if __name__ == "__main__":
    main()
