"""
Smart Classroom Vision - Prototipo base (Equipo Imagen)
Detecta personas y objetos académicos, clasifica comportamientos observables,
anonimiza rostros y registra indicadores agregados por minuto (sin guardar video).
"""
import json
import os
import secrets
import signal
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Configuración (variables de entorno; los valores por defecto sirven para ejecutarlo directamente)
FUENTE = os.getenv("FUENTE", "0")                           # "0" = cámara USB; o "rtsp://..." para cámara IP
FUENTE = int(FUENTE) if FUENTE.isdigit() else FUENTE
INTERVALO_S = int(os.getenv("INTERVALO_S", "60"))           # indicadores agregados cada 60 segundos
SALIDA_CSV = Path(os.getenv("SALIDA_CSV", "indicadores_aula.csv"))
MOSTRAR_VENTANA = os.getenv("MOSTRAR_VENTANA", "1") == "1"  # 0 en Docker/servidor sin pantalla
PANEL = os.getenv("PANEL", "1") == "1"                      # panel web con los indicadores (RF09)
PANEL_HOST = os.getenv("PANEL_HOST", "127.0.0.1")           # 0.0.0.0 para abrirlo desde otro equipo del aula
PANEL_PUERTO = int(os.getenv("PANEL_PUERTO", "8000"))
PANEL_USUARIO = os.getenv("PANEL_USUARIO", "docente")
PANEL_CLAVE = os.getenv("PANEL_CLAVE") or secrets.token_urlsafe(6)   # RNF02: si no se define, se genera una

# Envío opcional de indicadores agregados a un panel en la nube (p. ej. Render). Desactivado si NUBE_URL está vacía.
NUBE_URL = os.getenv("NUBE_URL", "")
NUBE_CLAVE = os.getenv("NUBE_CLAVE", "")
AULA_ID = os.getenv("AULA_ID", "aula-1")
SESION = time.strftime("%Y%m%d_%H%M%S")
pendientes, candado_nube = [], threading.Lock()

# Vista previa opcional en el panel local: solo la imagen YA anonimizada, con las detecciones dibujadas.
# Se genera únicamente mientras alguien la mira, no se guarda en disco y nunca se envía a la nube.
VISTA_PREVIA = os.getenv("VISTA_PREVIA", "0") == "1"
ETIQUETAS_VISTA = {"person": "Persona", "laptop": "Laptop", "cell phone": "Celular", "book": "Libro", "hand-raising": "Levanta la mano",
                   "read": "Leer", "write": "Escribir", "BowHead": "Cabeza agachada", "TurnHead": "Cabeza girada"}
vista = {"jpg": None, "pedida": 0.0}

# Estado compartido con el panel web (solo indicadores agregados; nunca imágenes)
estado = {"vista": VISTA_PREVIA, "camara": "conectando", "actual": {}, "iluminacion": None, "apta": None, "fps": 0.0, "registros": []}
bloqueo = threading.Lock()

modelo_coco = YOLO("yolo11n.pt")                  # personas, celular, laptop, libro
RUTA_COMP = Path("modelo_comportamientos.pt")     # entrenado con SCB-Dataset + datos locales
modelo_comp = YOLO(str(RUTA_COMP)) if RUTA_COMP.exists() else None
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

# Detector de rostros con red neuronal (SSD ResNet-10 de OpenCV); se descarga la primera vez
PROTO, MODELO = "deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel"
if not Path(MODELO).exists():
    urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/"
                               "face_detector/deploy.prototxt", PROTO)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
                               "dnn_samples_face_detector_20170830/" + MODELO, MODELO)
red_rostros = cv2.dnn.readNetFromCaffe(PROTO, MODELO)


def preprocesar(frame):
    """RF01-RF03: reduce ruido (bilateral), corrige iluminación (gamma + CLAHE) y evalúa la calidad."""
    frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))   # resolución de trabajo (YOLO usa 640)
    gris640 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brillo = float(gris640.mean())
    apta = float(cv2.Laplacian(gris640, cv2.CV_64F).var()) >= 100   # imagen movida o desenfocada: no apta
    frame = cv2.bilateralFilter(frame, 9, 40, 40)                    # 1) ruido, antes de aclarar
    g = np.log(0.5) / np.log(max(brillo / 255, 1e-3))                # 2) gamma automática
    frame = cv2.LUT(frame, (((np.arange(256) / 255.0) ** g) * 255).astype(np.uint8))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])                         # 3) contraste local
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    alerta = "baja" if brillo < 60 else "alta" if brillo > 200 else "adecuada"
    return frame, alerta, apta


def anonimizar(frame, umbral=0.5):
    """RF07: difumina los rostros detectados por la red neuronal antes de mostrar la imagen."""
    h, w = frame.shape[:2]
    red_rostros.setInput(cv2.dnn.blobFromImage(frame, 1.0, (w, h), (104, 177, 123)))
    for d in red_rostros.forward()[0, 0]:
        if d[2] > umbral:
            x1, y1, x2, y2 = (d[3:7] * [w, h, w, h]).astype(int)
            x1, y1 = max(x1, 0), max(y1, 0)
            if x2 > x1 and y2 > y1:
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (31, 31), 20)
    return frame


def analizar(frame):
    """RF04, RF05 y RF06: conteo de estudiantes, comportamientos y objetos académicos. Devuelve (conteo, cajas)."""
    conteo, cajas = {}, []
    resultados = [modelo_coco(frame, classes=[0, 63, 67, 73], verbose=False)[0]]   # persona, laptop, celular, libro
    if modelo_comp:
        resultados.append(modelo_comp(frame, verbose=False)[0])
    for r in resultados:
        for c, caja, conf in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            nombre = r.names[int(c)]
            conteo[nombre] = conteo.get(nombre, 0) + 1
            cajas.append((*map(int, caja), nombre, conf))
    return conteo, cajas


def publicar_vista(frame, cajas):
    """RF07: los rostros se difuminan ANTES de dibujar o publicar la imagen."""
    img = anonimizar(frame.copy())
    for x1, y1, x2, y2, nombre, conf in cajas:
        color = tuple(int(v) for v in np.random.default_rng(abs(hash(nombre)) % 2**32).integers(60, 255, 3))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1 if nombre == "person" else 2)
        if nombre != "person":
            texto = f"{ETIQUETAS_VISTA.get(nombre, nombre)} {conf:.0%}"
            cv2.putText(img, texto, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        with bloqueo:
            vista["jpg"] = buf.tobytes()


def guardar_registros(registros):
    """RF09: exporta el reporte CSV de la sesión."""
    pd.DataFrame(registros).to_csv(SALIDA_CSV, index=False)


def enviar_a_nube(fila):
    """RNF02: envía solo el indicador agregado (números), nunca imágenes. Si falla, reintenta con el siguiente minuto."""
    pendientes.append({**fila, "aula": AULA_ID, "sesion": SESION})
    del pendientes[:-500]

    def _enviar():
        with candado_nube:
            while pendientes:
                req = urllib.request.Request(NUBE_URL.rstrip("/") + "/api/indicadores", json.dumps(pendientes[0]).encode(),
                                             {"Content-Type": "application/json", "X-API-Key": NUBE_CLAVE})
                try:
                    urllib.request.urlopen(req, timeout=10).read()
                    pendientes.pop(0)
                except Exception as e:
                    print("Panel en la nube no disponible, se reintentará:", e, flush=True)
                    return
    threading.Thread(target=_enviar, daemon=True).start()


def bucle_captura(detener):
    cap = cv2.VideoCapture(FUENTE)
    acumulado, n_frames, inicio, fps, ultima_vista = {}, 0, time.time(), 0.0, 0.0
    while not detener.is_set():
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:                                   # RNF05: reintento de conexión
            with bloqueo:
                estado["camara"] = "reconectando"
            cap.release(); time.sleep(2); cap = cv2.VideoCapture(FUENTE); continue
        frame, iluminacion, apta = preprocesar(frame)
        conteo, cajas = analizar(frame) if apta else ({}, [])
        if apta:                                     # RF03: solo cuadros aptos alimentan indicadores
            for k, v in conteo.items():
                acumulado[k] = acumulado.get(k, 0) + v
            n_frames += 1
        fps = 0.9 * fps + 0.1 / max(time.time() - t0, 1e-3) if fps else 1 / max(time.time() - t0, 1e-3)
        with bloqueo:
            estado.update(camara="conectada", iluminacion=iluminacion, apta=apta, fps=round(fps, 1))
            if apta:
                estado["actual"] = conteo
        if time.time() - inicio >= INTERVALO_S and n_frames:   # RF08: indicador agregado (sin identidades)
            fila = {k: round(v / n_frames, 1) for k, v in acumulado.items()}
            fila.update(minuto=len(estado["registros"]) + 1, iluminacion=iluminacion)
            with bloqueo:
                estado["registros"].append(fila)
                registros = list(estado["registros"])
            guardar_registros(registros)
            if NUBE_URL:
                enviar_a_nube(fila)
            acumulado, n_frames, inicio = {}, 0, time.time()
        if VISTA_PREVIA and time.time() - vista["pedida"] < 10 and time.time() - ultima_vista >= 0.4:
            ultima_vista = time.time()               # solo se genera si alguien la está mirando
            publicar_vista(frame, cajas)
        if MOSTRAR_VENTANA:                          # RF07: solo se muestra la imagen con los rostros difuminados
            cv2.imshow("Smart Classroom Vision (anonimizado)", anonimizar(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if MOSTRAR_VENTANA:
        cv2.destroyAllWindows()


def iniciar_panel():
    from panel_web import crear_app
    from waitress import serve
    app = crear_app(estado, bloqueo, SALIDA_CSV, PANEL_USUARIO, PANEL_CLAVE, vista if VISTA_PREVIA else None)
    threading.Thread(target=serve, args=(app,), kwargs={"host": PANEL_HOST, "port": PANEL_PUERTO, "_quiet": True},
                     daemon=True).start()
    print(f"Panel del docente: http://{'localhost' if PANEL_HOST in ('0.0.0.0', '127.0.0.1') else PANEL_HOST}:{PANEL_PUERTO}"
          f"  usuario: {PANEL_USUARIO}  contraseña: {PANEL_CLAVE}", flush=True)


def main():
    if SALIDA_CSV.exists():                          # conserva el reporte de la sesión anterior
        SALIDA_CSV.rename(SALIDA_CSV.with_name(
            f"{SALIDA_CSV.stem}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(SALIDA_CSV.stat().st_mtime))}{SALIDA_CSV.suffix}"))
    detener = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: detener.set())     # docker stop
    if PANEL:
        iniciar_panel()
    try:
        bucle_captura(detener)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
