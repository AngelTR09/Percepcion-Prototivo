import cv2
import os
from ultralytics import YOLO


# ==========================================================
# RUTA DEL PROYECTO
# ==========================================================

CARPETA_PROYECTO = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# ==========================================================
# MODELO ENTRENADO
# ==========================================================

RUTA_MODELO = os.path.join(
    CARPETA_PROYECTO,
    "modelo_ia",
    "best.pt"
)

modelo = YOLO(RUTA_MODELO)


# ==========================================================
# CLASES
# ==========================================================

CLASES = {
    0: "persona",
    1: "mochila",
    2: "telefono",
    3: "cuaderno",
    4: "libro",
    5: "audifonos",
    6: "reloj",
    7: "laptop"
}


# ==========================================================
# UMBRALES POR CLASE
# ==========================================================

UMBRAL = {
    "persona": 0.20,
    "mochila": 0.15,
    "telefono": 0.15,
    "audifonos": 0.15,
    "reloj": 0.15,
    "cuaderno": 0.80,
    "libro": 0.80,
    "laptop": 0.25
}


# ==========================================================
# DETECTAR OBJETOS
# ==========================================================

def detectar_objetos(frame):

    detecciones = []

    resultados = modelo.predict(
        source=frame,
        conf=0.15,
        iou=0.45,
        verbose=False
    )

    for resultado in resultados:

        if resultado.boxes is None:
            continue

        for caja in resultado.boxes:

            confianza = float(caja.conf[0])
            clase_id = int(caja.cls[0])

            if clase_id not in CLASES:
                continue

            nombre = CLASES[clase_id]

            umbral_clase = UMBRAL.get(
                nombre,
                0.25
            )

            if confianza < umbral_clase:
                continue

            x1, y1, x2, y2 = map(
                int,
                caja.xyxy[0]
            )

            detecciones.append({
                "clase": nombre,
                "confianza": confianza,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

    return detecciones


# ==========================================================
# DIBUJAR DETECCIONES
# ==========================================================

def dibujar_detecciones(frame, detecciones):

    objetos_permitidos = {
        "persona",
        "mochila"
    }

    nombres_pantalla = {
        "persona": "Persona",
        "mochila": "Mochila",
        "telefono": "Telefono",
        "audifonos": "Audifonos",
        "reloj": "Reloj",
        "cuaderno": "Cuaderno/Libro",
        "libro": "Cuaderno/Libro",
        "laptop": "Laptop"
    }

    for deteccion in detecciones:

        clase = deteccion["clase"]
        confianza = deteccion["confianza"]

        x1 = deteccion["x1"]
        y1 = deteccion["y1"]
        x2 = deteccion["x2"]
        y2 = deteccion["y2"]

        # Azul = permitido
        if clase in objetos_permitidos:
            color = (255, 0, 0)

        # Rojo = no permitido
        else:
            color = (0, 0, 255)

        etiqueta = nombres_pantalla.get(
            clase,
            clase
        )

        texto = f"{etiqueta} {confianza:.2f}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        posicion_y = y1 - 10

        if posicion_y < 30:
            posicion_y = y1 + 30

        cv2.putText(
            frame,
            texto,
            (x1, posicion_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )

    return frame
