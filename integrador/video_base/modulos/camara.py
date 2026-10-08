import cv2
from configuracion.config import CAMARA


def iniciar_camara():

    camara = cv2.VideoCapture(CAMARA)

    if not camara.isOpened():
        print("Error: No se pudo abrir la cámara")
        return None

    return camara


def obtener_frame(camara):

    correcto, frame = camara.read()

    if correcto:
        return frame

    return None


def cerrar_camara(camara):

    camara.release()
    cv2.destroyAllWindows()