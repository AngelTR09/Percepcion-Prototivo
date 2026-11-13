"""Extracción de características (S06): bordes, esquinas, ORB y color."""
import cv2
import numpy as np


def densidad_bordes(gris, umbrales):
    bordes = cv2.Canny(gris, *umbrales)
    return float(np.count_nonzero(bordes) / bordes.size)


def contar_esquinas(gris, p):
    respuesta = cv2.cornerHarris(np.float32(gris), p["bloque"], p["apertura"], p["k"])
    return int(np.count_nonzero(respuesta > p["umbral"] * respuesta.max()))


def contar_orb(gris, max_puntos):
    puntos = cv2.ORB_create(nfeatures=max_puntos).detect(gris, None)
    return len(puntos)


def media_hsv(frame):
    """Media de H, S y V normalizada a [0, 1]."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
    media = hsv.mean(axis=0) / np.array([179.0, 255.0, 255.0])
    return [round(float(v), 4) for v in media]


def extraer(frame, n_objetos, cfg):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "n_objetos": n_objetos,
        "densidad_bordes": round(densidad_bordes(gris, cfg.CANNY), 5),
        "n_esquinas": contar_esquinas(gris, cfg.HARRIS),
        "n_keypoints_orb": contar_orb(gris, cfg.ORB_PUNTOS),
        "hist_hsv": media_hsv(frame),
    }


# ---- S06 completo: Sobel y reconocimiento de patrones (no cambian `extraer`, que es el contrato con PREDICCIÓN) ----

def _validar(*imagenes):
    for i in imagenes:
        if i is None or getattr(i, "size", 0) == 0:
            raise ValueError("imagen vacía o corrupta")


def magnitud_sobel(gris, ksize):
    """Magnitud del gradiente: sqrt(gx^2 + gy^2) con el operador de Sobel."""
    _validar(gris)
    gx = cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gris, cv2.CV_32F, 0, 1, ksize=ksize)
    return cv2.magnitude(gx, gy)


def bordes_sobel(gris, ksize, umbral_relativo):
    """Mapa binario de bordes (0/255): magnitud de Sobel por encima de `umbral_relativo` x su máximo."""
    magnitud = magnitud_sobel(gris, ksize)
    maximo = float(magnitud.max())
    if maximo <= 0:
        return np.zeros(gris.shape[:2], np.uint8)
    return ((magnitud > umbral_relativo * maximo) * 255).astype(np.uint8)


def _cubierto(a, b, tolerancia):
    """Fracción de los píxeles de borde de `a` que tienen un borde de `b` a <= `tolerancia` píxeles."""
    total = np.count_nonzero(a)
    if total == 0:
        return 1.0 if np.count_nonzero(b) == 0 else 0.0
    nucleo = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * tolerancia + 1, 2 * tolerancia + 1))
    cerca = cv2.dilate(b, nucleo)
    return float(np.count_nonzero((a > 0) & (cerca > 0)) / total)


def comparar_bordes(gris, canny, sobel):
    """Canny vs Sobel sobre la misma imagen. `canny` = (bajo, alto); `sobel` = dict de config.SOBEL.
    Devuelve densidades, ms de cada uno y cuánto del borde de uno está cubierto por el otro (con tolerancia)."""
    from . import metricas
    _validar(gris)
    c, ms_c = metricas.cronometrar(cv2.Canny, gris, *canny)
    s, ms_s = metricas.cronometrar(bordes_sobel, gris, sobel["ksize"], sobel["umbral_relativo"])
    t = sobel["tolerancia_px"]
    return {
        "densidad_canny": round(float(np.count_nonzero(c) / c.size), 5),
        "densidad_sobel": round(float(np.count_nonzero(s) / s.size), 5),
        "canny_cubierto_por_sobel": round(_cubierto(c, s, t), 3),
        "sobel_cubierto_por_canny": round(_cubierto(s, c, t), 3),
        "ms_canny": round(ms_c, 2), "ms_sobel": round(ms_s, 2),
    }


def descriptores_orb(gris, max_puntos, borde=31):
    """Puntos clave y descriptores binarios ORB (a diferencia de `contar_orb`, que solo cuenta).
    `borde` es la franja del contorno donde ORB no detecta (31 por defecto). En recortes pequeños esa franja se come
    casi todo: un parche de 120 px quedaba con 4 puntos; con 15 queda con ~50."""
    _validar(gris)
    return cv2.ORB_create(nfeatures=max_puntos, edgeThreshold=borde, patchSize=borde).detectAndCompute(gris, None)


def emparejar_orb(des_a, des_b, ratio):
    """Correspondencias entre dos conjuntos de descriptores: distancia de Hamming con fuerza bruta y prueba de ratio de Lowe."""
    if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
        return []
    pares = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_a, des_b, k=2)
    return [p[0] for p in pares if len(p) == 2 and p[0].distance < ratio * p[1].distance]


def localizar_con_orb(plantilla, escena, p):
    """Busca `plantilla` (gris) dentro de `escena` (gris) con ORB + homografía RANSAC. `p` = dict de config.ORB_EMPAREJAR.
    Soporta rotación y cambio de escala. Devuelve {bbox [x1,y1,x2,y2], inliers, correspondencias} o None."""
    _validar(plantilla, escena)
    kp_p, des_p = descriptores_orb(plantilla, p["puntos"], p["borde"])
    kp_e, des_e = descriptores_orb(escena, p["puntos"], p["borde"])
    buenas = emparejar_orb(des_p, des_e, p["ratio"])
    if len(buenas) < max(4, p["min_inliers"]):
        return None
    origen = np.float32([kp_p[m.queryIdx].pt for m in buenas]).reshape(-1, 1, 2)
    destino = np.float32([kp_e[m.trainIdx].pt for m in buenas]).reshape(-1, 1, 2)
    H, mascara = cv2.findHomography(origen, destino, cv2.RANSAC, p["ransac_px"])
    if H is None or int(mascara.sum()) < p["min_inliers"]:
        return None
    alto, ancho = plantilla.shape[:2]
    esquinas = np.float32([[0, 0], [ancho, 0], [ancho, alto], [0, alto]]).reshape(-1, 1, 2)
    puntos = cv2.perspectiveTransform(esquinas, H).reshape(-1, 2)
    # RANSAC puede aceptar una homografía degenerada: se exige un cuadrilátero convexo de tamaño plausible
    relativa = abs(cv2.contourArea(puntos.astype(np.float32))) / float(alto * ancho)
    if not cv2.isContourConvex(puntos.astype(np.float32).reshape(-1, 1, 2)) or not (
            p["area_relativa"][0] <= relativa <= p["area_relativa"][1]):
        return None
    x1, y1 = puntos.min(axis=0)
    x2, y2 = puntos.max(axis=0)
    return {"bbox": [float(x1), float(y1), float(x2), float(y2)], "inliers": int(mascara.sum()), "correspondencias": len(buenas)}


def localizar_con_plantilla(plantilla, escena, umbral):
    """matchTemplate (correlación normalizada). Solo encuentra la plantilla con su misma orientación y escala.
    Devuelve {bbox, puntaje} o None si el mejor puntaje no llega a `umbral`."""
    _validar(plantilla, escena)
    if plantilla.shape[0] > escena.shape[0] or plantilla.shape[1] > escena.shape[1]:
        return None
    resultado = cv2.matchTemplate(escena, plantilla, cv2.TM_CCOEFF_NORMED)
    _, mejor, _, (x, y) = cv2.minMaxLoc(resultado)
    if mejor < umbral:
        return None
    return {"bbox": [float(x), float(y), float(x + plantilla.shape[1]), float(y + plantilla.shape[0])], "puntaje": round(float(mejor), 3)}
