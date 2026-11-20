"""Verificacion de un objeto senalado por VIDEO usando los temas del curso de IMAGEN.

VIDEO detecta (clase + confianza + caja). IMAGEN recibe el recorte de ese objeto y lo analiza con las
tecnicas vistas en el curso, mostrando el resultado de cada una:

  S02  representacion: color y escala de grises
  S03  preprocesamiento: filtros media, gaussiano, mediana y bilateral (ruido, nitidez, tiempo)
  S04  mejoramiento: gamma, ecualizacion, CLAHE, histogramas; transformaciones geometricas
  S05  segmentacion: Otsu + morfologia, umbral adaptativo, watershed, contornos
  S06  caracteristicas: bordes Canny y Sobel, esquinas Harris

Con esas evidencias y una nueva pasada del detector de VIDEO sobre el recorte mejorado, IMAGEN emite un
veredicto. Es evidencia para revision humana, no una prueba: la compatibilidad de forma usa rangos
aproximados por clase.
"""
import base64

import cv2
import numpy as np

import ensamble  # noqa: F401  (prepara sys.path)
from imagen import caracteristicas, config, mejoramiento, metricas, preprocesamiento, segmentacion, transformaciones

from . import servicio

ANCHO_MAX = 360          # el recorte se reduce a este ancho para procesarlo y mostrarlo
ANCHO_MIN = 160          # y se amplia si es muy pequeno, para que los bordes sean visibles
UMBRAL_REDETECCION = 0.35
AREA_MINIMA_RELATIVA = 0.08   # el contorno principal debe cubrir al menos esta fraccion del recorte
PASAN_A_PREDICCION = {"verificado", "probable"}   # "no_verificado" se descarta como falso positivo
EQUIVALENTES = {"cuaderno": {"cuaderno", "libro"}, "libro": {"cuaderno", "libro"}}

# Rangos aproximados de forma por clase (lado largo / lado corto del rectangulo minimo y rectangularidad).
# Son heuristicas: un objeto inclinado o parcialmente tapado puede salirse del rango.
FORMA = {
    "telefono":  {"aspecto": (1.3, 2.8), "rectangularidad": 0.70, "texto": "rectangular y alargado"},
    "cuaderno":  {"aspecto": (1.0, 1.9), "rectangularidad": 0.70, "texto": "rectangular"},
    "libro":     {"aspecto": (1.0, 1.9), "rectangularidad": 0.70, "texto": "rectangular"},
    "laptop":    {"aspecto": (1.1, 2.2), "rectangularidad": 0.65, "texto": "rectangular"},
    "reloj":     {"aspecto": (1.0, 1.7), "rectangularidad": 0.55, "texto": "compacto"},
    "audifonos": {"aspecto": (1.0, 2.2), "rectangularidad": 0.0, "solidez_max": 0.90, "texto": "con huecos (arco o auricular)"},
    "mochila":   {"aspecto": (1.0, 2.0), "rectangularidad": 0.55, "texto": "voluminoso"},
}


def _png(imagen):
    ok, buf = cv2.imencode(".png", imagen)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def _jpg(imagen):
    ok, buf = cv2.imencode(".jpg", imagen, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def _item(titulo, nota, imagen):
    codificar = _png if imagen.ndim == 2 else _jpg
    return {"titulo": titulo, "nota": nota, "imagen": codificar(imagen)}


def _ajustar_tamano(recorte):
    alto, ancho = recorte.shape[:2]
    if ancho > ANCHO_MAX:
        return preprocesamiento.normalizar_tamano(recorte, ANCHO_MAX)
    if ancho < ANCHO_MIN:
        factor = ANCHO_MIN / ancho
        return cv2.resize(recorte, (ANCHO_MIN, max(1, round(alto * factor))), interpolation=cv2.INTER_CUBIC)
    return recorte


def _kernel(gris):
    return max(3, (min(gris.shape) // 40) | 1)


def _orientar(mascara):
    """El borde del recorte es mayormente fondo: si la mascara lo marca como objeto, se invierte."""
    anillo = np.ones_like(mascara, bool)
    anillo[2:-2, 2:-2] = False
    return cv2.bitwise_not(mascara) if mascara[anillo].mean() > 127 else mascara


def _geometria(mascara, original):
    """Contorno principal y sus medidas. Devuelve (dibujo, medidas | None)."""
    dibujo = original.copy()
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return dibujo, None
    principal = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(principal)
    if area < AREA_MINIMA_RELATIVA * mascara.size:
        return dibujo, None
    perimetro = cv2.arcLength(principal, True)
    aprox = cv2.approxPolyDP(principal, 0.02 * perimetro, True)
    (cx, cy), (rw, rh), angulo = cv2.minAreaRect(principal)
    lado_largo, lado_corto = max(rw, rh), max(1.0, min(rw, rh))
    casco = cv2.contourArea(cv2.convexHull(principal))
    caja = cv2.boxPoints(((cx, cy), (rw, rh), angulo)).astype(np.int32)
    cv2.drawContours(dibujo, [principal], -1, (0, 255, 0), 2)
    cv2.polylines(dibujo, [caja], True, (0, 140, 255), 2)
    cv2.polylines(dibujo, [aprox], True, (255, 80, 80), 1)
    return dibujo, {
        "cobertura": round(float(area / mascara.size), 3),
        "vertices": int(len(aprox)),
        "aspecto": round(float(lado_largo / lado_corto), 2),
        "rectangularidad": round(float(area / max(1.0, rw * rh)), 3),
        "solidez": round(float(area / max(1.0, casco)), 3),
    }


def _evaluar_forma(clase, medidas):
    """(compatible | None, motivo). None = sin contorno o clase sin regla."""
    regla = FORMA.get(clase)
    if medidas is None:
        return None, "No se obtuvo un contorno claro del objeto en el recorte."
    if regla is None:
        return None, f"No hay regla de forma para la clase {clase}."
    bajo, alto = regla["aspecto"]
    fallos = []
    if not (bajo <= medidas["aspecto"] <= alto):
        fallos.append(f"proporción {medidas['aspecto']} fuera de {bajo}-{alto}")
    if medidas["rectangularidad"] < regla["rectangularidad"]:
        fallos.append(f"rectangularidad {medidas['rectangularidad']} < {regla['rectangularidad']}")
    if "solidez_max" in regla and medidas["solidez"] > regla["solidez_max"]:
        fallos.append(f"solidez {medidas['solidez']} > {regla['solidez_max']} (sin huecos)")
    if fallos:
        return False, "Forma poco compatible: " + "; ".join(fallos) + "."
    return True, f"Forma {regla['texto']}: proporción {medidas['aspecto']}, rectangularidad {medidas['rectangularidad']}."


def _histograma(*series):
    """Histograma de grises superpuesto: [(gris, BGR color, etiqueta)]. Devuelve una imagen 256x160."""
    lienzo = np.full((160, 256, 3), 24, np.uint8)
    for gris, color, _ in series:
        h = cv2.calcHist([gris], [0], None, [256], [0, 256]).ravel()
        h = h / max(1.0, h.max()) * 150
        puntos = np.array([[x, 159 - int(v)] for x, v in enumerate(h)], np.int32)
        cv2.polylines(lienzo, [puntos], False, color, 1, cv2.LINE_AA)
    return lienzo


def _calidad(imagen):
    c = metricas.calidad_sin_referencia(imagen)
    return f"brillo {c['brillo']} · contraste {c['contraste']} · nitidez {c['nitidez']} · ruido {c['ruido']}"


def _s02_s03(original, gris):
    items = [
        _item("Original (BGR)", _calidad(original), original),
        _item("Escala de grises", "Luminancia 0-255: base de bordes y umbral", gris),
    ]
    for nombre in ("media", "gaussiano", "mediana", "bilateral"):
        imagen, ms = metricas.cronometrar(preprocesamiento.aplicar_filtro, original, nombre, config.FILTROS[nombre])
        items.append(_item(f"Filtro {nombre}", f"{_calidad(imagen)} · {round(ms, 2)} ms", imagen))
    return items


def _s04(original):
    gamma_img, gamma = mejoramiento.correccion_gamma(original, config.BRILLO_OBJETIVO)
    ecualizada = mejoramiento.ecualizar_histograma(original)
    clahe = mejoramiento.clahe(original, config.CLAHE["clip"], config.CLAHE["grilla"])
    g0, g1 = preprocesamiento.a_grises(original), preprocesamiento.a_grises(clahe)
    items = [
        _item("Corrección gamma", f"γ = {round(float(gamma), 2)} · {_calidad(gamma_img)}", gamma_img),
        _item("Ecualización de histograma", _calidad(ecualizada), ecualizada),
        _item("CLAHE", _calidad(clahe), clahe),
        _item("Histograma: antes y después", "Gris: original · Naranja: CLAHE", _histograma((g0, (200, 200, 200), "antes"), (g1, (0, 140, 255), "despues"))),
    ]
    for t in (
        ("Rotación +15°", lambda f: transformaciones.rotar(f, 15.0)),
        ("Escalado 0.7", lambda f: transformaciones.escalar(f, 0.7)),
        ("Traslación 10 % / 5 %", lambda f: transformaciones.trasladar(f, 0.10 * f.shape[1], 0.05 * f.shape[0])),
    ):
        items.append(_item(t[0], "warpAffine: la imagen se transforma y se recorta", t[1](original)[0]))
    return items


def _s05(mejorada, gris):
    kernel = _kernel(gris)
    otsu = _orientar(segmentacion.mascara_otsu(gris, kernel))
    adaptativa = _orientar(segmentacion.mascara_adaptativa(gris, 31, 3, kernel))
    try:
        watershed, etiquetas = segmentacion.segmentar_watershed(mejorada, kernel, config.SEGMENTACION["watershed"]["fraccion_distancia"])
        regiones = len([i for i in np.unique(etiquetas) if i > 0])
        coloreada = np.zeros_like(mejorada)
        rng = np.random.default_rng(3)
        for i in np.unique(etiquetas):
            if i > 0:
                coloreada[etiquetas == i] = rng.integers(60, 255, 3)
    except Exception:
        coloreada, regiones = np.zeros_like(mejorada), 0
    iou = metricas.iou_mascaras(otsu, adaptativa)
    return otsu, [
        _item("Otsu + morfología", f"Blanco y negro · objeto en blanco ({round(float(np.count_nonzero(otsu)) / otsu.size * 100, 1)} % del recorte)", otsu),
        _item("Umbral adaptativo", f"Umbral local · IoU con Otsu {round(iou, 2)}", adaptativa),
        _item("Watershed (regiones)", f"{regiones} región(es) separadas por distancia", coloreada),
    ]


def _s06(gris, original):
    canny = cv2.Canny(gris, *config.CANNY)
    sobel = caracteristicas.bordes_sobel(gris, config.SOBEL["ksize"], config.SOBEL["umbral_relativo"])
    bordes = caracteristicas.comparar_bordes(gris, config.CANNY, config.SOBEL)
    resp = cv2.cornerHarris(np.float32(gris), config.HARRIS["bloque"], config.HARRIS["apertura"], config.HARRIS["k"])
    esquinas = original.copy()
    ys, xs = np.nonzero(resp > config.HARRIS["umbral"] * resp.max())
    for x, y in zip(xs[::3], ys[::3]):
        cv2.circle(esquinas, (int(x), int(y)), 1, (0, 0, 255), -1)
    return canny, sobel, bordes, [
        _item("Bordes Canny", f"Densidad {bordes['densidad_canny']} · {bordes['ms_canny']} ms", canny),
        _item("Bordes Sobel", f"Densidad {bordes['densidad_sobel']} · Canny cubierto por Sobel {bordes['canny_cubierto_por_sobel']}", sobel),
        _item("Esquinas Harris", f"{int(np.count_nonzero(resp > config.HARRIS['umbral'] * resp.max()))} píxeles de esquina", esquinas),
    ]


def _entregar_a_prediccion(forma, clase, confianza_video, redeteccion, mejorada):
    """Arma el contrato VIDEO + IMAGEN y se lo entrega al motor de PREDICCION (el mismo de la cadena completa)."""
    try:
        from backend.modulos.prediccion.integracion import integrar_fuentes
        from . import puente
        alto, ancho = forma[:2]
        det_video = [{"clase": clase, "confianza": float(confianza_video), "bbox": [0, 0, ancho, alto]}]
        payload = puente.construir_payload(0, 0.0, metricas.contraste(mejorada), det_video, list(redeteccion), [])
        return integrar_fuentes(payload)
    except Exception as error:  # PREDICCION no debe impedir mostrar el analisis de IMAGEN
        return {"error": f"{type(error).__name__}: {error}"}


def verificar_recorte(jpeg_bytes, clase, confianza_video):
    """Analiza con los temas del curso el recorte de un objeto que VIDEO senalo. Devuelve un dict serializable."""
    crudo = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if crudo is None or crudo.size == 0:
        raise servicio.ImagenInvalida("No se pudo decodificar el recorte")
    original = _ajustar_tamano(crudo)

    # Mejora igual que el modo produccion de IMAGEN: gamma -> filtro fijo -> CLAHE
    iluminado, gamma = mejoramiento.correccion_gamma(original, config.BRILLO_OBJETIVO)
    filtrado = preprocesamiento.aplicar_filtro(iluminado, "gaussiano", config.FILTROS["gaussiano"])
    mejorada = mejoramiento.clahe(filtrado, config.CLAHE["clip"], config.CLAHE["grilla"])
    gris = preprocesamiento.a_grises(mejorada)

    otsu, items_s05 = _s05(mejorada, gris)
    dibujo, medidas = _geometria(otsu, original)
    items_s05.append(_item("Contornos y forma", "Verde: contorno · naranja: rectángulo mínimo · azul: polígono", dibujo))
    canny, sobel, bordes, items_s06 = _s06(gris, original)
    forma_ok, motivo_forma = _evaluar_forma(clase, medidas)

    # Nueva pasada del detector de VIDEO sobre el recorte ya mejorado (como en la cadena de produccion)
    detector = servicio.obtener_detector()
    with servicio._candado_vivo:
        redeteccion = detector(mejorada)
    validas = EQUIVALENTES.get(clase, {clase})
    misma = [o for o in redeteccion if o["clase"] in validas]
    mejor = max(misma, key=lambda o: o["confianza"], default=None)
    redetectado = bool(mejor and mejor["confianza"] >= UMBRAL_REDETECCION)
    if redetectado:
        motivo_det = f"VIDEO vuelve a ver {mejor['clase']} en el recorte mejorado ({round(mejor['confianza'] * 100)} %)."
    else:
        otras = sorted({o["clase"] for o in redeteccion})
        motivo_det = ("VIDEO no confirma la clase en el recorte mejorado" + (f" (ve: {', '.join(otras)})." if otras else "."))

    if redetectado and forma_ok:
        veredicto = "verificado"
    elif redetectado or forma_ok:
        veredicto = "probable"
    else:
        veredicto = "no_verificado"

    # IMAGEN tiene la ultima palabra: lo que no verifica es un falso positivo y no llega a PREDICCION.
    pasa = veredicto in PASAN_A_PREDICCION
    prediccion = _entregar_a_prediccion(original.shape, clase, confianza_video, redeteccion, mejorada) if pasa else None

    return {
        "clase": clase,
        "confianza_video": round(float(confianza_video), 3),
        "veredicto": veredicto,
        "pasa_a_prediccion": pasa,
        "decision": ("IMAGEN confirma el objeto: el hallazgo pasa a PREDICCIÓN." if pasa else
                     "IMAGEN no logra verificar el objeto: se descarta como falso positivo y no llega a PREDICCIÓN."),
        "prediccion": prediccion,
        "motivos": [motivo_det, motivo_forma],
        "redeteccion": {"confirmada": redetectado, "confianza": round(mejor["confianza"], 3) if mejor else None},
        "forma": medidas,
        "bordes": bordes,
        "tamano_recorte": [int(original.shape[1]), int(original.shape[0])],
        "secciones": [
            {"id": "s03", "semana": "S03", "titulo": "Preprocesamiento",
             "descripcion": "El recorte en escala de grises y cuatro filtros de ruido comparados sin imagen de referencia.",
             "items": _s02_s03(original, gris)},
            {"id": "s04", "semana": "S04", "titulo": "Mejoramiento y transformaciones",
             "descripcion": "Corrección de iluminación, ecualización de histograma y transformaciones geométricas del recorte.",
             "items": _s04(original)},
            {"id": "s05", "semana": "S05", "titulo": "Segmentación",
             "descripcion": "Umbralización (blanco y negro), segmentación por regiones, operaciones morfológicas y contornos.",
             "items": items_s05},
            {"id": "s06", "semana": "S06", "titulo": "Bordes y características",
             "descripcion": "Operadores de borde Sobel y Canny, esquinas de Harris.",
             "items": items_s06},
        ],
    }
