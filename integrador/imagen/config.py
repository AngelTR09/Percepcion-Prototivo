"""Parámetros del módulo IMAGEN (RNF-06: nada va "quemado" en el código)."""

VERSION_CONTRATO = "0.1"

# Normalización de tamaño (RF-05)
ANCHO_PROCESO = 960

# Degradación sintética para la prueba (simula poca luz + ruido del sensor)
DEGRADACION = {"factor_luz": 0.35, "sigma_ruido": 18.0, "semilla": 7}

# Filtros de ruido candidatos (RF-03); kernel impar
FILTROS = {
    "media": {"k": 5},
    "gaussiano": {"k": 5, "sigma": 1.2},
    "mediana": {"k": 5},
    "bilateral": {"d": 7, "sigma_color": 50, "sigma_espacio": 50},
}

# Mejora de contraste e iluminación (RF-04)
CLAHE = {"clip": 2.0, "grilla": 8}
BRILLO_OBJETIVO = 0.45  # media de luminancia a la que apunta la corrección gamma

# Segmentación (RF-06, RF-07)
AREA_MINIMA = 800       # px² a la resolución de proceso
MAX_OBJETOS = 30
KERNEL_MORFOLOGIA = 5

# Características (RF-08)
CANNY = (80, 160)
HARRIS = {"bloque": 2, "apertura": 3, "k": 0.04, "umbral": 0.01}
ORB_PUNTOS = 500

# Comparación con las detecciones de VIDEO (RF-09)
IOU_MINIMO = 0.5

# Mejora adaptativa de las salidas de VIDEO (sin imagen de referencia)
UMBRAL_BRILLO = 0.40      # luminancia media (0-1) por debajo de la cual se aplica gamma
UMBRAL_CONTRASTE = 55.0   # desviación estándar de luminancia por debajo de la cual se aplica CLAHE
UMBRAL_RUIDO = 3.0        # sigma estimado (Immerkaer) por encima del cual se filtra
FILTRO_SALIDAS = "gaussiano"
EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png", ".bmp")
EXTENSIONES_VIDEO = (".mp4", ".avi", ".mov", ".webm", ".mkv")
