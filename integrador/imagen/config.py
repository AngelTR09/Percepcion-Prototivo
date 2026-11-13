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
PIXELES_RECORTE_PEQUENO = 300_000   # recortes y capturas chicas usan NLM (más fuerte, aún rápido)
NLM_FUERZA = 12

# Transformaciones geométricas (S04): se usan como aumento de datos para medir la robustez del detector de VIDEO
TRANSFORMACIONES = [
    {"nombre": "rotar_+10", "tipo": "rotar", "grados": 10.0},
    {"nombre": "rotar_-10", "tipo": "rotar", "grados": -10.0},
    {"nombre": "rotar_+25", "tipo": "rotar", "grados": 25.0},
    {"nombre": "escalar_0.7", "tipo": "escalar", "factor": 0.7},
    {"nombre": "escalar_1.3", "tipo": "escalar", "factor": 1.3},
    {"nombre": "trasladar_10%", "tipo": "trasladar", "dx": 0.10, "dy": 0.05},  # fracción del ancho y del alto
    {"nombre": "voltear_horizontal", "tipo": "voltear"},
]
VISIBLE_MINIMO = 0.5   # un objeto cuenta como referencia si queda al menos esta fracción de su caja dentro del encuadre

# Segmentación (S05): alternativas a Otsu que se comparan por IoU de máscaras
SEGMENTACION = {
    "adaptativo": {"bloque": 101, "c": 3},     # umbral local: la ventana debe ser bastante mayor que el objeto (elegido sobre padang.webm)
    "watershed": {"fraccion_distancia": 0.6},  # semillas = píxeles a >= 60 % de la distancia máxima de su propia componente
}
