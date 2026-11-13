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

# Bordes y reconocimiento de patrones (S06)
SOBEL = {"ksize": 3, "umbral_relativo": 0.2, "tolerancia_px": 1}  # borde = magnitud > 20 % de la máxima; tolerancia al comparar con Canny
ORB_EMPAREJAR = {"puntos": 1000, "borde": 15, "ratio": 0.75, "min_inliers": 8, "ransac_px": 5.0,
                 "area_relativa": (0.25, 4.0)}  # área hallada / área de la plantilla: fuera de este rango se descarta  # borde: franja sin detección (15 en vez de 31 para recortes chicos); ratio de Lowe; homografía RANSAC
PLANTILLA_UMBRAL = 0.6  # correlación normalizada mínima para aceptar matchTemplate

# Rendimiento y tiempo real (S15)
FPS_TIEMPO_REAL = 15.0                       # umbral usado para decir "tiempo real" (video de aula, no deportes)
CONFIGS_RENDIMIENTO = [                      # (nombre, ancho de proceso, ¿segmentar y extraer características?)
    {"nombre": "completo_960", "ancho": 960, "analisis": True},
    {"nombre": "completo_640", "ancho": 640, "analisis": True},
    {"nombre": "solo_mejora_640", "ancho": 640, "analisis": False},
    {"nombre": "solo_mejora_480", "ancho": 480, "analisis": False},
]

# Análisis de sesgo por condición (S15, RNF-09)
SESGO = {
    "tamano_cortes": (0.01, 0.05),           # fracción del área del frame que ocupa la caja: <1 % pequeño, <5 % mediano, resto grande
    "region_central": (0.25, 0.75),          # el centro de la caja dentro de este rango en x e y = "centro"; si no, "borde"
    "min_por_grupo": 10,                     # grupos con menos referencias no entran a la brecha (ruido estadístico)
}

# MLP con Keras (S12)
MLP = {"capas": (128, 64), "dropout": 0.0, "l2": 0.0, "tasa_aprendizaje": 1e-3,
       "epocas": 30, "lote": 64, "validacion": 0.2, "paciencia": 4, "semilla": 7}
FASHION_REDUCIDO = {"entrenamiento": 10000, "prueba": 2000, "semilla": 7}   # el sílabo pide un dataset reducido
RECORTES = {"lado": 32, "cada_n": 10, "min_lado_px": 24, "clases": ("persona", "mochila"), "fraccion_prueba": 0.3}

# CNN y transfer learning con Keras (S13)
CNN = {"filtros": (32, 64), "denso": 64, "dropout": 0.0, "l2": 0.0, "tasa_aprendizaje": 1e-3, "semilla": 7,
       "aumento": {"rotacion": 0.07, "traslacion": 0.1, "zoom": 0.1}}   # fracciones: 0.07 de vuelta = ~25 grados
TRANSFER = {"base": "MobileNetV2", "lado": 96, "pesos": "imagenet", "tasa_cabeza": 1e-3,
            "tasa_afinado": 1e-5, "capas_afinadas": 30, "epocas_afinado": 10}
ROBUSTEZ_GRADOS = (0.0, 10.0, 25.0)  # rotaciones (S04) con que se prueba el modelo ya entrenado
