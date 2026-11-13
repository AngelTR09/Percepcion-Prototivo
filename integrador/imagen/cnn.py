"""S13: red neuronal convolucional (convolución + pooling) y transfer learning con Keras.

Mismo entorno opcional que `mlp.py` (`requirements-lab.txt`). El entrenamiento y la evaluación se reutilizan de `mlp.py`
(`entrenar`, `evaluar`, `matriz_confusion`): una CNN es un modelo Keras más.
"""
import numpy as np

from . import config, mlp


def _keras():
    return mlp._keras()


def construir_cnn(forma_entrada, n_clases, filtros=None, denso=None, dropout=None, l2=None, tasa=None,
                  aumento=False, semilla=None):
    """[aumento] -> (Conv3x3 relu -> MaxPool2x2) x N -> Flatten -> Dense relu -> Dense softmax.
    `forma_entrada` = (alto, ancho, canales). Con `aumento=True` se añaden rotación, traslación y zoom aleatorios
    (equivalentes a las transformaciones geométricas de S04) que solo actúan al entrenar."""
    p = config.CNN
    filtros = p["filtros"] if filtros is None else filtros
    denso = p["denso"] if denso is None else denso
    dropout = p["dropout"] if dropout is None else dropout
    l2 = p["l2"] if l2 is None else l2
    tasa = p["tasa_aprendizaje"] if tasa is None else tasa
    k = _keras()
    k.utils.set_random_seed(p["semilla"] if semilla is None else semilla)
    reg = k.regularizers.l2(l2) if l2 else None
    modelo = k.Sequential([k.layers.Input(shape=tuple(forma_entrada))])
    if aumento:
        a = p["aumento"]
        modelo.add(k.layers.RandomRotation(a["rotacion"]))
        modelo.add(k.layers.RandomTranslation(a["traslacion"], a["traslacion"]))
        modelo.add(k.layers.RandomZoom(a["zoom"]))
    for f in filtros:
        modelo.add(k.layers.Conv2D(f, 3, padding="same", activation="relu", kernel_regularizer=reg))
        modelo.add(k.layers.MaxPooling2D(2))
    modelo.add(k.layers.Flatten())
    modelo.add(k.layers.Dense(denso, activation="relu", kernel_regularizer=reg))
    if dropout:
        modelo.add(k.layers.Dropout(dropout))
    modelo.add(k.layers.Dense(n_clases, activation="softmax"))
    modelo.compile(optimizer=k.optimizers.Adam(tasa), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


def construir_transfer(n_clases, lado=None, pesos="config", tasa=None, semilla=None):
    """Transfer learning: MobileNetV2 (sin la capa final) congelada + cabeza nueva (GlobalAveragePooling -> Dropout -> Dense).
    Entrada: imágenes RGB `lado x lado` con valores en [0, 1]; el modelo las lleva al rango [-1, 1] que espera MobileNetV2.
    `pesos`: "config" usa `config.TRANSFER["pesos"]` ("imagenet" descarga los preentrenados); `None` inicializa al azar (pruebas sin red).
    Los colores deben venir en RGB (OpenCV entrega BGR: invertir el último eje antes de entrenar)."""
    t = config.TRANSFER
    lado = t["lado"] if lado is None else lado
    pesos = t["pesos"] if pesos == "config" else pesos
    k = _keras()
    k.utils.set_random_seed(config.CNN["semilla"] if semilla is None else semilla)
    base = k.applications.MobileNetV2(input_shape=(lado, lado, 3), include_top=False, weights=pesos)
    base.trainable = False
    entrada = k.Input(shape=(lado, lado, 3))
    x = k.layers.Rescaling(2.0, offset=-1.0)(entrada)            # [0, 1] -> [-1, 1]
    x = base(x, training=False)                                   # BatchNorm de la base en modo inferencia
    x = k.layers.GlobalAveragePooling2D()(x)
    x = k.layers.Dropout(0.2)(x)
    salida = k.layers.Dense(n_clases, activation="softmax")(x)
    modelo = k.Model(entrada, salida)
    modelo.compile(optimizer=k.optimizers.Adam(t["tasa_cabeza"] if tasa is None else tasa),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


def afinar(modelo, n_capas=None, tasa=None):
    """Fine-tuning: descongela las últimas `n_capas` de la base (BatchNorm sigue congelada) y recompila con una tasa muy baja."""
    t = config.TRANSFER
    n_capas = t["capas_afinadas"] if n_capas is None else n_capas
    k = _keras()
    base = next(c for c in modelo.layers if isinstance(c, k.Model))
    base.trainable = True
    for capa in base.layers[:-n_capas] if n_capas > 0 else base.layers:
        capa.trainable = False
    for capa in base.layers:
        if isinstance(capa, k.layers.BatchNormalization):
            capa.trainable = False
    modelo.compile(optimizer=k.optimizers.Adam(t["tasa_afinado"] if tasa is None else tasa),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


def parametros(modelo):
    """(entrenables, no entrenables) tras la última compilación."""
    entrenables = int(sum(int(np.prod(w.shape)) for w in modelo.trainable_weights))
    return entrenables, int(modelo.count_params()) - entrenables


def rotar_lote(x, grados):
    """Rota cada imagen del lote (N, alto, ancho, canales) con la transformación de S04. Con 0 grados no cambia nada."""
    if grados == 0:
        return x
    from . import transformaciones
    return np.stack([transformaciones.rotar((i * 255).astype(np.uint8), grados)[0] for i in x]).astype(np.float32) / 255.0
