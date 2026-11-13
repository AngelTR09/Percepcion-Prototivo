"""S12: perceptrón multicapa (MLP) con Keras para clasificar imágenes simples.

Keras/TensorFlow es dependencia opcional (`requirements-lab.txt`): se importa dentro de las funciones, así que el
resto de IMAGEN y el Docker no lo necesitan. La evaluación (matriz de confusión, precisión, recall, F1) está en NumPy
para no exigir scikit-learn en el entorno del laboratorio.
"""
import numpy as np

from . import config


def _keras():
    try:
        import keras
    except ImportError as e:
        raise RuntimeError("falta Keras/TensorFlow: pip install -r requirements-lab.txt") from e
    return keras


def cargar_fashion_mnist(n_entrenamiento, n_prueba, semilla):
    """Fashion-MNIST reducido (el sílabo pide un dataset reducido). Devuelve imágenes float32 en [0, 1] y etiquetas.
    Submuestrea al azar y de forma reproducible; las clases quedan aproximadamente balanceadas."""
    (xe, ye), (xp, yp) = _keras().datasets.fashion_mnist.load_data()
    rng = np.random.default_rng(semilla)
    ie = rng.permutation(len(xe))[:n_entrenamiento]
    ip = rng.permutation(len(xp))[:n_prueba]
    return (xe[ie].astype(np.float32) / 255.0, ye[ie]), (xp[ip].astype(np.float32) / 255.0, yp[ip])


def construir_mlp(forma_entrada, n_clases, capas=None, dropout=None, l2=None, tasa=None, semilla=None):
    """Flatten -> Dense(relu) x N -> Dense(softmax). Parámetros por defecto en `config.MLP`."""
    p = config.MLP
    capas = p["capas"] if capas is None else capas
    dropout = p["dropout"] if dropout is None else dropout
    l2 = p["l2"] if l2 is None else l2
    tasa = p["tasa_aprendizaje"] if tasa is None else tasa
    k = _keras()
    k.utils.set_random_seed(p["semilla"] if semilla is None else semilla)
    regularizador = k.regularizers.l2(l2) if l2 else None
    modelo = k.Sequential([k.layers.Input(shape=tuple(forma_entrada)), k.layers.Flatten()])
    for unidades in capas:
        modelo.add(k.layers.Dense(unidades, activation="relu", kernel_regularizer=regularizador))
        if dropout:
            modelo.add(k.layers.Dropout(dropout))
    modelo.add(k.layers.Dense(n_clases, activation="softmax"))
    modelo.compile(optimizer=k.optimizers.Adam(tasa), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


def entrenar(modelo, x, y, epocas=None, lote=None, validacion=None, paciencia=None, pesos_clase=None, parada_temprana=True):
    """Entrena con una fracción de validación y parada temprana (restaura los mejores pesos).
    Devuelve {loss, val_loss, accuracy, val_accuracy} por época."""
    p = config.MLP
    k = _keras()
    parada = k.callbacks.EarlyStopping(monitor="val_loss", patience=p["paciencia"] if paciencia is None else paciencia,
                                       restore_best_weights=True)
    # parada_temprana=False entrena todas las épocas (sirve para medir qué aporta la parada temprana)
    historia = modelo.fit(x, y, epochs=p["epocas"] if epocas is None else epocas,
                          batch_size=p["lote"] if lote is None else lote,
                          validation_split=p["validacion"] if validacion is None else validacion,
                          callbacks=[parada] if parada_temprana else [], class_weight=pesos_clase, verbose=0, shuffle=True)
    return {k_: [float(v) for v in vs] for k_, vs in historia.history.items()}


def matriz_confusion(y_real, y_pred, n_clases):
    """Filas = clase real, columnas = clase predicha."""
    m = np.zeros((n_clases, n_clases), np.int64)
    np.add.at(m, (np.asarray(y_real, int), np.asarray(y_pred, int)), 1)
    return m


def metricas_clasificacion(matriz):
    """Exactitud y precisión/recall/F1 por clase a partir de la matriz de confusión (0 cuando no hay muestras)."""
    m = np.asarray(matriz, np.float64)
    tp = np.diag(m)
    precision = np.divide(tp, m.sum(axis=0), out=np.zeros_like(tp), where=m.sum(axis=0) > 0)
    recall = np.divide(tp, m.sum(axis=1), out=np.zeros_like(tp), where=m.sum(axis=1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {
        "exactitud": float(tp.sum() / m.sum()) if m.sum() else 0.0,
        "precision": precision.round(4).tolist(), "recall": recall.round(4).tolist(), "f1": f1.round(4).tolist(),
        "soporte": m.sum(axis=1).astype(int).tolist(),
    }


def evaluar(modelo, x, y, n_clases):
    """Predice en `x` y devuelve (métricas, matriz de confusión)."""
    pred = np.argmax(modelo.predict(x, verbose=0), axis=1)
    matriz = matriz_confusion(y, pred, n_clases)
    return metricas_clasificacion(matriz), matriz


def linea_base_mayoritaria(y_entrenamiento, y_prueba):
    """Exactitud de predecir siempre la clase más frecuente del entrenamiento. Un modelo que no la supera no aprendió nada útil."""
    mayoritaria = int(np.bincount(np.asarray(y_entrenamiento, int)).argmax())
    return float(np.mean(np.asarray(y_prueba) == mayoritaria))


def dividir_por_tiempo(grupos, fraccion_prueba):
    """Máscara de prueba por bloques: los últimos `fraccion_prueba` de los grupos (p. ej. frame_id) van a prueba.
    Evita que frames contiguos casi idénticos caigan a ambos lados y inflen la exactitud."""
    grupos = np.asarray(grupos)
    unicos = np.unique(grupos)
    corte = unicos[int(round(len(unicos) * (1 - fraccion_prueba)))] if len(unicos) > 1 else unicos[0]
    prueba = grupos >= corte
    if prueba.all() or not prueba.any():
        raise ValueError("no hay suficientes grupos distintos para dividir")
    return prueba
