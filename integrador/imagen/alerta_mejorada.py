"""Alerta de VIDEO + mejora instantánea de IMAGEN sobre lo que la activó.

Sigue la misma regla de VIDEO (objeto no permitido visible TIEMPO_ALERTA segundos -> alerta, el contador
se reinicia si desaparece), usando SU detector, SU clasificación de objetos no permitidos y SU
guardar_alerta. En el instante de la alerta, IMAGEN mejora el frame y recorta y mejora el objeto.
El contador usa el timestamp del video, así funciona igual con cámara en vivo o con un archivo.

  python -m imagen.alerta_mejorada --fuente datos/examen.mp4 --salida salidas/alertas
  python -m imagen.alerta_mejorada --fuente 0 --mostrar          # cámara en vivo, q para salir
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from . import adaptador_video, config, mejoramiento


def recorte_con_margen(frame, bbox, margen=0.15):
    x1, y1, x2, y2 = bbox
    mx, my = int((x2 - x1) * margen), int((y2 - y1) * margen)
    alto, ancho = frame.shape[:2]
    return frame[max(0, y1 - my):min(alto, y2 + my), max(0, x1 - mx):min(ancho, x2 + mx)]


def ampliar(imagen, lado_minimo=320):
    """Agranda recortes pequeños para que se vean bien en el informe."""
    escala = max(1.0, lado_minimo / max(imagen.shape[:2]))
    return cv2.resize(imagen, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)


def mejorar_alerta(frame, deteccion, ruta_alerta_video, carpeta, timestamp):
    """Genera la mejora de una alerta: frame completo, objeto recortado y registro JSON."""
    inicio = time.perf_counter()
    base = os.path.splitext(os.path.basename(ruta_alerta_video))[0]
    completo, antes, despues, pasos = mejoramiento.mejorar_adaptativo(frame, config)
    recorte = recorte_con_margen(frame, deteccion["bbox"])
    objeto, antes_obj, despues_obj, pasos_obj = mejoramiento.mejorar_adaptativo(recorte, config)
    cv2.imwrite(os.path.join(carpeta, f"{base}_frame_mejorado.jpg"), completo)
    cv2.imwrite(os.path.join(carpeta, f"{base}_objeto_comparacion.jpg"),
                np.hstack([ampliar(recorte), cv2.resize(ampliar(objeto), (ampliar(recorte).shape[1], ampliar(recorte).shape[0]))]))
    registro = {
        "timestamp": round(timestamp, 3), "clase": deteccion["clase"], "confianza": round(deteccion["confianza"], 3),
        "bbox": deteccion["bbox"], "alerta_video": ruta_alerta_video,
        "frame": {"pasos": pasos, "antes": antes, "despues": despues},
        "objeto": {"pasos": pasos_obj, "antes": antes_obj, "despues": despues_obj},
        "ms_mejora": round((time.perf_counter() - inicio) * 1000, 1),
    }
    with open(os.path.join(carpeta, "alertas_mejoradas.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    return registro


def main():
    p = argparse.ArgumentParser(description="Alerta de VIDEO con mejora instantánea de IMAGEN")
    p.add_argument("--fuente", required=True, help="ruta de un video o índice de cámara")
    p.add_argument("--salida", default="salidas/alertas")
    p.add_argument("--tiempo", type=float, default=None, help="segundos para alertar (por defecto el de VIDEO)")
    p.add_argument("--tambien-prohibir", nargs="*", default=[], help="clases extra a tratar como no permitidas (pruebas)")
    p.add_argument("--degradar", action="store_true", help="simula poca luz y ruido para la prueba")
    p.add_argument("--mostrar", action="store_true")
    a = p.parse_args()

    os.makedirs(a.salida, exist_ok=True)
    detectar = adaptador_video.cargar_detector()
    dibujar = adaptador_video.cargar_dibujante()
    es_no_permitido, tiempo_video, guardar_alerta_video = adaptador_video.cargar_alerta()
    umbral = a.tiempo if a.tiempo is not None else tiempo_video
    extra = {c.lower() for c in a.tambien_prohibir}

    inicio_visible, alertada, alertas = {}, {}, []
    for entrada in adaptador_video.leer_fuente(a.fuente):
        frame, ts = entrada["frame"], entrada["timestamp"]
        if a.degradar:
            d = config.DEGRADACION
            frame = mejoramiento.degradar(frame, d["factor_luz"], d["sigma_ruido"], d["semilla"])
        detecciones = detectar(frame)
        presentes = {}
        for det in detecciones:
            if es_no_permitido(det["clase"]) or det["clase"] in extra:
                presentes.setdefault(det["clase"], det)  # una por clase, como el contador de VIDEO

        for clase, det in presentes.items():
            inicio_visible.setdefault(clase, ts)
            if ts - inicio_visible[clase] >= umbral and not alertada.get(clase):
                alertada[clase] = True
                ruta = guardar_alerta_video(frame.copy(), clase)          # la captura de VIDEO, intacta
                reg = mejorar_alerta(frame, det, ruta, a.salida, ts)       # la mejora de IMAGEN
                alertas.append(reg)
                print(f"ALERTA {clase} en t={ts:.1f}s -> mejorada en {reg['ms_mejora']} ms "
                      f"(pasos frame: {'+'.join(reg['frame']['pasos']) or 'ninguno'})")
        for clase in [c for c in inicio_visible if c not in presentes]:   # desapareció: reinicia
            del inicio_visible[clase]
            alertada.pop(clase, None)

        if a.mostrar:
            cv2.imshow("VIDEO + alerta mejorada", dibujar(frame, detecciones))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
    print(f"Alertas generadas: {len(alertas)}")


if __name__ == "__main__":
    main()
