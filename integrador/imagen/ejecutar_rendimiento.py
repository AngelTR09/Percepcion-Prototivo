"""S15: FPS reales por configuración y análisis de sesgo por condición.

  python -m imagen.ejecutar_rendimiento --video datos/padang.webm --salida salidas/rendimiento --cada-n 30
  python -m imagen.ejecutar_rendimiento --video datos/padang.webm --salida salidas/rendimiento --sesgo --luz 0.15

Rendimiento: ms por etapa y FPS de punta a punta para cada configuración de `config.CONFIGS_RENDIMIENTO`, y cuántas
de las detecciones a 960 px se conservan al achicar el frame (las cajas se reescalan a 960 para comparar).
Sesgo (`--sesgo`): recuperación de detecciones bajo poca luz, sin y con mejora, por clase, tamaño y posición.
"""
import argparse
import csv
import os

from . import adaptador_video, config, mejoramiento, metricas, pipeline, preprocesamiento, rendimiento, sesgo


def _a_960(detecciones, forma, ancho_ref):
    """Reescala cajas desde el tamaño procesado al de referencia, para poder compararlas."""
    k = ancho_ref / float(forma[1])
    return [{**d, "bbox": [v * k for v in d["bbox"]]} for d in detecciones]


def correr_rendimiento(a, detector):
    mediciones = {c["nombre"]: [] for c in config.CONFIGS_RENDIMIENTO}
    conservadas = {c["nombre"]: [0, 0] for c in config.CONFIGS_RENDIMIENTO}
    n = 0
    for entrada in adaptador_video.leer_frames(a.video, a.cada_n):
        try:
            referencia = detector(preprocesamiento.normalizar_tamano(entrada["frame"], config.ANCHO_PROCESO))
            for c in config.CONFIGS_RENDIMIENTO:
                ms, dets, forma = rendimiento.medir_etapas(entrada["frame"], detector, c["ancho"], c["analisis"])
                mediciones[c["nombre"]].append(ms)
                esc = _a_960(dets, forma, config.ANCHO_PROCESO)
                conservadas[c["nombre"]][0] += metricas.recuperadas(referencia, esc, config.IOU_MINIMO)
                conservadas[c["nombre"]][1] += len(referencia)
            n += 1
        except Exception as e:  # RNF-05
            print(f"frame {entrada['frame_id']} omitido: {e}")
    ruta = os.path.join(a.salida, "tabla_rendimiento.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["configuracion", "ms_total", "fps", "tiempo_real", *[f"ms_{e}" for e in rendimiento.ETAPAS],
                    "detecciones_conservadas"])
        for c in config.CONFIGS_RENDIMIENTO:
            r = rendimiento.resumir(mediciones[c["nombre"]])
            ok, tot = conservadas[c["nombre"]]
            w.writerow([c["nombre"], r["ms_total"], r["fps"], rendimiento.es_tiempo_real(r["fps"]),
                        *[r["ms_por_etapa"][e] for e in rendimiento.ETAPAS], round(ok / tot, 3) if tot else ""])
            print(f"{c['nombre']:17s} {r['ms_total']:7.1f} ms  {r['fps']:5.1f} FPS  "
                  f"tiempo_real={rendimiento.es_tiempo_real(r['fps'])}  conserva={ok}/{tot}  etapas={r['ms_por_etapa']}")
    print(f"{n} frames -> {ruta}")


def correr_sesgo(a, detector):
    tabla, n = {}, 0
    d = {**config.DEGRADACION, "factor_luz": a.luz}
    for entrada in adaptador_video.leer_frames(a.video, a.cada_n):
        try:
            ref = preprocesamiento.normalizar_tamano(entrada["frame"], config.ANCHO_PROCESO)
            referencia = detector(ref)
            degradado = mejoramiento.degradar(ref, d["factor_luz"], d["sigma_ruido"], d["semilla"] + entrada["frame_id"])
            _, _, det_mejorado = pipeline.procesar_frame_produccion(
                {"frame_id": entrada["frame_id"], "timestamp": entrada["timestamp"], "frame": degradado}, detector)
            sesgo.acumular(tabla, referencia, {"degradado": detector(degradado), "mejorado": det_mejorado},
                           ref.shape, config.IOU_MINIMO)
            n += 1
        except Exception as e:  # RNF-05
            print(f"frame {entrada['frame_id']} omitido: {e}")
    ruta = os.path.join(a.salida, f"tabla_sesgo_luz{a.luz}.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["criterio", "grupo", "referencias", "recuperadas_degradado", "recuperadas_mejorado",
                    "tasa_degradado", "tasa_mejorado"])
        for criterio, grupos in tabla.items():
            for g, fila in sorted(grupos.items()):
                w.writerow([criterio, g, fila["n"], fila["degradado"], fila["mejorado"],
                            round(fila["degradado"] / fila["n"], 3), round(fila["mejorado"] / fila["n"], 3)])
                print(f"{criterio:7s} {g:10s} n={fila['n']:4d}  degradado={fila['degradado'] / fila['n']:.2f}  "
                      f"mejorado={fila['mejorado'] / fila['n']:.2f}")
            print(f"  brecha {criterio}: degradado={sesgo.brecha(grupos, 'degradado')}  mejorado={sesgo.brecha(grupos, 'mejorado')}")
    print(f"{n} frames con luz {a.luz} -> {ruta}")


def main():
    p = argparse.ArgumentParser(description="FPS reales y análisis de sesgo por condición")
    p.add_argument("--video", required=True)
    p.add_argument("--salida", default="salidas/rendimiento")
    p.add_argument("--cada-n", type=int, default=30)
    p.add_argument("--sesgo", action="store_true", help="en vez del rendimiento, mide el sesgo por condición")
    p.add_argument("--luz", type=float, default=0.15)
    a = p.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    detector = adaptador_video.cargar_detector()
    (correr_sesgo if a.sesgo else correr_rendimiento)(a, detector)


if __name__ == "__main__":
    main()
