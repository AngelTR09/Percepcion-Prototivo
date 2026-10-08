"""Secuencia completa VIDEO -> IMAGEN en un solo bucle, con salida de video.

  python -m imagen.ejecutar_integrado --fuente datos/padang.webm --salida salidas/integrado
  python -m imagen.ejecutar_integrado --fuente 0 --mostrar        # cámara en vivo (q para salir)

VIDEO entrega el frame (cámara o archivo) y su detector; IMAGEN lo mejora, segmenta y extrae
características. Salidas: video_integrado.mp4 (frame mejorado + cajas de VIDEO), imagen.jsonl, imagen.csv
y resumen_integrado.json con los FPS medidos de punta a punta.
"""
import argparse
import json
import os
import time

import cv2

from . import adaptador_video, config, exportar, pipeline


def main():
    p = argparse.ArgumentParser(description="VIDEO -> IMAGEN con salida de video")
    p.add_argument("--fuente", required=True, help="ruta de un video o índice de cámara (0, 1...)")
    p.add_argument("--salida", default="salidas/integrado")
    p.add_argument("--max-frames", type=int, default=0, help="0 = hasta que termine la fuente")
    p.add_argument("--mostrar", action="store_true", help="muestra la ventana (q para salir)")
    p.add_argument("--degradar", action="store_true", help="simula poca luz y ruido antes de mejorar")
    a = p.parse_args()

    os.makedirs(a.salida, exist_ok=True)
    detectar = adaptador_video.cargar_detector()
    dibujar = adaptador_video.cargar_dibujante()
    registros, omitidos, escritor = [], 0, None
    inicio = time.perf_counter()

    for entrada in adaptador_video.leer_fuente(a.fuente):
        if a.degradar and entrada["frame"] is not None:
            from . import mejoramiento
            d = config.DEGRADACION
            entrada["frame"] = mejoramiento.degradar(entrada["frame"], d["factor_luz"], d["sigma_ruido"], d["semilla"])
        try:
            registro, mejorado, detecciones = pipeline.procesar_frame_produccion(entrada, detectar)
        except Exception as e:  # RNF-05: un frame malo no detiene la secuencia
            omitidos += 1
            print(f"frame {entrada['frame_id']} omitido: {e}")
            continue
        registros.append(registro)
        anotado = dibujar(mejorado, detecciones)
        if escritor is None:
            alto, ancho = anotado.shape[:2]
            fps_video = entrada["fps"] if entrada["fps"] > 0 else 20.0
            escritor = cv2.VideoWriter(os.path.join(a.salida, "video_integrado.mp4"),
                                       cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (ancho, alto))
        escritor.write(anotado)
        if a.mostrar:
            cv2.imshow("VIDEO -> IMAGEN", anotado)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if a.max_frames and len(registros) >= a.max_frames:
            break

    total_s = time.perf_counter() - inicio
    if escritor:
        escritor.release()
    cv2.destroyAllWindows()
    exportar.escribir_jsonl(registros, os.path.join(a.salida, "imagen.jsonl"))
    exportar.escribir_csv(registros, os.path.join(a.salida, "imagen.csv"))
    n = max(len(registros), 1)
    resumen = {
        "fuente": str(a.fuente),
        "frames_procesados": len(registros),
        "frames_omitidos": omitidos,
        "fps_punta_a_punta": round(len(registros) / total_s, 1) if total_s else 0.0,
        "ms_por_frame_imagen": round(sum(r["tiempo_ms"] for r in registros) / n, 1),
        "detecciones_video": sum(r["deteccion_video"]["n"] for r in registros),
    }
    with open(os.path.join(a.salida, "resumen_integrado.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
