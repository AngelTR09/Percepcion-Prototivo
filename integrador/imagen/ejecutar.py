"""Punto de entrada: python -m imagen.ejecutar --video datos/prueba.mp4 --salida salidas/"""
import argparse
import json
import os
import sys

from . import adaptador_video, config, exportar, pipeline


def main():
    p = argparse.ArgumentParser(description="Módulo IMAGEN: VIDEO -> adaptador -> pipeline -> exportar")
    p.add_argument("--video", required=True)
    p.add_argument("--salida", default="salidas/")
    p.add_argument("--cada-n", type=int, default=1, help="procesa 1 de cada N frames")
    p.add_argument("--sin-degradar", action="store_true", help="no simula poca luz/ruido")
    p.add_argument("--luz", type=float, default=None, help="factor de luz de la degradación (menor = más oscuro)")
    p.add_argument("--ruido", type=float, default=None, help="sigma del ruido de la degradación")
    p.add_argument("--evidencias", type=int, default=3, help="cuántos frames guardar como antes/después")
    a = p.parse_args()

    os.makedirs(a.salida, exist_ok=True)
    detector = adaptador_video.cargar_detector()
    degradacion = {**config.DEGRADACION}
    if a.luz is not None:
        degradacion["factor_luz"] = a.luz
    if a.ruido is not None:
        degradacion["sigma_ruido"] = a.ruido
    registros, errores, filtros_acum = [], 0, {}

    for entrada in adaptador_video.leer_frames(a.video, a.cada_n):
        try:
            registro, imagenes, tabla = pipeline.procesar_frame(entrada, detector, not a.sin_degradar, degradacion)
        except Exception as e:  # RNF-05: un frame malo no detiene el pipeline
            errores += 1
            print(f"frame {entrada['frame_id']} omitido: {e}", file=sys.stderr)
            continue
        registros.append(registro)
        for nombre, m in tabla.items():
            acum = filtros_acum.setdefault(nombre, {"psnr": 0.0, "ssim": 0.0, "ms": 0.0, "n": 0})
            for k in ("psnr", "ssim", "ms"):
                acum[k] += m[k]
            acum["n"] += 1
        if len(registros) <= a.evidencias:
            exportar.guardar_evidencias(imagenes, os.path.join(a.salida, "evidencias"), registro["frame_id"])

    exportar.escribir_jsonl(registros, os.path.join(a.salida, "imagen.jsonl"))
    exportar.escribir_csv(registros, os.path.join(a.salida, "imagen.csv"))

    n = max(len(registros), 1)
    resumen = {
        "frames_procesados": len(registros),
        "frames_omitidos": errores,
        "tiempo_medio_ms": round(sum(r["tiempo_ms"] for r in registros) / n, 1),
        "psnr_degradado": round(sum(r["calidad"]["psnr_degradado"] for r in registros) / n, 2),
        "psnr_mejorado": round(sum(r["calidad"]["psnr"] for r in registros) / n, 2),
        "ssim_degradado": round(sum(r["calidad"]["ssim_degradado"] for r in registros) / n, 4),
        "ssim_mejorado": round(sum(r["calidad"]["ssim"] for r in registros) / n, 4),
        "detecciones_referencia": sum(r["deteccion_video"]["referencia"] for r in registros),
        "recuperadas_sobre_degradado": sum(r["deteccion_video"]["sobre_degradado"] for r in registros),
        "recuperadas_sobre_solo_filtro": sum(r["deteccion_video"]["sobre_solo_filtro"] for r in registros),
        "recuperadas_sobre_mejorado": sum(r["deteccion_video"]["sobre_mejorado"] for r in registros),
        "filtros": {k: {"psnr": round(v["psnr"] / v["n"], 2), "ssim": round(v["ssim"] / v["n"], 4),
                        "ms": round(v["ms"] / v["n"], 1)} for k, v in filtros_acum.items()},
    }
    with open(os.path.join(a.salida, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
