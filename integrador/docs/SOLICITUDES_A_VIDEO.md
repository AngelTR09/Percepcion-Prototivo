# Solicitudes al grupo VIDEO

No modificamos su código. Cada ítem lo resolvemos en `imagen/adaptador_video.py`; si ellos lo aplican, solo se ajusta el adaptador.

| # | Solicitud | Por qué | Mientras tanto |
|---|---|---|---|
| 1 | Entregar `frame_id`, `timestamp` y `fps` con cada frame | PREDICCIÓN sincroniza imagen y audio por `timestamp` | El adaptador calcula `frame_id / fps` |
| 2 | Aceptar un archivo de video como fuente, no solo cámara | Pruebas repetibles y exposición | El adaptador usa `cv2.VideoCapture(ruta)` |
| 3 | Separar la pista de audio a `.wav` (indicar frecuencia de muestreo) | La necesita AUDIO | Fuera de nuestro alcance |
| 4 | Que `detectar_objetos` no cargue el modelo al importar | Importar el módulo tarda y falla sin `best.pt` | Importación diferida en el adaptador |
| 5 | Documentar qué valores de `config.py` están en uso | `RUTA_MODELO` y `CONFIANZA_MINIMA` no se usan | Ignoramos esos valores |
