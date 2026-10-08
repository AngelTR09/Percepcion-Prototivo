# Sistema de Supervision Inteligente

Sistema de supervision de examenes mediante inteligencia artificial y vision por computadora.

## ¿De que trata el proyecto?

El proyecto busca apoyar la supervision de evaluaciones mediante una camara y un modelo de inteligencia artificial entrenado con YOLO.

El sistema analiza en tiempo real lo que aparece frente a la camara y permite detectar diferentes elementos relacionados con el estudiante.

### ¿Que hace?

- Detecta personas.
- Detecta mochilas.
- Detecta telefonos.
- Detecta cuadernos/libros.
- Detecta audifonos.
- Detecta relojes.
- Detecta laptops.
- Muestra las detecciones mediante cuadros de colores.
- Los elementos permitidos se muestran en azul.
- Los elementos no permitidos se muestran en rojo.
- Inicia un contador cuando detecta un elemento no permitido.
- Si permanece durante 5 segundos, genera una alerta.
- Guarda una captura de la alerta.
- Reproduce un sonido de alerta.
- El contador se reinicia cuando el elemento deja de ser detectado.

### 1. Instalar Python

Descargar Python 3.12:

[https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe]

Durante la instalacion marcar:

`Add Python to PATH`

### 2. Descargar el proyecto

Clonar el repositorio:

`git clone https://github.com/cesarcabanillas1921-bit/SistemaSupervisionExamenes.git`

Entrar a la carpeta:

`cd SistemaSupervisionExamenes`

### 3. Instalar las librerias

Ejecutar:

`pip install -r requisitos.txt`

### 4. Ejecutar el sistema

Conectar la camara y ejecutar:

`python principal.py`

### 5. Salir del sistema

Presionar:

`Q`

### Importante

El modelo se encuentra en:

`modelo_ia/best.pt`

El sonido de alerta se encuentra en:

`sonido/alerta.mp3`

No eliminar estos archivos.
