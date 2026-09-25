"""
Smart Classroom Vision - Panel web para el docente (RF09, RNF02, RNF04, RNF10).
Muestra solo indicadores agregados: nunca imágenes, video ni datos de personas.
El acceso requiere usuario y contraseña (RNF02).
"""
import secrets
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file

PAGINA = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Classroom Vision</title>
<style>
  :root { --fondo:#f6f7f9; --tarjeta:#fff; --texto:#1c2430; --suave:#5d6877; --linea:#e2e6ec;
          --acento:#2a6fdb; --ok:#1f8a4c; --alerta:#b7791f; --mal:#c0392b; }
  @media (prefers-color-scheme: dark) {
    :root { --fondo:#12161c; --tarjeta:#1a2028; --texto:#e8ecf1; --suave:#9aa5b4; --linea:#2b3440;
            --acento:#6ea3ff; --ok:#4cc27f; --alerta:#e0a94a; --mal:#ef7a6c; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--fondo); color:var(--texto);
         font:16px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  main { max-width:960px; margin:0 auto; padding:16px; }
  h1 { font-size:1.3rem; margin:8px 0 2px; }
  h2 { font-size:1rem; margin:0 0 10px; color:var(--suave); font-weight:600; }
  .sub { color:var(--suave); margin:0 0 16px; font-size:.9rem; }
  .estado { display:inline-block; padding:2px 10px; border-radius:99px; font-size:.85rem;
            border:1px solid var(--linea); background:var(--tarjeta); }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:16px 0; }
  .card { background:var(--tarjeta); border:1px solid var(--linea); border-radius:10px; padding:14px; }
  .card .n { font-size:2rem; font-weight:700; line-height:1.1; }
  .card .t { color:var(--suave); font-size:.85rem; margin-top:4px; }
  section.card { margin:12px 0; overflow-x:auto; }
  table { width:100%; border-collapse:collapse; font-size:.9rem; }
  th, td { text-align:right; padding:6px 8px; border-bottom:1px solid var(--linea); white-space:nowrap; }
  th:first-child, td:first-child { text-align:left; }
  svg { width:100%; height:auto; display:block; }
  .aviso { font-size:.85rem; color:var(--suave); margin:16px 0; }
  a.boton { color:var(--acento); }
  #vista-img { width:100%; max-width:640px; border-radius:8px; border:1px solid var(--linea); background:#000; }
  button { font:inherit; padding:6px 14px; border-radius:8px; border:1px solid var(--linea); background:var(--tarjeta); color:var(--texto); cursor:pointer; }
</style>
</head>
<body>
<main>
  <h1>Smart Classroom Vision</h1>
  <p class="sub">Indicadores de la clase en curso · <span id="cam" class="estado">conectando…</span></p>

  <div class="grid" id="ahora"></div>

  <section class="card" id="sec-vista" hidden>
    <h2>Vista previa (rostros difuminados)</h2>
    <p><button id="btn-vista" type="button">Ver vista previa</button></p>
    <div id="vista-caja" hidden><img id="vista-img" alt="Vista previa anonimizada del aula"></div>
    <p class="sub">Solo se genera mientras esta sección está abierta, no se guarda y no se envía a la nube. El difuminado puede fallar con
       rostros pequeños, de perfil o parcialmente tapados: no la muestres fuera del aula.</p>
  </section>

  <section class="card">
    <h2>Estudiantes detectados por minuto</h2>
    <div id="grafico"><span class="sub">Aún no hay minutos completos. El primer indicador aparece al terminar el primer minuto.</span></div>
  </section>

  <section class="card">
    <h2>Detalle por minuto</h2>
    <div id="tabla"></div>
    <p class="sub"><a class="boton" href="indicadores.csv">Descargar reporte CSV</a></p>
  </section>

  <p class="aviso">Solo se muestran promedios por minuto, sin identificar a ninguna persona. No se guarda video ni imágenes.
     Los indicadores son <strong>orientativos</strong> y no deben usarse para evaluar o sancionar a estudiantes.</p>
</main>
<script>
const NOMBRES = {"person":"Personas","laptop":"Laptops","cell phone":"Celulares","book":"Libros",
                 "hand-raising":"Levantar la mano","read":"Leer","write":"Escribir","BowHead":"Cabeza agachada","TurnHead":"Cabeza girada"};
const nombre = k => NOMBRES[k] || k;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function tarjetas(e) {
  const t = [];
  const act = e.actual || {};
  const claves = Object.keys(act);
  if (!claves.includes("person")) claves.unshift("person");
  for (const k of claves) t.push([act[k] || 0, nombre(k) + " ahora"]);
  t.push([e.iluminacion ? e.iluminacion : "—", "Iluminación"]);
  t.push([e.apta === null ? "—" : (e.apta ? "Apta" : "No apta"), "Calidad de imagen"]);
  t.push([e.fps ? e.fps.toFixed(1) : "—", "Cuadros por segundo"]);
  $("ahora").innerHTML = t.map(([n, txt]) =>
    `<div class="card"><div class="n">${esc(n)}</div><div class="t">${esc(txt)}</div></div>`).join("");
}

function grafico(reg) {
  if (!reg.length) return;
  const W = 640, H = 200, m = {l:34, r:10, t:10, b:24};
  const ys = reg.map(r => r.person || 0), max = Math.max(1, ...ys);
  const x = i => m.l + (reg.length === 1 ? (W-m.l-m.r)/2 : i*(W-m.l-m.r)/(reg.length-1));
  const y = v => H - m.b - v*(H-m.t-m.b)/max;
  const pts = ys.map((v,i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  $("grafico").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Estudiantes detectados por minuto">
    <line x1="${m.l}" y1="${H-m.b}" x2="${W-m.r}" y2="${H-m.b}" stroke="var(--linea)"/>
    <line x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${H-m.b}" stroke="var(--linea)"/>
    <text x="${m.l-6}" y="${m.t+4}" text-anchor="end" font-size="11" fill="var(--suave)">${max}</text>
    <text x="${m.l-6}" y="${H-m.b}" text-anchor="end" font-size="11" fill="var(--suave)">0</text>
    <text x="${W-m.r}" y="${H-6}" text-anchor="end" font-size="11" fill="var(--suave)">minuto ${reg[reg.length-1].minuto}</text>
    <polyline points="${pts}" fill="none" stroke="var(--acento)" stroke-width="2.5"/>
    ${ys.map((v,i) => `<circle cx="${x(i)}" cy="${y(v)}" r="3.5" fill="var(--acento)"/>`).join("")}
  </svg>`;
}

function tabla(reg) {
  if (!reg.length) { $("tabla").innerHTML = ""; return; }
  const cols = [];
  for (const r of reg) for (const k of Object.keys(r)) if (!["minuto","iluminacion"].includes(k) && !cols.includes(k)) cols.push(k);
  const filas = reg.slice(-15).reverse().map(r =>
    `<tr><td>${r.minuto}</td>${cols.map(c => `<td>${r[c] ?? 0}</td>`).join("")}<td>${esc(r.iluminacion ?? "")}</td></tr>`).join("");
  $("tabla").innerHTML = `<table><thead><tr><th>Minuto</th>${cols.map(c => `<th>${esc(nombre(c))}</th>`).join("")}<th>Iluminación</th></tr></thead><tbody>${filas}</tbody></table>`;
}

async function actualizar() {
  try {
    const e = await (await fetch("api/estado")).json();
    $("cam").textContent = e.camara === "conectada" ? "cámara conectada" : "cámara: " + e.camara;
    $("cam").style.color = e.camara === "conectada" ? "var(--ok)" : "var(--alerta)";
    tarjetas(e); grafico(e.registros || []); tabla(e.registros || []);
  } catch (_) {
    $("cam").textContent = "sin conexión con el sistema"; $("cam").style.color = "var(--mal)";
  }
}
let vistaActiva = false, temporizadorVista = null;
function refrescarVista() {
  if (!vistaActiva) return;
  const img = $("vista-img"), nueva = new Image();
  nueva.onload = () => { img.src = nueva.src; temporizadorVista = setTimeout(refrescarVista, 500); };
  nueva.onerror = () => { temporizadorVista = setTimeout(refrescarVista, 1500); };
  nueva.src = "vista.jpg?t=" + Date.now();
}
function alternarVista(activar) {
  vistaActiva = activar; $("vista-caja").hidden = !activar;
  $("btn-vista").textContent = activar ? "Ocultar vista previa" : "Ver vista previa";
  clearTimeout(temporizadorVista); if (activar) refrescarVista(); else $("vista-img").removeAttribute("src");
}
$("btn-vista").onclick = () => alternarVista(!vistaActiva);
fetch("api/estado").then(r => r.json()).then(e => { if (e.vista) { $("sec-vista").hidden = false; if (location.hash === "#vista") alternarVista(true); } });
actualizar(); setInterval(actualizar, 3000);
</script>
</body>
</html>
"""


def crear_app(estado, bloqueo, ruta_csv, usuario, clave, vista=None):
    """Crea la app web. `estado`, `bloqueo` y `vista` los comparte con el hilo de captura."""
    app = Flask(__name__)
    usuario_b, clave_b = usuario.encode(), clave.encode()

    @app.before_request
    def autenticar():
        a = request.authorization
        if not (a and secrets.compare_digest((a.username or "").encode(), usuario_b)
                and secrets.compare_digest((a.password or "").encode(), clave_b)):
            return Response("Acceso restringido", 401, {"WWW-Authenticate": 'Basic realm="Smart Classroom Vision"'})

    @app.after_request
    def sin_cache(resp):
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/")
    def inicio():
        return Response(PAGINA, mimetype="text/html")

    @app.get("/api/estado")
    def api_estado():
        with bloqueo:
            return jsonify(estado)

    @app.get("/vista.jpg")
    def vista_jpg():
        if vista is None:
            return Response("Vista previa desactivada", 404)
        with bloqueo:
            vista["pedida"] = time.time()            # avisa al hilo de captura de que alguien está mirando
            jpg = vista["jpg"]
        return Response(jpg, mimetype="image/jpeg") if jpg else Response("Sin imagen todavía", 503)

    @app.get("/indicadores.csv")
    def descargar_csv():
        ruta = Path(ruta_csv)
        if not ruta.exists():
            return Response("Aún no hay indicadores: el primer minuto no ha terminado.", 404)
        return send_file(ruta, mimetype="text/csv", as_attachment=True, download_name=ruta.name)

    return app
