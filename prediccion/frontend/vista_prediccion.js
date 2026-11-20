// Servicio PREDICCION: muestra lo que recibio de VIDEO (deteccion) y de IMAGEN (verificacion), con la
// evaluacion de su motor, y analiza a la persona captada en ese momento (rostro y expresion).
// Solo llega lo que IMAGEN confirmo: lo que no verifica es un falso positivo y no llega a este servicio.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const pct = (v) => Math.round(v * 100) + "%";
  const VEREDICTO = { verificado: "✅ verificado", probable: "🟡 probable", no_verificado: "❌ no verificado" };
  const NIVEL = { bajo: "bajo", moderado: "moderado", alto: "alto" };
  const PAUSA_ANALISIS_MS = 8000;
  let vistoTs = 0, ultimoAnalisis = -Infinity, personaActual = null;

  function el(tag, texto, clase) { const n = document.createElement(tag); if (texto != null) n.textContent = texto; if (clase) n.className = clase; return n; }
  function lista(items) { const ul = el("ul"); items.forEach((t) => ul.append(el("li", t))); return ul; }
  function figura(src, titulo) { const f = el("figure"); const i = el("img"); i.src = src; i.alt = titulo; f.append(i, el("figcaption", titulo)); return f; }

  function columnaVideo(u, persona) {
    const h = u.hallazgo, col = el("div", null, "cadena-col");
    col.append(el("h4", "🎥 VIDEO · lo que detectó"));
    const par = el("div", null, "cadena-miniaturas");
    par.append(figura(u.imagen.objeto, "Objeto prohibido"));
    if (persona) par.append(figura(persona.dataUrl, persona.detectada ? `Persona en ese momento (${pct(persona.confianza)})` : "Fotograma completo"));
    col.append(par);
    col.append(el("p", h.reclasificado_desde
      ? `VIDEO dijo ${h.reclasificado_desde} (${pct(h.confianza)}); IMAGEN lo reclasificó como ${h.etiqueta}.`
      : `${h.etiqueta} con ${pct(h.confianza)} de confianza (umbral ${pct(h.umbral)}).`));
    col.append(el("p", `Recibido a las ${new Date(u.ts * 1000).toLocaleTimeString("es")}. IMAGEN verificó el objeto y lo entregó a PREDICCIÓN; la persona se analiza aquí.`));
    return col;
  }

  function columnaImagen(u) {
    const col = el("div", null, "cadena-col");
    col.append(el("h4", `🖼️ IMAGEN · ${VEREDICTO[u.imagen.veredicto] || u.imagen.veredicto}`));
    col.append(lista(u.imagen.motivos));
    const mini = el("div", null, "cadena-miniaturas");
    u.imagen.miniaturas.forEach((m) => mini.append(figura(m.imagen, m.titulo)));
    col.append(mini);
    col.append(el("p", "El detalle completo de cada paso está en el servicio Imagen."));
    return col;
  }

  function columnaPrediccion(p) {
    const col = el("div", null, "cadena-col");
    col.append(el("h4", "🔮 PREDICCIÓN · evaluación del motor"));
    if (!p || p.error) { col.append(el("p", "El motor de PREDICCIÓN no pudo evaluar este hallazgo." + (p && p.error ? " " + p.error : ""))); return col; }
    col.append(el("div", p.indice_revision + " / 100", "cadena-indice"));
    col.append(el("p", `Nivel de revisión: ${NIVEL[p.nivel_revision] || p.nivel_revision}` + (p.requiere_revision_humana ? " · requiere revisión humana" : "")));
    if (p.evidencia && p.evidencia.length) col.append(lista(p.evidencia.map((e) => `${e.senal}: ${e.explicacion} (${e.modalidades.join(" + ")})`)));
    if (p.modalidades_faltantes && p.modalidades_faltantes.length) col.append(el("p", "Sin datos de: " + p.modalidades_faltantes.join(", ") + "."));
    col.append(el("p", "Recomendación: " + p.recomendacion));
    col.append(el("p", "El análisis facial de la persona (rostro y expresión) aparece más abajo, en «Detección y Análisis Facial»."));
    if (p.limitaciones && p.limitaciones.length) col.append(el("p", p.limitaciones.join(" ")));
    return col;
  }

  function pintar(u) {
    $("cadena-grid").replaceChildren(columnaVideo(u, personaActual), columnaImagen(u), columnaPrediccion(u.prediccion));
    $("cadena-grid").hidden = false; $("cadena-vacio").hidden = true;
  }

  // La persona se analiza con el mismo flujo que una foto subida: rostro, expresion y resultados del panel.
  async function analizarPersona(u) {
    personaActual = u.persona;
    pintar(u);
    await procesarCaptura(() => u.persona.dataUrl, "archivo");
    // El recuadro de VIDEO puede no incluir la cara: si no hay rostro, se reintenta con el fotograma completo.
    if (u.persona.completa && /No se detectó ningún rostro/.test($("capture-status-text")?.textContent || "")) {
      personaActual = { dataUrl: u.persona.completa, detectada: false, confianza: null };
      pintar(u);
      await procesarCaptura(() => u.persona.completa, "archivo");
    }
  }

  async function consultar() {
    try {
      const u = await (await fetch("/api/ultimo")).json();
      if (!u.ts || u.ts === vistoTs) return;
      vistoTs = u.ts;
      const libre = typeof procesarCaptura === "function" && typeof peticionCaptura !== "undefined" && !peticionCaptura;
      if (u.persona && libre && performance.now() - ultimoAnalisis >= PAUSA_ANALISIS_MS) {
        ultimoAnalisis = performance.now();
        analizarPersona(u);
      } else {
        pintar(u);
      }
    } catch (e) { /* el servicio se esta reiniciando */ }
  }

  document.addEventListener("DOMContentLoaded", () => { consultar(); setInterval(consultar, 1500); });
})();
