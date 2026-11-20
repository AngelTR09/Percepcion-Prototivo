// Apartado PREDICCIONES: muestra lo que recibio de VIDEO (deteccion) y de IMAGEN (verificacion),
// junto con la evaluacion que el motor de PREDICCION hizo con esos datos.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const pct = (v) => Math.round(v * 100) + "%";
  const VEREDICTO = { verificado: "✅ verificado", probable: "🟡 probable", no_verificado: "❌ no verificado" };
  const NIVEL = { bajo: "bajo", moderado: "moderado", alto: "alto" };
  // Pasos de IMAGEN que se muestran como evidencia resumida
  const CLAVE = ["CLAHE", "Bordes Canny", "Otsu + morfología", "Contornos y forma"];

  function el(tag, texto, clase) { const n = document.createElement(tag); if (texto != null) n.textContent = texto; if (clase) n.className = clase; return n; }
  function lista(items) { const ul = el("ul"); items.forEach((t) => ul.append(el("li", t))); return ul; }

  function columnaVideo(h, v, hora, persona) {
    const col = el("div", null, "cadena-col");
    col.append(el("h4", "🎥 VIDEO · lo que detectó"));
    const original = v.secciones[0].items[0];
    const par = el("div", null, "cadena-miniaturas");
    const f1 = el("figure"); const i1 = el("img"); i1.src = original.imagen; i1.alt = "Objeto prohibido";
    f1.append(i1, el("figcaption", "Objeto prohibido"));
    par.append(f1);
    if (persona) {
      const f2 = el("figure"); const i2 = el("img"); i2.src = persona.dataUrl; i2.alt = "Persona en ese momento";
      f2.append(i2, el("figcaption", persona.detectada ? `Persona en ese momento (${pct(persona.confianza)})` : "Fotograma completo"));
      par.append(f2);
    }
    col.append(par);
    col.append(el("p", `${h.etiqueta} con ${pct(h.confianza)} de confianza (umbral ${pct(h.umbral)}).`));
    col.append(el("p", `Detectado a las ${hora.toLocaleTimeString("es")}. El objeto pasa a IMAGEN; solo si IMAGEN lo verifica, la persona se analiza en PREDICCIÓN.`));
    return col;
  }

  function columnaImagen(v) {
    const col = el("div", null, "cadena-col");
    col.append(el("h4", `🖼️ IMAGEN · ${VEREDICTO[v.veredicto] || v.veredicto}`));
    col.append(lista(v.motivos));
    const mini = el("div", null, "cadena-miniaturas");
    const todos = v.secciones.flatMap((s) => s.items);
    for (const titulo of CLAVE) {
      const it = todos.find((x) => x.titulo === titulo);
      if (!it) continue;
      const fig = el("figure"); const img = el("img"); img.src = it.imagen; img.alt = it.titulo;
      fig.append(img, el("figcaption", it.titulo)); mini.append(fig);
    }
    col.append(mini);
    col.append(el("p", "El detalle completo de cada paso está en el apartado Imagen."));
    return col;
  }

  function columnaPrediccion(p, pasa) {
    const col = el("div", null, "cadena-col");
    col.append(el("h4", "🔮 PREDICCIÓN · evaluación del motor"));
    if (pasa === false) {
      col.append(el("p", "Sin evaluación: IMAGEN no verificó el objeto y lo descartó como falso positivo, por eso no llega a PREDICCIÓN."));
      return col;
    }
    if (!p || p.error) {
      col.append(el("p", "El motor de PREDICCIÓN no pudo evaluar este hallazgo." + (p && p.error ? " " + p.error : "")));
      return col;
    }
    col.append(el("div", p.indice_revision + " / 100", "cadena-indice"));
    col.append(el("p", `Nivel de revisión: ${NIVEL[p.nivel_revision] || p.nivel_revision}` + (p.requiere_revision_humana ? " · requiere revisión humana" : "")));
    if (p.evidencia && p.evidencia.length) col.append(lista(p.evidencia.map((e) => `${e.senal}: ${e.explicacion} (${e.modalidades.join(" + ")})`)));
    if (p.modalidades_faltantes && p.modalidades_faltantes.length) col.append(el("p", "Sin datos de: " + p.modalidades_faltantes.join(", ") + "."));
    col.append(el("p", "Recomendación: " + p.recomendacion));
    col.append(el("p", "El análisis facial de la persona (rostro y expresión) aparece más abajo, en «Detección y Análisis Facial»."));
    if (p.limitaciones && p.limitaciones.length) col.append(el("p", p.limitaciones.join(" ")));
    return col;
  }

  // La persona se analiza con el mismo flujo que una foto subida: rostro, expresion y resultados del panel.
  const PAUSA_ANALISIS_MS = 8000;
  let personaActual = null, ultimoAnalisis = 0;

  function pintar(hallazgo, resultado, hora, descartado) {
    const grid = $("cadena-grid");
    grid.replaceChildren(columnaVideo(hallazgo, resultado, hora, descartado ? null : personaActual), columnaImagen(resultado), columnaPrediccion(resultado.prediccion, resultado.pasa_a_prediccion));
    grid.hidden = false; $("cadena-vacio").hidden = true;
  }

  async function analizarPersona(nueva, hallazgo, resultado, hora) {
    personaActual = nueva;
    pintar(hallazgo, resultado, hora);
    await procesarCaptura(() => nueva.dataUrl, "archivo");   // analisis facial de la persona (panel de abajo)
    // El recuadro de VIDEO puede no incluir la cara: si no hay rostro, se reintenta con el fotograma completo.
    if (nueva.completa && /No se detectó ningún rostro/.test($("capture-status-text")?.textContent || "")) {
      personaActual = { dataUrl: nueva.completa, detectada: false, confianza: null };
      pintar(hallazgo, resultado, hora);
      await procesarCaptura(() => nueva.completa, "archivo");
    }
  }

  document.addEventListener("analisis-imagen", (e) => {
    const { hallazgo, resultado, hora } = e.detail;
    if (!$("cadena-grid")) return;
    const nueva = hallazgo.persona;
    const libre = typeof procesarCaptura === "function" && typeof peticionCaptura !== "undefined" && !peticionCaptura;
    // IMAGEN tiene la ultima palabra: solo lo verificado llega a PREDICCION (persona y evaluacion).
    if (!resultado.pasa_a_prediccion) { pintar(hallazgo, resultado, hora, true); return; }
    if (nueva && libre && performance.now() - ultimoAnalisis >= PAUSA_ANALISIS_MS) {
      ultimoAnalisis = performance.now();
      analizarPersona(nueva, hallazgo, resultado, hora);
    } else {
      pintar(hallazgo, resultado, hora);
    }
  });
})();
