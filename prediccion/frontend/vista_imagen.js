// Apartado IMAGEN: recibe el objeto que VIDEO detecto y muestra su analisis con los temas del curso.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const pct = (v) => Math.round(v * 100) + "%";
  const ESTILO = { verificado: ["ok", "✅ verificado"], probable: ["medio", "🟡 probable"], no_verificado: ["mal", "❌ no verificado"] };
  let ocupado = false;

  function el(tag, texto, clase) { const n = document.createElement(tag); if (texto != null) n.textContent = texto; if (clase) n.className = clase; return n; }

  async function analizar(h) {
    if (ocupado) return;
    ocupado = true;
    $("i-video").className = "verif-paso aviso";
    $("i-video").textContent = `🎥 VIDEO: ${h.etiqueta} ${pct(h.confianza)} (≥ ${pct(h.umbral)})`;
    $("i-imagen").className = "verif-paso aviso"; $("i-imagen").textContent = "🖼️ IMAGEN: analizando…";
    try {
      const r = await fetch(`/api/integrado/verificar?clase=${encodeURIComponent(h.clase)}&confianza=${h.confianza}`,
        { method: "POST", headers: { "Content-Type": "image/jpeg" }, body: h.recorte });
      if (!r.ok) throw new Error("HTTP " + r.status);
      pintar(await r.json(), h);
      const badge = document.querySelector('[data-vista="imagen"] .menu-badge-paso');
      if (badge && $("vista-imagen").hidden) badge.classList.add("nuevo");
    } catch (e) {
      $("i-imagen").className = "verif-paso mal"; $("i-imagen").textContent = "🖼️ IMAGEN: no se pudo analizar";
    } finally { ocupado = false; }
  }

  function pintar(v, h) {
    const [clase, etiqueta] = ESTILO[v.veredicto] || ["", v.veredicto];
    $("i-imagen").className = "verif-paso " + clase; $("i-imagen").textContent = `🖼️ IMAGEN: ${h.etiqueta} ${etiqueta}`;
    const pp = $("i-prediccion");
    pp.className = "verif-paso " + (v.pasa_a_prediccion ? "ok" : "mal");
    pp.textContent = v.pasa_a_prediccion ? "🔮 PREDICCIÓN: recibido" : "🔮 PREDICCIÓN: no recibe nada · falso positivo";
    $("i-ayuda").textContent = `Último hallazgo a las ${new Date().toLocaleTimeString("es")}. IMAGEN analiza el recorte de ${v.tamano_recorte[0]} × ${v.tamano_recorte[1]} px que VIDEO entregó.`;

    const det = $("i-detalle"); det.replaceChildren();
    const motivos = el("div"); motivos.append(el("h4", "Por qué este veredicto"));
    const ul = el("ul"); v.motivos.forEach((m) => ul.append(el("li", m)));
    ul.append(el("li", v.decision)); motivos.append(ul);
    const medidas = el("div"); medidas.append(el("h4", "Medidas de IMAGEN"));
    const dl = el("dl"); const filas = [];
    if (v.forma) filas.push(["Proporción (largo/ancho)", v.forma.aspecto], ["Vértices del contorno", v.forma.vertices],
      ["Rectangularidad", v.forma.rectangularidad], ["Solidez", v.forma.solidez], ["Cobertura del contorno", pct(v.forma.cobertura)]);
    filas.push(["VIDEO sobre el recorte mejorado", v.redeteccion.confirmada ? "confirma " + pct(v.redeteccion.confianza) : "no confirma"]);
    filas.forEach(([k, val]) => dl.append(el("dt", k), el("dd", String(val))));
    medidas.append(dl); det.append(motivos, medidas); det.hidden = false;

    const cont = $("i-secciones"); cont.replaceChildren();
    for (const s of v.secciones) {
      const card = el("section", null, "card i-seccion");
      const cab = el("div", null, "i-cabecera");
      cab.append(el("span", s.semana, "i-semana"), el("h3", s.titulo, "card-title"));
      card.append(cab, el("p", s.descripcion, "verif-ayuda"));
      const rejilla = el("div", null, "verif-pasos");
      for (const it of s.items) {
        const fig = el("figure"); const img = el("img"); img.src = it.imagen; img.alt = it.titulo;
        const cap = el("figcaption"); cap.append(el("strong", it.titulo), it.nota);
        fig.append(img, cap); rejilla.append(fig);
      }
      card.append(rejilla); cont.append(card);
    }
    // Lo recibido de VIDEO y lo verificado por IMAGEN pasa a PREDICCION (la tarjeta de su apartado lo muestra).
    document.dispatchEvent(new CustomEvent("analisis-imagen", { detail: { hallazgo: h, resultado: v, hora: new Date() } }));
  }

  document.addEventListener("hallazgo-video", (e) => analizar(e.detail));
  document.addEventListener("vista-cambiada", (e) => {
    if (e.detail === "imagen") document.querySelector('[data-vista="imagen"] .menu-badge-paso')?.classList.remove("nuevo");
  });
})();
