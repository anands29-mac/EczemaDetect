// Set window.ECZEMA_API (e.g. "http://localhost:8000") if the frontend is hosted separately.
const API = window.ECZEMA_API || "";
const $ = (id) => document.getElementById(id);

const els = {
  file: $("file"), drop: $("drop"), intro: $("intro"), status: $("status"), result: $("result"),
  photo: $("photo"), heat: $("heat"), again: $("again"),
};

fetch(`${API}/api/health`).then((r) => r.json()).then((h) => {
  const tag = $("modelTag");
  const names = { efficientnet_b0: "EfficientNet-B0", simple_cnn: "Custom CNN" };
  let text = `Model: ${names[h.model] || h.model}`;
  if (h.test_metrics) text += `, ${(h.test_metrics.accuracy * 100).toFixed(0)}% test accuracy`;
  tag.textContent = text;
  tag.hidden = false;
}).catch(() => setStatus("Can't reach the analysis server. Start it with: uvicorn backend.app:app --port 8000", true));

function setStatus(msg, isError = false, busy = false) {
  els.status.textContent = msg;
  els.status.classList.toggle("is-error", isError);
  els.status.classList.toggle("is-busy", busy);
}

els.file.addEventListener("change", () => els.file.files[0] && analyse(els.file.files[0]));
els.drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.file.click(); } });
["dragenter", "dragover"].forEach((t) => els.drop.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.add("is-over"); }));
["dragleave", "drop"].forEach((t) => els.drop.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.remove("is-over"); }));
els.drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) analyse(f); });

els.again.addEventListener("click", () => {
  els.result.hidden = true; els.intro.hidden = false; els.file.value = ""; setStatus("");
  window.scrollTo({ top: 0 }); els.drop.focus();
});

document.querySelectorAll(".seg-btn").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
function showView(view) {
  document.querySelectorAll(".seg-btn").forEach((b) => {
    const on = b.dataset.view === view;
    b.classList.toggle("is-on", on); b.setAttribute("aria-pressed", on);
  });
  els.heat.hidden = view !== "heat";
}

async function analyse(file) {
  if (!file.type.startsWith("image/")) return setStatus("That file isn't an image. Choose a JPG, PNG or WEBP photo.", true);
  if (file.size > 12 * 1024 * 1024) return setStatus("That photo is over 12 MB. Choose a smaller one.", true);
  setStatus("Analysing photo…", false, true);
  const body = new FormData();
  body.append("file", file);
  try {
    const res = await fetch(`${API}/api/predict`, { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "The server couldn't analyse this photo.");
    els.photo.src = URL.createObjectURL(file);
    render(data);
    setStatus("");
  } catch (err) {
    setStatus(err.message || "The server couldn't analyse this photo.", true);
  }
}

const pct = (p) => `${Math.round(p * 100)}%`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const list = (items) => `<ul>${items.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>`;

function confidenceWords(p) {
  if (p >= 0.85) return "high confidence";
  if (p >= 0.65) return "fairly confident";
  if (p >= 0.45) return "moderate confidence";
  return "low confidence";
}

function render(d) {
  const c = d.condition;
  els.heat.src = d.heatmap_png;
  showView("photo");

  const verdict = $("verdict");
  verdict.classList.toggle("is-uncertain", d.uncertain);
  if (d.uncertain) {
    $("kicker").textContent = "No clear match";
    $("verdictTitle").textContent = `Closest match: ${c.name}`;
  } else if (d.is_eczema) {
    $("kicker").textContent = "Looks like eczema";
    $("verdictTitle").textContent = c.name;
  } else {
    $("kicker").textContent = "No eczema pattern found";
    $("verdictTitle").textContent = "Looks like healthy skin";
  }
  let conf = `${pct(d.confidence)} for this answer, ${confidenceWords(d.confidence)}. `;
  conf += d.is_eczema
    ? `Combined chance that it is some type of eczema: ${pct(d.eczema_probability)}.`
    : `Chance that it is some type of eczema: ${pct(d.eczema_probability)}; the closest type would be ${d.most_likely_eczema_type}.`;
  $("verdictConf").textContent = conf;
  $("warnings").innerHTML = d.warnings.map((w) => `<p class="warn">${esc(w)}</p>`).join("");

  $("probList").innerHTML = d.probabilities.map((p, i) => `
    <li class="prob${i === 0 ? " is-top" : ""}">
      <span>${esc(p.label === "Normal" ? "Healthy skin" : p.label)}</span>
      <span class="prob-bar" aria-hidden="true"><span class="prob-fill" data-w="${(p.probability * 100).toFixed(1)}"></span></span>
      <span class="prob-val">${pct(p.probability)}</span>
    </li>`).join("");
  requestAnimationFrame(() => document.querySelectorAll(".prob-fill").forEach((f) => { f.style.width = `${f.dataset.w}%`; }));

  let info = `<section><p class="summary">${esc(c.summary)}</p>`;
  if (c.typical_look.length) info += `<h3 style="margin-top:20px">What it usually looks like</h3>${list(c.typical_look)}`;
  info += `</section>`;
  if (c.causes.length) info += `<section><h3>Common causes and triggers</h3>${list(c.causes)}</section>`;
  info += `<section><h3>What you can do now</h3>${list(c.self_care)}`;
  info += `<div class="doctor"><strong>When to see a doctor</strong>${esc(c.see_doctor_if)}</div></section>`;
  if (c.medical_options.length) info += `<details><summary>Treatments a clinician may offer</summary>${list(c.medical_options)}</details>`;
  if (d.is_eczema) info += `<details><summary>Everyday skin care for any eczema</summary>${list(d.general_care)}</details>`;
  $("info").innerHTML = info;

  $("flagList").innerHTML = d.red_flags.map((f) => `<li>${esc(f)}</li>`).join("");
  $("disclaimer").textContent = d.disclaimer;
  const srcs = [{ title: `More on ${c.name === "Normal" ? "eczema types" : c.name}`, url: c.source }, ...d.sources];
  $("sources").innerHTML = "Sources: " + srcs.map((s) => `<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title)}</a>`).join(", ");

  els.intro.hidden = true;
  els.result.hidden = false;
  window.scrollTo({ top: 0 });
}
