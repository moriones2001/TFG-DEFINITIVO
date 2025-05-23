// script.js — List view sin “desplazamientos” al añadir mensajes

const WS_URL = "ws://127.0.0.1:8000/ws";
const socket = new WebSocket(WS_URL);

const list = document.getElementById("message-list");
let selectedIndex = -1;

// Theme toggle
document.getElementById("toggle-theme").onclick = () => {
  const cur = document.documentElement.getAttribute("data-theme");
  document.documentElement.setAttribute("data-theme", cur === "dark" ? "light" : "dark");
};

// WS events
socket.addEventListener("open",  () => console.log("WS connected"));
socket.addEventListener("error", e => console.error("WS error", e));
socket.addEventListener("close", e => console.warn("WS closed", e.code));
socket.addEventListener("message", e => {
  try {
    const msg = JSON.parse(e.data);
    addMessageItem(msg);
  } catch (err) {
    console.error("Invalid JSON", err);
  }
});

// Crea y añade un <li> al final de la lista (append), sin modificar selectedIndex
function addMessageItem(msg) {
  // DEBUG: Log del mensaje recibido
  console.log("MSG RECIBIDO:", msg);

  const li = document.createElement("li");
  li.className = "message-item";

  // Avatar con fallback
  const img = document.createElement("img");
  img.src = `https://static-cdn.jtvnw.net/jtv_user_pictures/${msg.user?.toLowerCase() || "anon"}-profile_image-70x70.png`;
  img.onerror = () => { img.onerror = null; img.src = "default-avatar.png"; };
  li.appendChild(img);

  // Info
  const info = document.createElement("div");
  info.className = "info";
  const lora = msg.prediction_lora || {};
  const tfidf = msg.prediction_tfidf || {};
  const clasico = msg.prediction_clasico || {};

  // Si no hay al menos una predicción, ignora el mensaje
  if (!lora.label && !tfidf.label) {
    console.warn("Mensaje recibido sin predicción válida, ignorado:", msg);
    return;
  }

  info.innerHTML = `
    <div class="user">${msg.user || "?"}</div>
    <div class="channel">${msg.channel || "?"}</div>
    <div class="text">${msg.text || ""}</div>
    <div class="prediction">
      <span class="label ${lora.label || "unknown"}">[LoRA] ${lora.label !== undefined ? lora.label : "?"} ${(lora.prob !== undefined ? (lora.prob*100).toFixed(1)+'%' : '')}</span>
      <span class="label ${tfidf.label || "unknown"}">[TFIDF] ${tfidf.label !== undefined ? tfidf.label : "?"} ${(tfidf.prob !== undefined ? (tfidf.prob*100).toFixed(1)+'%' : '')}</span>
      <span class="label ${clasico.label || "unknown"}">[CLASICO] ${clasico.label !== undefined ? clasico.label : "?"} ${(clasico.prob !== undefined ? (clasico.prob*100).toFixed(1)+'%' : '')}</span>
    </div>
  `;
  li.appendChild(info);

  // Acciones
  const actions = document.createElement("div");
  actions.className = "actions";
  const banBtn = document.createElement("button");
  banBtn.className = "ban"; banBtn.textContent = "🚫";
  const allowBtn = document.createElement("button");
  allowBtn.className = "allow"; allowBtn.textContent = "✅";
  actions.append(banBtn, allowBtn);
  li.appendChild(actions);

  // Handlers de feedback
  banBtn.onclick   = () => rateItem(li, msg._id, true);
  allowBtn.onclick = () => rateItem(li, msg._id, false);

  // Insertar al final (nuevo mensaje abajo)
  list.appendChild(li);

  // Si es el primer mensaje, selecciónalo
  if (selectedIndex === -1) {
    selectItem(0);
  }
}

// Envía feedback y marca como reviewed
async function rateItem(li, id, confirmado) {
  if (li.classList.contains("reviewed")) return;
  try {
    const res = await fetch("http://127.0.0.1:8000/feedback", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({id, confirmado, moderador:"webui"})
    });
    const data = await res.json();
    if (data.status === "ok") {
      li.classList.add("reviewed");
      // Al calificar, avanza la selección
      selectItem(selectedIndex + 1);
    }
  } catch (err) {
    console.error(err);
  }
}

// Selecciona visualmente el item en selectedIndex
function selectItem(idx) {
  const items = Array.from(list.children);
  if (items[selectedIndex]) items[selectedIndex].classList.remove("selected");
  selectedIndex = idx;
  if (items[selectedIndex]) {
    items[selectedIndex].classList.add("selected");
    // Scroll a la vista solo cuando cambias manualmente
    items[selectedIndex].scrollIntoView({behavior:"smooth", block:"center"});
  }
}

// Atajos de teclado B/N y flechas
document.addEventListener("keydown", e => {
  const items = Array.from(list.children);
  if (!items[selectedIndex]) return;
  if (e.key.toLowerCase() === "b") {
    items[selectedIndex].querySelector("button.ban").click();
  } else if (e.key.toLowerCase() === "n") {
    items[selectedIndex].querySelector("button.allow").click();
  } else if (e.key === "ArrowDown" && selectedIndex < items.length - 1) {
    selectItem(selectedIndex + 1);
  } else if (e.key === "ArrowUp" && selectedIndex > 0) {
    selectItem(selectedIndex - 1);
  }
});
