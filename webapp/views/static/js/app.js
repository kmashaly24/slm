const query = document.getElementById("query");
const matches = document.getElementById("matches");
const casePanel = document.getElementById("case");
const symbolChip = document.getElementById("symbol-chip");
const company = document.getElementById("company");
const arabic = document.getElementById("arabic");
const meta = document.getElementById("meta");
const stats = document.getElementById("stats");
const closes = document.getElementById("closes");
const newsView = document.getElementById("news-view");
const news = document.getElementById("news");
const horizon = document.getElementById("horizon");
const analyzeButton = document.getElementById("analyze");
const status = document.getElementById("status");
const results = document.getElementById("results");

const BUCKETS = ["Corporate", "Sector", "Macro"];
const LABELS = {
  A: "Five-day direction",
  C: "Macro and sector",
  GRPO: "Return class",
};

let timer = null;
let symbol = "";
let chart = null;

query.addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(search, 250);
});

analyzeButton.addEventListener("click", runAnalysis);

async function search() {
  const q = query.value.trim();
  matches.innerHTML = "";
  if (!q) {
    matches.hidden = true;
    return;
  }
  try {
    const response = await fetch("/api/search?q=" + encodeURIComponent(q));
    const data = await readJson(response);
    if (!data.length) {
      matches.hidden = true;
      return;
    }
    for (const hit of data) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      const name = document.createElement("span");
      name.textContent = hit.name;
      const code = document.createElement("span");
      code.className = "symbol";
      code.textContent = hit.symbol;
      button.append(name, code);
      button.addEventListener("click", () => choose(hit));
      item.appendChild(button);
      matches.appendChild(item);
    }
    matches.hidden = false;
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function choose(hit) {
  symbol = hit.symbol;
  query.value = hit.name;
  matches.hidden = true;
  results.innerHTML = "";
  setStatus("The model is choosing the headlines that can affect this stock. The first load also loads the weights.");
  casePanel.hidden = false;
  try {
    const response = await fetch("/api/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol: hit.symbol }),
    });
    const data = await readJson(response);
    showCase(data);
    setStatus("");
  } catch (error) {
    setStatus(error.message, true);
  }
}

function showCase(data) {
  symbol = data.symbol;
  const parts = String(data.company || "").split(" / ");
  symbolChip.textContent = data.symbol;
  company.textContent = parts[0] || data.company;
  arabic.textContent = parts[1] || "";
  meta.innerHTML = "";
  for (const label of [formatDay(data.analysis_date), data.sector || "Sector unavailable"]) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = label;
    meta.appendChild(chip);
  }
  renderStats(data.summary);
  renderCloses(data.summary);
  renderNews(data.news_block);
  drawChart(data.chart);
}

function renderStats(summary) {
  const parsed = parseSummary(summary);
  stats.innerHTML = "";
  if (!parsed) {
    const note = document.createElement("p");
    note.className = "status";
    note.textContent = summary;
    stats.appendChild(note);
    return;
  }
  const changeClass = parsed.change.startsWith("-") ? "down" : "up";
  const tiles = [
    ["Last", parsed.last],
    ["Change", parsed.change + "%", changeClass],
    ["High", parsed.high],
    ["Low", parsed.low],
  ];
  for (const [label, value, tone] of tiles) {
    const tile = document.createElement("div");
    tile.className = "stat" + (tone ? " " + tone : "");
    const caption = document.createElement("span");
    caption.textContent = label;
    const number = document.createElement("strong");
    number.textContent = value;
    tile.append(caption, number);
    stats.appendChild(tile);
  }
}

function renderCloses(summary) {
  const parsed = parseSummary(summary);
  closes.innerHTML = "";
  if (!parsed) return;
  let previous = null;
  for (const point of parsed.closes) {
    const cell = document.createElement("div");
    const direction = previous == null ? "" : point.close < previous ? "down" : point.close > previous ? "up" : "";
    cell.className = "close-cell" + (direction ? " " + direction : "");
    const day = document.createElement("span");
    day.textContent = formatDay(point.date);
    const price = document.createElement("strong");
    price.textContent = point.close.toFixed(2);
    cell.append(day, price);
    closes.appendChild(cell);
    previous = point.close;
  }
}

function renderNews(block) {
  news.value = block;
  const groups = parseNews(block);
  newsView.innerHTML = "";
  for (const bucket of BUCKETS) {
    const group = document.createElement("section");
    group.className = "news-group";
    group.dataset.bucket = bucket;
    const title = document.createElement("h4");
    title.textContent = bucket;
    group.appendChild(title);
    const items = groups[bucket];
    if (!items.length) {
      const empty = document.createElement("p");
      empty.className = "empty-line";
      empty.contentEditable = "true";
      empty.textContent = "none in this window.";
      group.appendChild(empty);
    }
    for (const item of items) group.appendChild(headlineNode(item));
    const add = document.createElement("button");
    add.type = "button";
    add.className = "add-line";
    add.textContent = "Add headline";
    add.addEventListener("click", () => {
      const empty = group.querySelector(".empty-line");
      if (empty) empty.remove();
      const node = headlineNode({ title: "", source: "", when: "" });
      group.insertBefore(node, add);
      node.querySelector(".headline-title").focus();
    });
    group.appendChild(add);
    newsView.appendChild(group);
  }
}

function headlineNode(item) {
  const row = document.createElement("article");
  row.className = "headline";
  const title = document.createElement("p");
  title.className = "headline-title";
  title.contentEditable = "true";
  title.textContent = item.title;
  const details = document.createElement("p");
  details.className = "headline-meta";
  const source = document.createElement("span");
  source.contentEditable = "true";
  source.textContent = item.source;
  const when = document.createElement("span");
  when.contentEditable = "true";
  when.textContent = shortStamp(item.when);
  details.append(source, when);
  row.append(title, details);
  return row;
}

function collectNews() {
  const blocks = [];
  for (const group of newsView.querySelectorAll(".news-group")) {
    const lines = [];
    const empty = group.querySelector(".empty-line");
    if (empty) {
      const text = empty.textContent.trim();
      if (text) lines.push(text);
    }
    for (const row of group.querySelectorAll(".headline")) {
      const title = row.querySelector(".headline-title").textContent.trim();
      const source = row.querySelector(".headline-meta span:nth-child(1)").textContent.trim();
      const when = row.querySelector(".headline-meta span:nth-child(2)").textContent.trim();
      if (!title) continue;
      let line = title;
      if (source) line += ` (${source})`;
      if (when) line += ` [${when}]`;
      lines.push(line);
    }
    const body = lines.length ? lines.map((line) => (line.toLowerCase() === "none in this window." ? line : `- ${line}`)).join("\n") : "none in this window.";
    blocks.push(`${group.dataset.bucket}:\n${body}`);
  }
  news.value = blocks.join("\n");
  return news.value;
}

function drawChart(series) {
  const context = document.getElementById("chart");
  if (chart) chart.destroy();
  const values = series.closes;
  const rising = values[values.length - 1] >= values[0];
  const color = rising ? "#8fd4c8" : "#f0a097";
  chart = new Chart(context, {
    type: "line",
    data: {
      labels: series.dates.map(formatDay),
      datasets: [{
        data: values,
        borderColor: color,
        backgroundColor: rising ? "rgba(143, 212, 200, 0.16)" : "rgba(240, 160, 151, 0.14)",
        borderWidth: 2.5,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.25,
        fill: true,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#161f29",
          titleColor: "#b7c2cb",
          bodyColor: "#f2ece3",
          borderColor: "#31404c",
          borderWidth: 1,
          displayColors: false,
          callbacks: { label: (item) => Number(item.raw).toFixed(2) },
        },
      },
      scales: {
        x: { ticks: { maxTicksLimit: 6, color: "#b7c2cb" }, grid: { display: false } },
        y: {
          ticks: { color: "#b7c2cb", callback: (value) => Number(value).toFixed(2) },
          grid: { color: "#31404c" },
        },
      },
    },
  });
}

async function runAnalysis() {
  if (!symbol) return;
  analyzeButton.disabled = true;
  results.innerHTML = "";
  setStatus("Running the two questions. The first run loads the model and can take several minutes.");
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symbol,
        news_block: collectNews(),
        horizon: Number(horizon.value),
      }),
    });
    const data = await readJson(response);
    results.innerHTML = "";
    for (const card of data.results) results.appendChild(answerCard(card));
    setStatus("");
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    analyzeButton.disabled = false;
  }
}

function answerCard(card) {
  const article = document.createElement("article");
  article.className = "card" + (card.key === "GRPO" ? " grpo" : "");
  const key = document.createElement("p");
  key.className = "key";
  key.textContent = LABELS[card.key] || card.key;
  const question = document.createElement("h3");
  question.textContent = card.question;
  const body = document.createElement("div");
  body.className = "answer";
  for (const paragraph of String(card.response || "").split(/\n{2,}|\n/).map((line) => line.trim()).filter(Boolean)) {
    const p = document.createElement("p");
    p.innerHTML = escapeHtml(paragraph).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    body.appendChild(p);
  }
  article.append(key, question, body);
  return article;
}

function parseSummary(summary) {
  const match = String(summary || "").match(
    /Start close ([\d.]+), last close ([\d.]+), change ([+-]?[\d.]+)%\. High ([\d.]+), low ([\d.]+)\. Last 10 closes: (.*)$/
  );
  if (!match) return null;
  const points = match[6].replace(/\.$/, "").split(", ").map((part) => {
    const [date, price] = part.split(": ");
    return { date: (date || "").trim(), close: Number(String(price || "").replace(/\.$/, "")) };
  }).filter((point) => point.date && Number.isFinite(point.close));
  return {
    start: match[1],
    last: match[2],
    change: match[3],
    high: match[4],
    low: match[5],
    closes: points,
  };
}

function parseNews(block) {
  const groups = { Corporate: [], Sector: [], Macro: [] };
  let current = "Corporate";
  for (const rawLine of String(block || "").split("\n")) {
    const line = rawLine.trim();
    const heading = line.replace(/:$/, "");
    if (groups[heading] && line.endsWith(":")) {
      current = heading;
      continue;
    }
    if (!line.startsWith("- ")) continue;
    const text = line.slice(2).trim();
    if (!text || text.toLowerCase() === "none in this window.") continue;
    groups[current].push(parseHeadline(text));
  }
  return groups;
}

function parseHeadline(text) {
  const dated = text.match(/^(.*?)(?:\s+\[([^\]]+)\])$/);
  const body = (dated ? dated[1] : text).trim();
  const sourced = body.match(/^(.*)\s+\(([^)]+)\)$/);
  return {
    title: (sourced ? sourced[1] : body).trim(),
    source: sourced ? sourced[2].trim() : "",
    when: dated ? dated[2].trim() : "",
  };
}

function formatDay(value) {
  const date = new Date(`${value}T00:00:00`);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

function shortStamp(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

async function readJson(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    const message = typeof detail === "string" ? detail : "Request failed";
    throw new Error(message);
  }
  return data;
}

function setStatus(message, isError) {
  status.textContent = message || "";
  status.classList.toggle("error", Boolean(isError));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}
