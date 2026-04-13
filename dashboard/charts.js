/**
 * VaakSeva Dashboard - Chart.js visualizations
 *
 * Fetches metrics from /api/metrics and renders:
 * - Daily query trend (line chart)
 * - Pipeline latency breakdown (bar chart)
 * - Top queried schemes (horizontal bar)
 * - Input type distribution (doughnut)
 * - Language distribution (doughnut)
 */

const COLORS = {
  accent: "rgba(108, 99, 255, 0.85)",
  accentBorder: "rgba(108, 99, 255, 1)",
  teal: "rgba(0, 212, 184, 0.85)",
  tealBorder: "rgba(0, 212, 184, 1)",
  coral: "rgba(255, 107, 107, 0.85)",
  yellow: "rgba(255, 209, 102, 0.85)",
  purple: "rgba(180, 80, 255, 0.85)",
  grid: "rgba(255, 255, 255, 0.05)",
  text: "rgba(139, 147, 196, 1)",
};

Chart.defaults.color = COLORS.text;
Chart.defaults.borderColor = "rgba(255, 255, 255, 0.06)";
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;

const charts = {};

function makeLineChart(id, labels, data, label) {
  const ctx = document.getElementById(id).getContext("2d");
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label,
          data,
          fill: true,
          tension: 0.4,
          borderColor: COLORS.accentBorder,
          backgroundColor: "rgba(108, 99, 255, 0.12)",
          pointBackgroundColor: COLORS.accentBorder,
          pointRadius: 4,
          pointHoverRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: COLORS.grid } },
        y: { grid: { color: COLORS.grid }, beginAtZero: true },
      },
    },
  });
}

function makeBarChart(id, labels, data, colors) {
  const ctx = document.getElementById(id).getContext("2d");
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          data,
          backgroundColor: colors || COLORS.accent,
          borderRadius: 6,
          borderSkipped: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: COLORS.grid }, beginAtZero: true },
      },
    },
  });
}

function makeHorizontalBarChart(id, labels, data) {
  const ctx = document.getElementById(id).getContext("2d");
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          data,
          backgroundColor: [
            COLORS.accent, COLORS.teal, COLORS.coral, COLORS.yellow,
            COLORS.purple, COLORS.accent, COLORS.teal, COLORS.coral,
            COLORS.yellow, COLORS.purple,
          ],
          borderRadius: 4,
          borderSkipped: false,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: COLORS.grid }, beginAtZero: true },
        y: { grid: { display: false } },
      },
    },
  });
}

function makeDoughnutChart(id, labels, data) {
  const ctx = document.getElementById(id).getContext("2d");
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [
        {
          data,
          backgroundColor: [COLORS.accent, COLORS.teal, COLORS.coral, COLORS.yellow, COLORS.purple],
          hoverOffset: 8,
          borderWidth: 2,
          borderColor: "#161927",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          position: "bottom",
          labels: { padding: 12, boxWidth: 12 },
        },
      },
      cutout: "65%",
    },
  });
}

function renderMetrics(metrics) {
  const s = metrics.summary || {};
  const l = metrics.latency_stats || {};

  // Summary cards
  setText("stat-total", s.total ?? "--");
  setText("stat-today", `${s.today ?? "--"} today`);
  setText("stat-week", s.week ?? "--");

  const mean = l.total_ms?.mean;
  setText("stat-latency", mean ? `${mean.toFixed(0)} ms` : "--");
  setText("stat-p95", l.total_ms?.p95 ? `P95: ${l.total_ms.p95.toFixed(0)} ms` : "P95: --");

  const errRate = s.error_rate;
  setText("stat-error", errRate != null ? `${errRate.toFixed(1)}%` : "--");
  setText("stat-error-count", `${s.error_count ?? 0} errors`);

  // Daily trend chart
  const daily = metrics.daily_trend || [];
  makeLineChart(
    "daily-chart",
    daily.map((d) => d.date.slice(5)),  // "MM-DD"
    daily.map((d) => d.count),
    "Queries"
  );

  // Latency breakdown chart
  const stages = ["stt_ms", "embedding_ms", "retrieval_ms", "rerank_ms", "llm_ms", "tts_ms"];
  const stageLabels = ["STT", "Embed", "Retrieve", "Rerank", "LLM", "TTS"];
  const stageColors = [COLORS.teal, COLORS.accent, COLORS.yellow, COLORS.purple, COLORS.coral, COLORS.teal];
  const p50s = stages.map((s) => l[s]?.p50 ?? 0);

  makeBarChart("latency-chart", stageLabels, p50s, stageColors);

  // Top schemes
  const schemes = metrics.top_schemes || [];
  makeHorizontalBarChart(
    "schemes-chart",
    schemes.map((x) => x.scheme.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())),
    schemes.map((x) => x.count)
  );

  // Input types doughnut
  const inputTypes = metrics.input_types || {};
  makeDoughnutChart(
    "input-type-chart",
    Object.keys(inputTypes),
    Object.values(inputTypes)
  );

  // Language distribution
  const langs = metrics.languages || {};
  const langLabels = Object.keys(langs).map((l) => {
    const map = { hi: "Hindi", en: "English", "hi-IN": "Hindi (IN)" };
    return map[l] || l;
  });
  makeDoughnutChart("lang-chart", langLabels, Object.values(langs));

  // Update timestamp
  setText("last-updated", `Updated ${new Date().toLocaleTimeString()}`);
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

async function fetchAndRender() {
  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const metrics = await res.json();
    renderMetrics(metrics);
  } catch (err) {
    console.warn("Could not load metrics:", err.message);
    // Show demo data when disconnected
    renderMetrics(getDemoMetrics());
  }
}

function getDemoMetrics() {
  const days = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    days.push({
      date: d.toISOString().slice(0, 10),
      count: Math.floor(Math.random() * 80 + 20),
    });
  }
  return {
    summary: { total: 347, today: 42, week: 214, error_count: 3, error_rate: 0.86 },
    latency_stats: {
      stt_ms: { p50: 980, p95: 1800, p99: 2200, mean: 1050 },
      embedding_ms: { p50: 42, p95: 95, p99: 150, mean: 48 },
      retrieval_ms: { p50: 28, p95: 65, p99: 95, mean: 32 },
      rerank_ms: { p50: 15, p95: 40, p99: 60, mean: 18 },
      llm_ms: { p50: 2100, p95: 3800, p99: 5200, mean: 2400 },
      tts_ms: { p50: 320, p95: 720, p99: 980, mean: 380 },
      total_ms: { p50: 3500, p95: 6200, p99: 8100, mean: 3900 },
    },
    top_schemes: [
      { scheme: "pm_kisan", count: 89 },
      { scheme: "ayushman_bharat", count: 67 },
      { scheme: "mgnregs", count: 52 },
      { scheme: "mudra_yojana", count: 41 },
      { scheme: "ujjwala_yojana", count: 38 },
    ],
    input_types: { text: 218, voice: 129 },
    languages: { hi: 301, en: 46 },
    daily_trend: days,
  };
}

// Initial load + refresh every 30 seconds
fetchAndRender();
setInterval(fetchAndRender, 30000);
