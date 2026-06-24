/* Plotly chart helpers. All charts degrade gracefully when data is empty.
   Charts are theme-aware: transparent backgrounds + font/grid colors read from
   the current theme so they look right in both light and dark mode. */

const PF_COLORS = {
  primary: '#6366f1', success: '#22c55e', warning: '#d97706',
  danger: '#dc2626', info: '#06b6d4', orange: '#ea580c', slate: '#94a3b8'
};

const SEVERITY_COLORS = {
  critical: '#dc2626', high: '#ea580c', medium: '#d97706', low: '#64748b'
};

const PF_CONFIG = { responsive: true, displayModeBar: false };

// Build a theme-aware Plotly layout merged with chart-specific overrides.
function pfLayout(extra) {
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  const font = dark ? '#94a3b8' : '#64748b';
  const grid = dark ? 'rgba(148,163,184,.15)' : 'rgba(100,116,139,.15)';
  const base = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, system-ui, sans-serif', color: font, size: 12 },
    xaxis: { gridcolor: grid, zerolinecolor: grid },
    yaxis: { gridcolor: grid, zerolinecolor: grid },
  };
  // shallow-merge axis sub-objects so callers can add titles without losing grid colors
  const out = Object.assign({}, base, extra);
  if (extra && extra.xaxis) out.xaxis = Object.assign({}, base.xaxis, extra.xaxis);
  if (extra && extra.yaxis) out.yaxis = Object.assign({}, base.yaxis, extra.yaxis);
  return out;
}

function emptyChart(id, msg) {
  Plotly.purge(id);
  document.getElementById(id).innerHTML =
    `<div class="text-muted text-center pt-5">${msg || 'No data yet.'}</div>`;
}

function renderForecastChart(id, dates, values, lower, upper, level) {
  if (!dates || !dates.length) return emptyChart(id, 'Run Forecast to see demand.');
  const traces = [];
  // Prediction-interval band (fan chart) drawn first, behind the median line.
  if (lower && upper && lower.length === dates.length) {
    const pct = level ? Math.round(level * 100) : 80;
    traces.push({
      x: dates.concat([...dates].reverse()),
      y: upper.concat([...lower].reverse()),
      fill: 'toself', fillcolor: 'rgba(99,102,241,.18)',
      line: { color: 'rgba(0,0,0,0)' }, hoverinfo: 'skip',
      name: `${pct}% interval`, type: 'scatter'
    });
  }
  traces.push({
    x: dates, y: values, type: 'scatter', mode: 'lines+markers',
    line: { color: PF_COLORS.primary, width: 2.5, shape: 'spline' },
    marker: { size: 5 }, name: 'Forecast (P50)'
  });
  Plotly.newPlot(id, traces, pfLayout({
    margin: { t: 10, r: 10, b: 40, l: 45 },
    xaxis: { title: '' }, yaxis: { title: 'Units', rangemode: 'tozero' },
    showlegend: !!(lower && upper),
    legend: { orientation: 'h', y: 1.12 }
  }), PF_CONFIG);
}

function renderFeatureImportance(id, importances) {
  if (!importances || !importances.length) return emptyChart(id, 'Train models to see importances.');
  const top = importances.slice(0, 10).reverse();
  Plotly.newPlot(id, [{
    x: top.map(d => d.importance), y: top.map(d => d.feature),
    type: 'bar', orientation: 'h', marker: { color: PF_COLORS.success }
  }], pfLayout({
    margin: { t: 10, r: 10, b: 40, l: 130 },
    xaxis: { title: 'Importance' }
  }), PF_CONFIG);
}

function renderSeverityChart(id, severity) {
  const labels = Object.keys(severity || {});
  const values = labels.map(k => severity[k]);
  if (!values.reduce((a, b) => a + b, 0)) return emptyChart(id, 'No alerts yet.');
  Plotly.newPlot(id, [{
    labels, values, type: 'pie', hole: .6,
    marker: { colors: labels.map(l => SEVERITY_COLORS[l] || PF_COLORS.slate),
              line: { color: 'rgba(0,0,0,0)', width: 2 } },
    textinfo: 'label+value'
  }], pfLayout({ margin: { t: 10, r: 10, b: 10, l: 10 }, showlegend: false }), PF_CONFIG);
}

function renderCategoryChart(id, catStock) {
  const labels = Object.keys(catStock || {});
  if (!labels.length) return emptyChart(id, 'No stock data.');
  const values = labels.map(k => catStock[k]);
  Plotly.newPlot(id, [{
    x: values, y: labels, type: 'bar', orientation: 'h',
    marker: { color: PF_COLORS.info }
  }], pfLayout({
    margin: { t: 10, r: 10, b: 40, l: 120 },
    xaxis: { title: 'Total stock' }
  }), PF_CONFIG);
}

function renderModelChart(id, comparison) {
  if (!comparison || !comparison.length) return emptyChart(id, 'Train models first.');
  const names = comparison.map(c => c.model_name);
  Plotly.newPlot(id, [
    { x: names, y: comparison.map(c => c.mae), name: 'MAE', type: 'bar', marker: { color: PF_COLORS.primary } },
    { x: names, y: comparison.map(c => c.rmse), name: 'RMSE', type: 'bar', marker: { color: PF_COLORS.orange } }
  ], pfLayout({
    barmode: 'group', margin: { t: 10, r: 10, b: 40, l: 45 },
    legend: { orientation: 'h', y: 1.15 }
  }), PF_CONFIG);
}
