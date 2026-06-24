/* Plotly chart helpers. All charts degrade gracefully when data is empty. */

const PF_COLORS = {
  primary: '#2563eb', success: '#16a34a', warning: '#ca8a04',
  danger: '#dc2626', info: '#0891b2', orange: '#ea580c', slate: '#64748b'
};

const SEVERITY_COLORS = {
  critical: '#dc2626', high: '#ea580c', medium: '#ca8a04', low: '#64748b'
};

function emptyChart(id, msg) {
  Plotly.purge(id);
  document.getElementById(id).innerHTML =
    `<div class="text-muted text-center pt-5">${msg || 'No data yet.'}</div>`;
}

function renderForecastChart(id, dates, values) {
  if (!dates || !dates.length) return emptyChart(id, 'Run Forecast to see demand.');
  Plotly.newPlot(id, [{
    x: dates, y: values, type: 'scatter', mode: 'lines+markers',
    line: { color: PF_COLORS.primary, width: 2 }, fill: 'tozeroy',
    fillcolor: 'rgba(37,99,235,.1)', name: 'Forecast demand'
  }], {
    margin: { t: 10, r: 10, b: 40, l: 45 },
    xaxis: { title: '' }, yaxis: { title: 'Units' },
    showlegend: false
  }, { responsive: true, displayModeBar: false });
}

function renderSeverityChart(id, severity) {
  const labels = Object.keys(severity || {});
  const values = labels.map(k => severity[k]);
  if (!values.reduce((a, b) => a + b, 0)) return emptyChart(id, 'No alerts yet.');
  Plotly.newPlot(id, [{
    labels, values, type: 'pie', hole: .55,
    marker: { colors: labels.map(l => SEVERITY_COLORS[l] || PF_COLORS.slate) },
    textinfo: 'label+value'
  }], { margin: { t: 10, r: 10, b: 10, l: 10 }, showlegend: false },
  { responsive: true, displayModeBar: false });
}

function renderCategoryChart(id, catStock) {
  const labels = Object.keys(catStock || {});
  if (!labels.length) return emptyChart(id, 'No stock data.');
  const values = labels.map(k => catStock[k]);
  Plotly.newPlot(id, [{
    x: values, y: labels, type: 'bar', orientation: 'h',
    marker: { color: PF_COLORS.info }
  }], {
    margin: { t: 10, r: 10, b: 40, l: 120 },
    xaxis: { title: 'Total stock' }
  }, { responsive: true, displayModeBar: false });
}

function renderModelChart(id, comparison) {
  if (!comparison || !comparison.length) return emptyChart(id, 'Train models first.');
  const names = comparison.map(c => c.model_name);
  Plotly.newPlot(id, [
    { x: names, y: comparison.map(c => c.mae), name: 'MAE', type: 'bar', marker: { color: PF_COLORS.primary } },
    { x: names, y: comparison.map(c => c.rmse), name: 'RMSE', type: 'bar', marker: { color: PF_COLORS.orange } }
  ], {
    barmode: 'group', margin: { t: 10, r: 10, b: 40, l: 45 },
    legend: { orientation: 'h', y: 1.15 }
  }, { responsive: true, displayModeBar: false });
}
