/* Front-end logic: API calls, toasts, table rendering, page initializers. */

// --------------------------------------------------------------------------
// Utilities
// --------------------------------------------------------------------------
async function api(path, opts) {
  const res = await fetch(path, opts);
  const json = await res.json().catch(() => ({}));
  if (!res.ok || json.status === 'error') {
    throw new Error(json.message || `Request failed (${res.status})`);
  }
  return json;
}

function toast(message, type = 'info', timeout = 4000) {
  const area = document.getElementById('toast-area');
  if (!area) return alert(message);
  const el = document.createElement('div');
  el.className = `pf-toast ${type}`;
  el.textContent = message;
  area.appendChild(el);
  if (timeout) setTimeout(() => el.remove(), timeout);
  return el;
}

function severityBadge(sev) {
  return `<span class="badge badge-${sev || 'low'}">${(sev || 'low').toUpperCase()}</span>`;
}

function riskSpan(risk) {
  if (!risk) return '–';
  return `<span class="risk-${risk}">${risk}</span>`;
}

function fmt(n, d = 0) {
  if (n === null || n === undefined || isNaN(n)) return '–';
  return Number(n).toFixed(d);
}

// Generic action button handler (Seed / Train / Forecast)
async function runAction(path, workingMsg) {
  const t = toast(workingMsg || 'Working…', 'working', 0);
  try {
    const res = await api(path, { method: 'POST' });
    t.remove();
    toast(res.message || 'Done.', 'success');
    // Refresh whatever page we're on after an action.
    setTimeout(() => { if (window.__pfRefresh) window.__pfRefresh(); }, 400);
  } catch (e) {
    t.remove();
    toast(e.message, 'error', 7000);
  }
}

async function updateMongoStatus() {
  const badge = document.getElementById('mongo-status');
  if (!badge) return;
  try {
    const res = await api('/api/health');
    if (res.mongo_available) {
      badge.className = 'badge bg-success';
      badge.textContent = 'MongoDB connected';
    } else {
      badge.className = 'badge bg-warning text-dark';
      badge.textContent = 'CSV fallback mode';
    }
  } catch {
    badge.className = 'badge bg-danger';
    badge.textContent = 'offline';
  }
}

// --------------------------------------------------------------------------
// Dashboard page
// --------------------------------------------------------------------------
async function initDashboard() {
  updateMongoStatus();
  window.__pfRefresh = loadDashboard;
  await loadDashboard();
}

async function loadDashboard() {
  try {
    const summary = (await api('/api/dashboard-summary')).data;
    document.getElementById('card-medicines').textContent = summary.total_medicines;
    document.getElementById('card-lowstock').textContent = summary.low_stock_alerts;
    document.getElementById('card-expiry').textContent = summary.expiry_risk_medicines;
    document.getElementById('card-orderqty').textContent = summary.total_recommended_order_quantity;
    document.getElementById('card-model').textContent = summary.best_model || '–';
    document.getElementById('card-mae').textContent = summary.best_model_mae != null ? fmt(summary.best_model_mae, 2) : '–';

    renderSeverityChart('chart-severity', summary.alert_severity);
    renderCategoryChart('chart-category', summary.category_stock);
  } catch (e) { toast(e.message, 'error'); }

  // Aggregate forecast chart
  try {
    const fc = (await api('/api/forecasts')).data;
    const agg = {};
    (fc.forecasts || []).forEach(f => {
      agg[f.forecast_date] = (agg[f.forecast_date] || 0) + Number(f.predicted_quantity);
    });
    const dates = Object.keys(agg).sort();
    renderForecastChart('chart-forecast', dates, dates.map(d => agg[d]));
    renderRestockTable(fc.optimization || []);
    renderExpiryTable(fc.optimization || []);
  } catch (e) { /* charts already show empty state */ }

  // Model comparison chart
  try {
    const mc = (await api('/api/model-comparison')).data;
    renderModelChart('chart-models', mc.comparison || []);
  } catch (e) { /* empty */ }

  // Recent alerts
  try {
    const alerts = (await api('/api/alerts')).data || [];
    const body = document.getElementById('tbl-alerts');
    if (!alerts.length) { body.innerHTML = '<tr><td colspan="4" class="text-muted">No alerts yet.</td></tr>'; }
    else body.innerHTML = alerts.slice(0, 8).map(a => `<tr>
      <td>${severityBadge(a.severity)}</td><td><code>${a.alert_type}</code></td>
      <td>${a.message}</td><td class="small text-muted">${(a.created_at || '').slice(0, 19).replace('T', ' ')}</td></tr>`).join('');
  } catch (e) { /* empty */ }
}

function renderRestockTable(opt) {
  const body = document.getElementById('tbl-restock');
  const rows = opt.filter(o => o.recommended_order_quantity > 0)
    .sort((a, b) => b.recommended_order_quantity - a.recommended_order_quantity).slice(0, 8);
  if (!rows.length) { body.innerHTML = '<tr><td colspan="4" class="text-muted">No restock needed.</td></tr>'; return; }
  body.innerHTML = rows.map(o => `<tr>
    <td>${o.medicine_name}</td><td>${o.current_stock}</td>
    <td>${fmt(o.reorder_point)}</td>
    <td><span class="badge bg-info">${o.recommended_order_quantity}</span></td></tr>`).join('');
}

function renderExpiryTable(opt) {
  const body = document.getElementById('tbl-expiry');
  const rows = opt.filter(o => ['high', 'expired'].includes(o.expiry_risk))
    .sort((a, b) => (a.days_to_expiry ?? 999) - (b.days_to_expiry ?? 999)).slice(0, 8);
  if (!rows.length) { body.innerHTML = '<tr><td colspan="4" class="text-muted">No near-expiry items.</td></tr>'; return; }
  body.innerHTML = rows.map(o => `<tr>
    <td>${o.medicine_name}</td><td>${(o.expiry_date || '').slice(0, 10)}</td>
    <td>${o.days_to_expiry}</td><td>${riskSpan(o.expiry_risk)}</td></tr>`).join('');
}

// --------------------------------------------------------------------------
// Medicines page
// --------------------------------------------------------------------------
let __medicines = [];
async function initMedicines() {
  updateMongoStatus();
  window.__pfRefresh = loadMedicines;
  await loadMedicines();
  const search = document.getElementById('med-search');
  if (search) search.addEventListener('input', () => renderMedicines(search.value.toLowerCase()));
}

async function loadMedicines() {
  try {
    __medicines = (await api('/api/medicines')).data || [];
    renderMedicines('');
  } catch (e) { toast(e.message, 'error'); }
}

function renderMedicines(filter) {
  const body = document.getElementById('medicines-body');
  const rows = __medicines.filter(m =>
    !filter || (m.medicine_name || '').toLowerCase().includes(filter) ||
    (m.category || '').toLowerCase().includes(filter));
  if (!rows.length) { body.innerHTML = '<tr><td colspan="12" class="text-muted">No medicines found.</td></tr>'; return; }
  body.innerHTML = rows.map(m => `<tr>
    <td><code>${m.medicine_id}</code></td><td>${m.medicine_name}</td><td>${m.category}</td>
    <td>$${fmt(m.unit_price, 2)}</td><td>${m.supplier || '–'}</td><td>${m.storage_type || '–'}</td>
    <td>${m.current_stock ?? '–'}</td><td>${fmt(m.reorder_point)}</td>
    <td>${m.recommended_order_quantity ?? '–'}</td>
    <td>${m.days_until_stockout ?? '–'}</td>
    <td>${(m.expiry_date || '').slice(0, 10)}</td><td>${riskSpan(m.expiry_risk)}</td></tr>`).join('');
}

// --------------------------------------------------------------------------
// Forecasts page
// --------------------------------------------------------------------------
let __forecastData = { forecasts: [], optimization: [] };
async function initForecasts() {
  updateMongoStatus();
  window.__pfRefresh = loadForecasts;
  await loadForecasts();
  const sel = document.getElementById('forecast-medicine');
  if (sel) sel.addEventListener('change', () => drawMedicineForecast(sel.value));
}

async function loadForecasts() {
  try {
    __forecastData = (await api('/api/forecasts')).data || { forecasts: [], optimization: [] };
    const sel = document.getElementById('forecast-medicine');
    const ids = [...new Set(__forecastData.forecasts.map(f => f.medicine_id))];
    const nameById = {};
    (__forecastData.optimization || []).forEach(o => { nameById[o.medicine_id] = o.medicine_name; });
    sel.innerHTML = ids.length
      ? ids.map(id => `<option value="${id}">${nameById[id] || id}</option>`).join('')
      : '<option>Run Forecast first</option>';
    if (ids.length) drawMedicineForecast(ids[0]);
    else emptyChart('chart-medicine-forecast', 'Run Forecast to see results.');
    renderForecastTable();
  } catch (e) { toast(e.message, 'error'); }
}

function drawMedicineForecast(mid) {
  const rows = __forecastData.forecasts.filter(f => f.medicine_id === mid)
    .sort((a, b) => a.forecast_date.localeCompare(b.forecast_date));
  renderForecastChart('chart-medicine-forecast',
    rows.map(r => r.forecast_date), rows.map(r => Number(r.predicted_quantity)));
}

function renderForecastTable() {
  const body = document.getElementById('forecast-table-body');
  const opt = __forecastData.optimization || [];
  // confidence map from forecasts
  const conf = {};
  __forecastData.forecasts.forEach(f => { conf[f.medicine_id] = f.confidence_level; });
  if (!opt.length) { body.innerHTML = '<tr><td colspan="7" class="text-muted">Run Forecast to populate.</td></tr>'; return; }
  body.innerHTML = opt.sort((a, b) => b.forecasted_demand - a.forecasted_demand).map(o => {
    const c = conf[o.medicine_id] || 'low';
    const cb = c === 'high' ? 'success' : c === 'medium' ? 'warning text-dark' : 'secondary';
    return `<tr><td>${o.medicine_name}</td><td>${fmt(o.average_daily_demand, 2)}</td>
      <td>${fmt(o.forecasted_demand)}</td><td>${fmt(o.safety_stock)}</td>
      <td>${fmt(o.reorder_point)}</td>
      <td><span class="badge bg-info">${o.recommended_order_quantity}</span></td>
      <td><span class="badge bg-${cb}">${c}</span></td></tr>`;
  }).join('');
}

// --------------------------------------------------------------------------
// Alerts page
// --------------------------------------------------------------------------
let __alerts = [];
async function initAlerts() {
  updateMongoStatus();
  window.__pfRefresh = loadAlerts;
  await loadAlerts();
  document.querySelectorAll('#alert-filters button').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#alert-filters button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderAlerts(btn.dataset.filter);
    });
  });
}

async function loadAlerts() {
  try { __alerts = (await api('/api/alerts')).data || []; renderAlerts('all'); }
  catch (e) { toast(e.message, 'error'); }
}

function renderAlerts(filter) {
  const body = document.getElementById('alerts-body');
  const rows = __alerts.filter(a => filter === 'all' || a.severity === filter);
  if (!rows.length) { body.innerHTML = '<tr><td colspan="5" class="text-muted">No alerts.</td></tr>'; return; }
  body.innerHTML = rows.map(a => `<tr>
    <td>${severityBadge(a.severity)}</td><td><code>${a.alert_type}</code></td>
    <td>${a.medicine_id}</td><td>${a.message}</td>
    <td class="small text-muted">${(a.created_at || '').slice(0, 19).replace('T', ' ')}</td></tr>`).join('');
}

// --------------------------------------------------------------------------
// Model comparison page
// --------------------------------------------------------------------------
async function initModelComparison() {
  updateMongoStatus();
  window.__pfRefresh = loadModelComparison;
  await loadModelComparison();
}

async function loadModelComparison() {
  try {
    const mc = (await api('/api/model-comparison')).data;
    const comp = mc.comparison || [];
    renderModelChart('chart-comparison', comp);

    const body = document.getElementById('comparison-body');
    if (!comp.length) { body.innerHTML = '<tr><td colspan="6" class="text-muted">Train models first.</td></tr>'; }
    else body.innerHTML = comp.map(c => {
      const rowCls = c.is_best ? 'table-success' : (c.is_baseline ? 'table-light' : '');
      const nameCell = c.is_baseline ? `${c.model_name} <span class="badge bg-secondary">baseline</span>` : c.model_name;
      let skill = '–';
      if (!c.is_baseline && c.skill_vs_naive != null) {
        const pct = (c.skill_vs_naive * 100).toFixed(1) + '%';
        const cls = c.skill_vs_naive > 0 ? 'text-success' : 'text-danger';
        skill = `<span class="${cls}">${pct}</span>`;
      }
      const sel = c.is_best ? '<i class="bi bi-trophy-fill text-success"></i> Selected' : '';
      return `<tr class="${rowCls}"><td>${nameCell}</td><td>${fmt(c.mae, 3)}</td>
        <td>${fmt(c.rmse, 3)}</td><td>${fmt(c.mape, 2)}</td><td>${skill}</td><td>${sel}</td></tr>`;
    }).join('');

    const box = document.getElementById('selected-model-box');
    if (mc.selected) {
      const s = mc.selected;
      box.innerHTML = `<i class="bi bi-trophy-fill text-success" style="font-size:2.5rem"></i>
        <h4 class="mt-2">${s.best_model}</h4>
        <p class="text-muted mb-1">MAE: <strong>${fmt(s.metrics.mae, 3)}</strong></p>
        <p class="text-muted mb-1">RMSE: ${fmt(s.metrics.rmse, 3)} · MAPE: ${fmt(s.metrics.mape, 2)}%</p>
        <p class="small text-muted">Selected ${(s.generated_at || '').slice(0, 19).replace('T', ' ')}</p>`;
    }
  } catch (e) { toast(e.message, 'error'); }
}

// --------------------------------------------------------------------------
// Upload page
// --------------------------------------------------------------------------
async function uploadCsv(path, inputId) {
  const input = document.getElementById(inputId);
  if (!input.files.length) return toast('Choose a CSV file first.', 'error');
  const fd = new FormData();
  fd.append('file', input.files[0]);
  const t = toast('Uploading…', 'working', 0);
  try {
    const res = await api(path, { method: 'POST', body: fd });
    t.remove();
    toast(res.message || 'Uploaded.', 'success');
    document.getElementById('upload-result').innerHTML =
      `<div class="alert alert-success">${res.message} Now click <strong>Train</strong> then <strong>Forecast</strong>.</div>`;
  } catch (e) { t.remove(); toast(e.message, 'error', 7000); }
}
