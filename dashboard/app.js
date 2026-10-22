// dashboard/app.js — Semantic 3DGS viewer (real API only)
/* global THREE, Chart */

let scene, camera, renderer, controls, pointsMesh = null;
let sceneData = null;          // {xyz, labels, confidence, label_names, mode}
let ablationChart = null;
const visibleObjects = new Set(); // empty = show all
let minConfidence = 0.0;

const PALETTE = [
  [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
  [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
  [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
];
const UNKNOWN_COLOR = [120, 120, 120];

// ------------------------------------------------------------ helpers
const $ = (id) => document.getElementById(id);

function toast(msg, ok = false) {
  const el = $('toast');
  el.textContent = msg;
  el.classList.toggle('ok', ok);
  el.hidden = false;
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.hidden = true; }, 5000);
}

function labelColor(label) {
  if (label < 0) return UNKNOWN_COLOR;
  return PALETTE[label % PALETTE.length];
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// ------------------------------------------------------------ three.js
function initThree() {
  const container = $('canvas-container');
  const w = container.clientWidth, h = container.clientHeight;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0d12);

  camera = new THREE.PerspectiveCamera(55, w / h, 0.05, 500);
  camera.position.set(0, 1.2, 5);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  const grid = new THREE.GridHelper(10, 20, 0x1e293b, 0x0f172a);
  scene.add(grid);
  scene.add(new THREE.AmbientLight(0xffffff, 0.9));

  window.addEventListener('resize', () => {
    const cw = container.clientWidth, ch = container.clientHeight;
    camera.aspect = cw / ch;
    camera.updateProjectionMatrix();
    renderer.setSize(cw, ch);
  });

  (function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  })();
}

function sceneCenter(xyz) {
  if (!xyz.length) return [0, 0, 0];
  const n = xyz.length;
  const c = [0, 0, 0];
  for (const p of xyz) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; }
  return c.map(v => v / n);
}

function updatePointCloud() {
  if (!sceneData) return;
  const { xyz, labels, confidence } = sceneData;

  if (pointsMesh) {
    scene.remove(pointsMesh);
    pointsMesh.geometry.dispose();
    pointsMesh.material.dispose();
    pointsMesh = null;
  }

  const positions = [], colors = [];
  let shown = 0;
  for (let i = 0; i < xyz.length; i++) {
    if (confidence[i] < minConfidence) continue;
    const lab = labels[i];
    if (visibleObjects.size > 0 && !visibleObjects.has(lab)) continue;
    positions.push(xyz[i][0], xyz[i][1], xyz[i][2]);
    const c = labelColor(lab);
    colors.push(c[0] / 255, c[1] / 255, c[2] / 255);
    shown++;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  const mat = new THREE.PointsMaterial({ size: 0.02, vertexColors: true, opacity: 0.92, transparent: true });
  pointsMesh = new THREE.Points(geo, mat);
  scene.add(pointsMesh);

  $('stat-shown').textContent = shown;
}

function frameCamera(view) {
  if (!sceneData) return;
  const c = sceneCenter(sceneData.xyz);
  controls.target.set(c[0], c[1], c[2]);
  if (view === 'top') camera.position.set(c[0], Math.max(c[1] + 4, 4), c[2] + 0.001);
  else if (view === 'side') camera.position.set(c[0] + 5, c[1], c[2]);
  else camera.position.set(c[0] + 2.2, c[1] + 1.4, c[2] + 3.4);
  controls.update();
}

// ------------------------------------------------------------ data
function renderScene(scene) {
  sceneData = scene;
  const names = scene.label_names || {};

  $('mode-line').textContent = scene.mode === 'single_image_2.5d'
    ? 'single-image 2.5D — real segmentation, plane-assumption depth (not reconstruction)'
    : 'multi-view 2D→3D label transfer on Gaussian Splats';

  const list = $('object-list');
  list.innerHTML = '';
  const ids = Object.keys(names).map(Number).sort((a, b) => a - b);
  visibleObjects.clear();

  const counts = {};
  for (const l of scene.labels) counts[l] = (counts[l] || 0) + 1;

  for (const id of ids) {
    const li = document.createElement('li');
    const dot = document.createElement('span');
    dot.className = 'dot';
    const c = labelColor(id);
    dot.style.background = `rgb(${c[0]},${c[1]},${c[2]})`;
    const input = document.createElement('input');
    input.type = 'text';
    input.value = names[id];
    input.title = 'click to rename';
    input.addEventListener('change', async () => {
      try {
        const r = await api('/api/labels/rename', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id, label: input.value }),
        });
        sceneData = r.scene;
        toast('renamed', true);
      } catch (e) { toast(`rename failed: ${e.message}`); input.value = names[id]; }
    });
    const count = document.createElement('span');
    count.className = 'count';
    count.textContent = counts[id] || 0;
    const vis = document.createElement('input');
    vis.type = 'checkbox';
    vis.checked = true;
    vis.title = 'show/hide';
    vis.addEventListener('change', () => {
      if (vis.checked) visibleObjects.delete(id); else visibleObjects.add(id);
      updatePointCloud();
    });
    li.append(vis, dot, input, count);
    list.appendChild(li);
  }

  $('stat-points').textContent = scene.n_points;
  $('stat-objects').textContent = ids.length;
  $('badge-mode').textContent = `mode: ${scene.mode}`;
  $('badge-mode').className = 'badge ' + (scene.mode === 'single_image_2.5d' ? 'warn' : 'ok');

  updatePointCloud();
  frameCamera('orbit');
}

function renderMetrics(m) {
  // ablation
  const ablation = m.ablation || [];
  const labels = ablation.map(a => (a.name || a.strategy || '?').replace(/_/g, ' '));
  const rates = ablation.map(a => +(a.label_rate * 100).toFixed(1));

  if (ablationChart) ablationChart.destroy();
  ablationChart = new Chart($('ablation-chart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{ data: rates, backgroundColor: ['#94a3b8aa', '#3b82f6aa', '#10b981cc'], borderWidth: 0 }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { y: { beginAtZero: true, max: 100, ticks: { color: '#9aa3b5' }, grid: { color: '#ffffff10' } },
                x: { ticks: { color: '#9aa3b5' }, grid: { display: false } } },
      plugins: { legend: { display: false } },
    },
  });

  const tbody = $('ablation-table').querySelector('tbody');
  tbody.innerHTML = '';
  ablation.forEach(a => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${(a.name || a.strategy).replace(/_/g, ' ')}</td>
      <td class="num">${a.n_labeled}/${a.n_gaussians}</td>
      <td class="num">${(a.label_rate * 100).toFixed(1)}%</td>`;
    tbody.appendChild(tr);
  });

  // evaluation
  const evTbody = $('eval-table').querySelector('tbody');
  evTbody.innerHTML = '';
  const evaluation = m.evaluation || {};
  const evalObjs = Object.keys(evaluation);
  if (!evalObjs.length) {
    evTbody.innerHTML = '<tr><td colspan="4" style="color:var(--muted)">no eval_results.json yet (annotate GT, then run scripts/run_evaluation.py)</td></tr>';
  } else {
    evalObjs.forEach(name => {
      const s = evaluation[name];
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${name}</td>
        <td class="num">${(+s.mean_iou).toFixed(3)}</td>
        <td class="num">${(+s.mean_precision).toFixed(3)}</td>
        <td class="num">${(+s.mean_recall).toFixed(3)}</td>`;
      evTbody.appendChild(tr);
    });
  }

  // run metrics
  const rm = m.run_metrics;
  $('metrics-pre').textContent = rm ? JSON.stringify(rm, null, 2) :
    (m.metadata && m.metadata.mode ? `mode: ${m.metadata.mode} (single-image run — no multi-view metrics)` : 'no run yet');
}

async function refreshAll() {
  const st = await api('/api/status');
  const b = st.backends || {};
  $('badge-backend').textContent = `segmenter: ${b.sam2 ? 'SAM2' : 'classical (CPU)'}`;
  $('badge-backend').className = 'badge ' + (b.sam2 ? 'ok' : '');
  if (!b.torch) {
    const opt = $('backend-select').querySelector('option[value="sam2"]');
    opt.disabled = true; opt.textContent = 'SAM2 (torch missing)';
  }

  try {
    renderScene(await api('/api/scene'));
  } catch (e) {
    sceneData = null;
    $('mode-line').textContent = 'no semantic outputs yet — segment an image or run the pipeline';
    $('stat-points').textContent = '–';
    $('stat-shown').textContent = '–';
    $('stat-objects').textContent = '–';
  }
  renderMetrics(await api('/api/metrics'));
}

// ------------------------------------------------------------ events
function bindEvents() {
  $('conf-slider').addEventListener('input', (e) => {
    minConfidence = parseFloat(e.target.value);
    $('conf-val').textContent = minConfidence.toFixed(2);
    updatePointCloud();
  });

  $('btn-choose').addEventListener('click', () => $('file-input').click());
  $('file-input').addEventListener('change', (e) => {
    const f = e.target.files[0];
    if (!f) return;
    window._pendingFile = f;
    const img = $('preview');
    img.src = URL.createObjectURL(f);
    img.hidden = false;
    $('btn-segment').disabled = false;
  });

  $('btn-segment').addEventListener('click', async () => {
    const f = window._pendingFile;
    if (!f) return;
    const status = $('segment-status');
    status.className = 'status';
    status.textContent = 'segmenting (real backend, may take a moment)…';
    $('btn-segment').disabled = true;
    try {
      const dataURI = await new Promise((resolve, reject) => {
        const r = new FileReader();
        r.onload = () => resolve(r.result);
        r.onerror = reject;
        r.readAsDataURL(f);
      });
      const r = await api('/api/segment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: dataURI,
          filename: f.name,
          backend: $('backend-select').value,
          max_objects: parseInt($('max-objects').value, 10) || 8,
        }),
      });
      status.className = 'status ok';
      status.textContent = `${r.n_objects} objects · backend=${r.backend} · ${r.mode}`;
      renderScene(r.scene);
      toast('segmentation complete', true);
    } catch (e) {
      status.className = 'status err';
      status.textContent = e.message;
      toast(`segmentation failed: ${e.message}`);
    } finally {
      $('btn-segment').disabled = false;
    }
  });

  $('btn-run').addEventListener('click', async () => {
    const status = $('pipeline-status');
    status.className = 'status';
    status.textContent = 'running association…';
    try {
      const r = await api('/api/pipeline/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy: $('strategy-select').value,
          min_views: parseInt($('min-views').value, 10) || 1,
        }),
      });
      status.className = 'status ok';
      status.textContent = `${(r.metrics.label_rate * 100).toFixed(1)}% labeled (${r.metrics.n_labeled}/${r.metrics.n_gaussians})`;
      renderScene(await api('/api/scene'));
      renderMetrics(await api('/api/metrics'));
    } catch (e) {
      status.className = 'status err';
      status.textContent = e.message;
      toast(`pipeline failed: ${e.message}`);
    }
  });

  $('btn-ablation').addEventListener('click', async () => {
    const status = $('pipeline-status');
    status.className = 'status';
    status.textContent = 'running ablation sweep…';
    try {
      await api('/api/pipeline/ablation', { method: 'POST' });
      status.className = 'status ok';
      status.textContent = 'ablation complete';
      renderScene(await api('/api/scene'));
      renderMetrics(await api('/api/metrics'));
    } catch (e) {
      status.className = 'status err';
      status.textContent = e.message;
      toast(`ablation failed: ${e.message}`);
    }
  });

  $('btn-show-all').addEventListener('click', () => { visibleObjects.clear(); updatePointCloud(); });
  $('btn-hide-all').addEventListener('click', () => {
    // hide everything except nothing = show nothing (reset with Show all)
    Object.keys(sceneData?.label_names || {}).forEach(id => visibleObjects.add(Number(id)));
    updatePointCloud();
  });

  $('btn-view-orbit').addEventListener('click', () => frameCamera('orbit'));
  $('btn-view-top').addEventListener('click', () => frameCamera('top'));
  $('btn-view-side').addEventListener('click', () => frameCamera('side'));
}

// ------------------------------------------------------------ boot
document.addEventListener('DOMContentLoaded', async () => {
  initThree();
  bindEvents();
  try {
    await refreshAll();
  } catch (e) {
    toast(`server unreachable: ${e.message}`);
  }
});
