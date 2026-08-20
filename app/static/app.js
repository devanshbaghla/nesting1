import {PartViewer} from './viewer.js';

const $ = (s) => document.querySelector(s);
const fmt = (n) => n.toLocaleString(undefined, {maximumFractionDigits: 0});
const dim = (n) => n.toLocaleString(undefined, {maximumFractionDigits: 2});
let jobId = null, poller = null, viewer = null, selected = null;
let ratio = 0.02, reclassifyTimer = null, pitch = null;

/* ---------- upload ---------- */
const dz = $('#dropzone'), fileInput = $('#file');
$('#browse').onclick = () => fileInput.click();
dz.onclick = (e) => { if (e.target.tagName !== 'BUTTON') fileInput.click(); };
['dragenter','dragover'].forEach(t => dz.addEventListener(t, e => {
  e.preventDefault(); dz.classList.add('over');
}));
['dragleave','drop'].forEach(t => dz.addEventListener(t, e => {
  e.preventDefault(); dz.classList.remove('over');
}));
dz.addEventListener('drop', e => {
  if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; showName(); }
});
fileInput.onchange = showName;
function showName() {
  const f = fileInput.files[0];
  $('#filehint').textContent = f ? `${f.name} — ${(f.size/1e6).toFixed(1)} MB` : 'Open meshes are repaired automatically. Up to 300 MB.';
}

/* Upload goes to the preview, not to the queue: nothing is denoised or nested
   until the part has been looked at. */
$('#upload-form').onsubmit = async (e) => {
  e.preventDefault();
  if (!fileInput.files.length) { alert('Choose an STL first.'); return; }
  const fd = new FormData(e.target);
  fd.set('file', fileInput.files[0]);
  $('#submit').disabled = true;
  $('#submit').textContent = 'Reading…';
  hide('#error-panel'); hide('#results');
  try {
    const r = await fetch('/api/preview', {method: 'POST', body: fd});
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'upload rejected');
    jobId = data.job_id;
    hide('#upload-panel');
    show('#preview-panel');
    renderPreview(data.preview);
    await loadGeometry();
  } catch (err) {
    fail(err.message);
  } finally {
    $('#submit').disabled = false;
    $('#submit').textContent = 'Preview part';
  }
};

/* ---------- preview ---------- */
let lastPreview = null;
function renderPreview(p) {
  lastPreview = p;
  $('#preview-summary').textContent = p.summary;
  $('#preview-summary').className = 'summary' + (p.has_noise ? ' warn' : ' ok');

  const e = p.extents, o = p.object_extents;
  $('#preview-facts').innerHTML = [
    ['File', p.filename],
    ['Triangles', fmt(p.faces)],
    ['Bodies', p.bodies],
    ['Overall L × W × H', `${dim(e[0])} × ${dim(e[1])} × ${dim(e[2])}`],
    ['Object L × W × H', `${dim(o[0])} × ${dim(o[1])} × ${dim(o[2])}`],
    ['Watertight', p.watertight ? 'yes' : 'no'],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');

  ratio = p.ratio;
  $('#ratio').value = p.ratio;
  $('#ratio-out').textContent =
    `${(p.ratio * 100).toFixed(1)}% — under ${dim(p.threshold)}`;
  $('#preview-rule').innerHTML =
    `A body is <b>noise</b> only when it is small <b>both</b> ways: under
     <b>${dim(p.threshold)}</b> across (${(p.ratio * 100).toFixed(1)}% of the
     object's smallest dimension, ${dim(p.object_smallest)}) <b>and</b> under
     <b>${dim(p.area_limit)}</b> of surface. Anything carrying real surface is
     structure and stays, however small its box. Distance is context only.`;

  const frags = p.bodies > 1 ? p.fragments : [];
  $('#frag-list').innerHTML = frags.length ? frags
    .slice()
    .sort((a, b) => (b.is_noise - a.is_noise) || (b.largest - a.largest))
    .map(f => `
      <button type="button" class="frag ${f.is_noise ? 'is-noise' : ''}"
              data-body="${f.index}" title="${f.reason} — click to highlight">
        <span class="dot"></span>
        <span class="frag-id">body ${f.index}</span>
        <span class="frag-size">${dim(f.extents[0])} × ${dim(f.extents[1])} × ${dim(f.extents[2])}</span>
        <span class="frag-max">${f.gap === null ? '' : 'gap ' + dim(f.gap)}</span>
        <span class="frag-tag">${f.is_noise ? 'NOISE' : 'keep'}</span>
      </button>`).join('') : '';
  selected = null;
  $('#frag-list').querySelectorAll('.frag').forEach(el => {
    el.onclick = () => selectBody(Number(el.dataset.body), el);
  });

  renderPitch(p);

  const btn = $('#denoise');
  btn.disabled = !p.has_noise;
  btn.textContent = p.has_noise
    ? `Remove ${p.noise_bodies} noise ${p.noise_bodies === 1 ? 'body' : 'bodies'}`
    : 'No noise to remove';
  btn.title = p.has_noise
    ? `${fmt(p.noise_faces)} triangles will be deleted from the file to be nested`
    : '';
  $('#toggle-noise').disabled = !p.has_noise;
  $('#denoise-note').textContent = p.decimated_to
    ? `Preview decimated to ${fmt(p.decimated_to)} triangles for drawing; nesting uses the full mesh.`
    : '';
}

/* The translation search allocates one array per orientation, sized by the
   part and the pitch. On a large part the engine default is tens of GB, so the
   cost of every option is shown and the unaffordable ones are disabled. */
function renderPitch(p) {
  const sel = $('#pitch');
  const opts = p.pitch_options || [];
  if (!opts.length) { $('.pitch-box').classList.add('hidden'); return; }
  $('.pitch-box').classList.remove('hidden');

  if (pitch === null) pitch = p.suggested_pitch;
  sel.innerHTML = opts.map(o => {
    const gb = o.bytes / (1024 ** 3);
    const size = gb >= 0.1 ? `${gb.toFixed(2)} GB` : `${(gb * 1024).toFixed(0)} MB`;
    return `<option value="${o.pitch}" ${o.within_budget ? '' : 'disabled'}
                    ${o.pitch === pitch ? 'selected' : ''}>
              ${o.pitch} mm — ${size}${o.within_budget ? '' : ' (too large)'}
            </option>`;
  }).join('');
  if (sel.selectedIndex < 0 || sel.options[sel.selectedIndex].disabled) {
    sel.value = String(p.suggested_pitch);
    pitch = p.suggested_pitch;
  }
  describePitch(p);
}

function describePitch(p) {
  const o = (p.pitch_options || []).find(x => x.pitch === pitch);
  $('#pitch-out').textContent = pitch ? `${pitch} mm fine` : 'engine default';
  if (!o) { $('#pitch-note').textContent = ''; return; }
  const gb = o.bytes / (1024 ** 3);
  $('#pitch-note').innerHTML =
    `Translation search allocates <b>${o.shape.join(' × ')}</b> voxels
     (${gb >= 0.1 ? gb.toFixed(2) + ' GB' : (gb * 1024).toFixed(0) + ' MB'})
     per orientation. A finer pitch searches on a tighter lattice; refinement
     recovers the slack either way.` +
    (pitch > p.suggested_pitch
      ? ' <b>Coarser than needed</b> — a finer pitch would still fit.' : '');
}

$('#pitch').onchange = (e) => {
  pitch = Number(e.target.value);
  describePitch(lastPreview);
};

async function loadGeometry() {
  const el = $('#viewer-loading');
  el.classList.remove('hidden');
  try {
    const r = await fetch(`/api/preview/${jobId}/geometry?ratio=${ratio}`);
    const payload = await r.json();
    if (!r.ok) throw new Error(payload.detail || 'could not read geometry');
    if (!viewer) viewer = new PartViewer($('#preview-canvas'));
    clearSelection();
    viewer.load(payload);
    viewer.setNoiseVisible($('#toggle-noise').checked);
    viewer.setPartOpacity($('#toggle-ghost').checked ? 0.25 : 1);
    if (payload.decimated_to) {
      $('#denoise-note').textContent =
        `Preview decimated to ${fmt(payload.decimated_to)} triangles for drawing; nesting uses the full mesh.`;
    }
  } catch (err) {
    el.textContent = err.message;
    return;
  }
  el.classList.add('hidden');
}

/* ---------- inspecting one body ---------- */
async function selectBody(index, row) {
  if (!viewer) return;
  if (selected === index) { clearSelection(); return; }   // click again to drop

  $('#frag-list').querySelectorAll('.frag').forEach(e => e.classList.remove('on'));
  row.classList.add('on');
  selected = index;
  $('#body-info').textContent = 'loading body…';
  show('#body-bar');
  try {
    const r = await fetch(`/api/preview/${jobId}/body/${index}?ratio=${ratio}`);
    const b = await r.json();
    if (!r.ok) throw new Error(b.detail || 'could not read that body');
    if (selected !== index) return;                       // a later click won
    viewer.selectBody(b);
    const f = b.fragment, e = f.extents;
    $('#body-info').innerHTML =
      `<b>body ${index}</b> — ${fmt(f.faces)} triangles ·
       ${dim(e[0])} × ${dim(e[1])} × ${dim(e[2])} · ${f.reason}`;
  } catch (err) {
    $('#body-info').textContent = err.message;
  }
}

function clearSelection() {
  selected = null;
  if (viewer) viewer.clearSelection();
  $('#frag-list').querySelectorAll('.frag').forEach(e => e.classList.remove('on'));
  hide('#body-bar');
}

/* Dragging the threshold re-runs the rule server-side. Debounced, because a
   drag fires continuously and each pass reclassifies hundreds of bodies. */
$('#ratio').oninput = (e) => {
  ratio = Number(e.target.value);
  $('#ratio-out').textContent = `${(ratio * 100).toFixed(1)}% — recalculating…`;
  clearTimeout(reclassifyTimer);
  reclassifyTimer = setTimeout(async () => {
    try {
      const r = await fetch(`/api/preview/${jobId}/classify?ratio=${ratio}`);
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || 'could not reclassify');
      renderPreview(data.preview);
      await loadGeometry();
    } catch (err) {
      $('#ratio-out').textContent = err.message;
    }
  }, 350);
};

$('#body-clear').onclick = clearSelection;
$('#body-focus').onclick = () => viewer && viewer.focusSelection();
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && selected !== null) clearSelection();
});

$('#view-reset').onclick = () => { clearSelection(); viewer && viewer.resetView(); };
$('#toggle-noise').onchange = (e) => viewer && viewer.setNoiseVisible(e.target.checked);
$('#toggle-ghost').onchange = (e) => viewer && viewer.setPartOpacity(e.target.checked ? 0.25 : 1);
$('#preview-cancel').onclick = () => location.reload();

$('#denoise').onclick = async () => {
  const btn = $('#denoise');
  btn.disabled = true; btn.textContent = 'Removing…';
  try {
    const r = await fetch(`/api/preview/${jobId}/denoise?ratio=${ratio}`, {method: 'POST'});
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'denoise failed');
    renderPreview(data.preview);
    await loadGeometry();
    $('#denoise-note').textContent =
      `Removed ${data.removed} ${data.removed === 1 ? 'fragment' : 'fragments'} (${fmt(data.removed_faces || 0)} triangles). The file to be nested has been updated.`;
  } catch (err) {
    $('#denoise-note').textContent = err.message;
    btn.disabled = false;
  }
};

$('#nest-now').onclick = async () => {
  const btn = $('#nest-now');
  btn.disabled = true; btn.textContent = 'Starting…';
  try {
    const q = pitch ? `?fine_pitch=${pitch}&coarse_pitch=${pitch * 2}` : '';
    const r = await fetch(`/api/preview/${jobId}/nest${q}`, {method: 'POST'});
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'could not start the run');
    hide('#preview-panel'); show('#progress-panel');
    poller = setInterval(poll, 1200); poll();
  } catch (err) {
    fail(err.message);
    hide('#preview-panel');
  } finally { btn.disabled = false; btn.textContent = 'Nest it'; }
};

/* ---------- polling ---------- */
async function poll() {
  const r = await fetch(`/api/jobs/${jobId}`);
  const j = await r.json();
  $('#bar').style.width = `${(j.progress * 100).toFixed(0)}%`;
  $('#stage').textContent = `${j.stage} — ${(j.progress*100).toFixed(0)}%`;
  $('#log').textContent = (j.log || []).join('\n');
  $('#log').scrollTop = $('#log').scrollHeight;
  if (j.status === 'done') { clearInterval(poller); hide('#progress-panel'); renderResults(j); }
  if (j.status === 'failed') { clearInterval(poller); hide('#progress-panel'); fail(j.error || 'the job failed'); }
}

function fail(msg) { $('#error-msg').textContent = msg; show('#error-panel'); show('#upload-panel'); }
const show = (s) => $(s).classList.remove('hidden');
const hide = (s) => $(s).classList.add('hidden');
$('#retry').onclick = () => { hide('#error-panel'); show('#upload-panel'); };
$('#again').onclick = () => location.reload();

/* ---------- results ---------- */
function renderResults(job) {
  $('#dl-all').href = `/api/jobs/${jobId}/all.zip`;
  const tb = $('#summary tbody'); tb.innerHTML = '';
  const cards = $('#cards'); cards.innerHTML = '';

  job.recommendations.forEach(rec => {
    const e = rec.extents;
    const tr = document.createElement('tr');
    if (rec.pareto) tr.className = 'is-pareto';
    tr.innerHTML = `<td>${rec.rank}</td>
      <td>${e[0].toFixed(1)} × ${e[1].toFixed(1)} × ${e[2].toFixed(1)}</td>
      <td>${fmt(rec.volume)}</td><td>${fmt(rec.footprint)}</td>
      <td>${e[2].toFixed(1)}</td><td>${rec.gap.toFixed(3)}</td>
      <td>${rec.pareto ? '<span class="pareto-dot"></span>' : ''}</td>`;
    tb.appendChild(tr);

    const card = document.createElement('div');
    card.className = 'card' + (rec.pareto ? ' pareto' : '');
    card.innerHTML = `
      <img loading="lazy" src="/api/jobs/${jobId}/rec/${rec.rank}/image/iso" alt="isometric view ${rec.rank}">
      <div class="card-body">
        <h3>#${rec.rank} ${rec.pareto ? '<span class="tag">PARETO</span>' : ''}</h3>
        <p class="metrics">
          <b>${e[0].toFixed(1)} × ${e[1].toFixed(1)} × ${e[2].toFixed(1)}</b> mm<br>
          volume <b>${fmt(rec.volume)}</b> mm³<br>
          footprint <b>${fmt(rec.footprint)}</b> mm²<br>
          gap <b>${rec.gap.toFixed(3)}</b> mm ${rec.refined ? '' : '(unrefined)'}
        </p>
        <div class="card-actions">
          <button class="btn views-btn" data-rank="${rec.rank}">2D views</button>
          <a class="btn ghost" href="/api/jobs/${jobId}/rec/${rec.rank}/stl">STL</a>
        </div>
      </div>`;
    card.querySelector('img').onclick = () => openModal(rec);
    card.querySelector('.views-btn').onclick = () => openModal(rec);
    cards.appendChild(card);
  });
  show('#results');
}

/* ---------- modal with top / bottom / front ---------- */
function openModal(rec) {
  const e = rec.extents;
  $('#modal-title').textContent =
    `#${rec.rank} — ${e[0].toFixed(1)} × ${e[1].toFixed(1)} × ${e[2].toFixed(1)} mm, gap ${rec.gap.toFixed(3)} mm`;
  $('#modal-views').innerHTML = ['iso','top','bottom','front'].map(v => `
    <figure>
      <img loading="lazy" src="/api/jobs/${jobId}/rec/${rec.rank}/image/${v}" alt="${v} view">
      <figcaption>${v.toUpperCase()}</figcaption>
    </figure>`).join('');
  $('#modal-zip').href = `/api/jobs/${jobId}/rec/${rec.rank}/views.zip`;
  $('#modal-stl').href = `/api/jobs/${jobId}/rec/${rec.rank}/stl`;
  show('#modal');
}
$('#modal-close').onclick = () => hide('#modal');
$('#modal').onclick = (e) => { if (e.target.id === 'modal') hide('#modal'); };
document.addEventListener('keydown', e => { if (e.key === 'Escape') hide('#modal'); });
