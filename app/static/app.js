import {PartViewer} from './viewer.js';

const $ = (s) => document.querySelector(s);
const fmt = (n) => n.toLocaleString(undefined, {maximumFractionDigits: 0});
const dim = (n) => n.toLocaleString(undefined, {maximumFractionDigits: 2});
let jobId = null, poller = null, viewer = null;

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
function renderPreview(p) {
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

  $('#preview-rule').innerHTML =
    `A body is <b>noise</b> only when it is <b>both</b> detached from the object
     — standing more than <b>${dim(p.touch_tolerance)}</b> clear of it — and
     under <b>${dim(p.threshold)}</b> across, which is
     <b>${(p.ratio * 100).toFixed(0)}%</b> of the object's smallest dimension
     (${dim(p.object_smallest)}). Anything touching the object is kept whatever
     its size.`;

  const frags = p.bodies > 1 ? p.fragments : [];
  $('#frag-list').innerHTML = frags.length ? frags
    .slice()
    .sort((a, b) => (b.is_noise - a.is_noise) || (b.largest - a.largest))
    .map(f => `
      <div class="frag ${f.is_noise ? 'is-noise' : ''}" title="${f.reason}">
        <span class="dot"></span>
        <span class="frag-id">body ${f.index}</span>
        <span class="frag-size">${dim(f.extents[0])} × ${dim(f.extents[1])} × ${dim(f.extents[2])}</span>
        <span class="frag-max">${f.gap === null ? '' : 'gap ' + dim(f.gap)}</span>
        <span class="frag-tag">${f.is_noise ? 'NOISE' : 'keep'}</span>
      </div>`).join('') : '';

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

async function loadGeometry() {
  const el = $('#viewer-loading');
  el.classList.remove('hidden');
  try {
    const r = await fetch(`/api/preview/${jobId}/geometry`);
    const payload = await r.json();
    if (!r.ok) throw new Error(payload.detail || 'could not read geometry');
    if (!viewer) viewer = new PartViewer($('#preview-canvas'));
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

$('#view-reset').onclick = () => viewer && viewer.resetView();
$('#toggle-noise').onchange = (e) => viewer && viewer.setNoiseVisible(e.target.checked);
$('#toggle-ghost').onchange = (e) => viewer && viewer.setPartOpacity(e.target.checked ? 0.25 : 1);
$('#preview-cancel').onclick = () => location.reload();

$('#denoise').onclick = async () => {
  const btn = $('#denoise');
  btn.disabled = true; btn.textContent = 'Removing…';
  try {
    const r = await fetch(`/api/preview/${jobId}/denoise`, {method: 'POST'});
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
    const r = await fetch(`/api/preview/${jobId}/nest`, {method: 'POST'});
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
