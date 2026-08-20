import {PairViewer} from './viewer.js';

const $ = (s) => document.querySelector(s);
const fmt = (n) => n.toLocaleString(undefined, {maximumFractionDigits: 0});
let jobId = null, poller = null;

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

$('#upload-form').onsubmit = async (e) => {
  e.preventDefault();
  if (!fileInput.files.length) { alert('Choose an STL first.'); return; }
  const fd = new FormData(e.target);
  fd.set('file', fileInput.files[0]);
  $('#submit').disabled = true;
  hide('#error-panel'); hide('#results');
  try {
    const r = await fetch('/api/jobs', {method: 'POST', body: fd});
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'upload rejected');
    jobId = data.job_id;
    hide('#upload-panel'); show('#progress-panel');
    poller = setInterval(poll, 1200); poll();
  } catch (err) {
    fail(err.message);
  } finally { $('#submit').disabled = false; }
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

/* ---------- live viewers, kept to a budget -----------------------------------
 * A browser will only hand out a limited number of WebGL contexts — well under
 * twenty, and a lost context takes the whole page's canvases with it. Ten
 * recommendations would sail past that, so a card only gets a viewer while it
 * is on screen, and the oldest is torn down once the budget is spent.
 * -------------------------------------------------------------------------- */
function showLoadError(host, err) {
  if (!host) return;
  host.dataset.error = '1';
  const msg = host.querySelector('.loading');
  if (msg) msg.textContent = `could not load the model — ${err.message || err}`;
}

const MAX_LIVE = 5;
const live = new Map();               // canvas -> PairViewer, in insertion order

function acquire(canvas, url, onReady) {
  if (live.has(canvas)) return live.get(canvas);
  while (live.size >= MAX_LIVE) {
    const [oldest, viewer] = live.entries().next().value;
    viewer.dispose();
    live.delete(oldest);
    oldest.closest('.stage3d')?.classList.remove('ready');
  }
  const v = new PairViewer(canvas);
  live.set(canvas, v);
  v.load(url).then(() => onReady && onReady(v))
             .catch(err => showLoadError(canvas.closest('.stage3d'), err));
  return v;
}

// Only build a viewer once the card is actually scrolled into view.
const seen = new IntersectionObserver((entries) => {
  entries.forEach(en => {
    if (!en.isIntersecting) return;
    const host = en.target;
    const canvas = host.querySelector('canvas');
    acquire(canvas, host.dataset.url, () => host.classList.add('ready'));
  });
}, {rootMargin: '200px'});

/* ---------- results ---------- */
function renderResults(job) {
  $('#dl-all').href = `/api/jobs/${jobId}/all.zip`;
  const tb = $('#summary tbody'); tb.innerHTML = '';
  const cards = $('#cards'); cards.innerHTML = '';
  live.forEach(v => v.dispose()); live.clear();

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

    const url = `/api/jobs/${jobId}/rec/${rec.rank}/model.glb`;
    const card = document.createElement('div');
    card.className = 'card' + (rec.pareto ? ' pareto' : '');
    card.innerHTML = `
      <div class="stage3d" data-url="${url}">
        <canvas></canvas>
        <span class="loading">loading model…</span>
        <span class="hint3d">drag to rotate · scroll to zoom</span>
      </div>
      <div class="card-body">
        <h3>#${rec.rank} ${rec.pareto ? '<span class="tag">PARETO</span>' : ''}</h3>
        <p class="metrics">
          <b>${e[0].toFixed(1)} × ${e[1].toFixed(1)} × ${e[2].toFixed(1)}</b> mm<br>
          volume <b>${fmt(rec.volume)}</b> mm³<br>
          footprint <b>${fmt(rec.footprint)}</b> mm²<br>
          gap <b>${rec.gap.toFixed(3)}</b> mm ${rec.refined ? '' : '(unrefined)'}
        </p>
        <div class="card-actions">
          <button class="btn expand-btn">Inspect</button>
          <a class="btn ghost" href="/api/jobs/${jobId}/rec/${rec.rank}/stl">STL</a>
          <a class="btn ghost" href="${url}" download>GLB</a>
        </div>
      </div>`;
    card.querySelector('.expand-btn').onclick = () => openModal(rec, url);
    cards.appendChild(card);
    seen.observe(card.querySelector('.stage3d'));
  });
  show('#results');
}

/* ---------- full-size interactive view ---------- */
let modalViewer = null;

function openModal(rec, url) {
  const e = rec.extents;
  $('#modal-title').textContent =
    `#${rec.rank} — ${e[0].toFixed(1)} × ${e[1].toFixed(1)} × ${e[2].toFixed(1)} mm, gap ${rec.gap.toFixed(3)} mm`;
  $('#modal-stl').href = `/api/jobs/${jobId}/rec/${rec.rank}/stl`;
  $('#modal-glb').href = url;
  show('#modal');

  // Built fresh each time and destroyed on close, so the modal never competes
  // with the cards for a context.
  if (modalViewer) modalViewer.dispose();
  const canvas = $('#modal-canvas');
  modalViewer = new PairViewer(canvas);
  $('#modal-stage').classList.remove('ready');
  modalViewer.load(url)
    .then(() => $('#modal-stage').classList.add('ready'))
    .catch(err => showLoadError($('#modal-stage'), err));
}

function closeModal() {
  hide('#modal');
  if (modalViewer) { modalViewer.dispose(); modalViewer = null; }
  isolated = null;
}
$('#modal-close').onclick = closeModal;
$('#modal').onclick = (e) => { if (e.target.id === 'modal') closeModal(); };
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

$('#view-reset').onclick = () => modalViewer && modalViewer.resetView();
$('#view-spin').onchange = (e) => modalViewer && modalViewer.setSpin(e.target.checked);

// Cycling which copy is solid is how you check an interlock: dim one and the
// mating faces of the other become visible from the inside.
let isolated = null;
$('#view-isolate').onclick = () => {
  if (!modalViewer) return;
  isolated = isolated === null ? 0 : isolated === 0 ? 1 : null;
  modalViewer.isolate(isolated);
  $('#view-isolate').textContent =
    isolated === null ? 'Isolate a copy' : `Showing copy ${'AB'[isolated]}`;
};
