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
  $('#filehint').textContent = f ? `${f.name} — ${(f.size/1e6).toFixed(1)} MB` : 'Open meshes are repaired automatically. Up to 64 MB.';
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
