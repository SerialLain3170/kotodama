// Nostalgic fake hit counter — increments once per visit, persisted in localStorage.
(function () {
  const key = 'vn_maker_hit_counter';
  let n = parseInt(localStorage.getItem(key) || '128401', 10);
  n += 1;
  localStorage.setItem(key, String(n));
  document.getElementById('counter').textContent = String(n).padStart(7, '0');
})();

const FIELD_IDS = ['character_name', 'character_description', 'background_description', 'persona', 'context', 'genre'];
const STORAGE_KEY = 'vn_maker_fields';

function loadFields() {
  const raw = sessionStorage.getItem(STORAGE_KEY);
  if (!raw) return;
  try {
    const values = JSON.parse(raw);
    FIELD_IDS.forEach(id => { if (values[id] !== undefined) document.getElementById(id).value = values[id]; });
  } catch (e) { /* ignore malformed storage */ }
}
loadFields();

const formPanel = document.getElementById('form-panel');
const waitingPanel = document.getElementById('waiting-panel');
const videoPanel = document.getElementById('video-panel');
const errorPanel = document.getElementById('error-panel');
const waitMessage = document.getElementById('wait-message');

function showOnly(panel) {
  [formPanel, waitingPanel, videoPanel, errorPanel].forEach(p => p.classList.add('hidden'));
  panel.classList.remove('hidden');
}

document.getElementById('retro-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const payload = {
    character_name: document.getElementById('character_name').value,
    character_description: document.getElementById('character_description').value,
    background_description: document.getElementById('background_description').value,
    persona: document.getElementById('persona').value,
    context: document.getElementById('context').value,
    genre: document.getElementById('genre').value,
    retro: true,
  };

  showOnly(waitingPanel);
  waitMessage.textContent = '☆彡 送信中… ☆彡';

  let job_id;
  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error('サーバーエラー: ' + res.status);
    ({ job_id } = await res.json());
  } catch (err) {
    document.getElementById('retro-error').textContent = String(err);
    showOnly(errorPanel);
    return;
  }

  poll(job_id);
});

async function poll(job_id) {
  const messages = [
    '☆彡 いま頑張って作っています ☆彡',
    '★ FM音源でBGMを演奏中… ★',
    '☆ 640x400 の絵を描いています ☆',
    '★ もうすこしお待ちください ★',
  ];
  let i = 0;
  const tick = setInterval(() => {
    waitMessage.textContent = messages[i % messages.length];
    i++;
  }, 3500);

  try {
    while (true) {
      await new Promise(r => setTimeout(r, 3000));
      const res = await fetch(`/api/status/${job_id}`);
      if (!res.ok) throw new Error('サーバーエラー: ' + res.status);
      const data = await res.json();

      if (data.status === 'done') {
        clearInterval(tick);
        document.getElementById('retro-video').src = data.video_url;
        document.getElementById('retro-download').href = data.video_url;
        showOnly(videoPanel);
        return;
      } else if (data.status === 'error') {
        clearInterval(tick);
        document.getElementById('retro-error').textContent = data.error || '不明なエラーです。';
        showOnly(errorPanel);
        return;
      } else if (data.status === 'queued') {
        waitMessage.textContent = `★ 順番待ち中…（あと ${data.queue_position} 件） ★`;
      }
    }
  } catch (err) {
    clearInterval(tick);
    document.getElementById('retro-error').textContent = String(err);
    showOnly(errorPanel);
  }
}

document.getElementById('retro-again-btn').addEventListener('click', () => showOnly(formPanel));
document.getElementById('retro-retry-btn').addEventListener('click', () => showOnly(formPanel));
