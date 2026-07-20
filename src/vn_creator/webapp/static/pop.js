const FIELD_IDS = ['character_name', 'character_description', 'background_description', 'persona', 'context', 'genre'];
const STORAGE_KEY = 'vn_maker_fields';

function saveFields() {
  const values = {};
  FIELD_IDS.forEach(id => { values[id] = document.getElementById(id).value; });
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(values));
}

function loadFields() {
  const raw = sessionStorage.getItem(STORAGE_KEY);
  if (!raw) return;
  try {
    const values = JSON.parse(raw);
    FIELD_IDS.forEach(id => { if (values[id] !== undefined) document.getElementById(id).value = values[id]; });
  } catch (e) { /* ignore malformed storage */ }
}

loadFields();

const formCard = document.getElementById('form-card');
const statusCard = document.getElementById('status-card');
const resultCard = document.getElementById('result-card');
const errorCard = document.getElementById('error-card');
const statusMessage = document.getElementById('status-message');

function showOnly(card) {
  [formCard, statusCard, resultCard, errorCard].forEach(c => c.classList.add('hidden'));
  card.classList.remove('hidden');
}

// Flipping the retro switch jumps straight to the retro-styled page, which has
// its own copy of this form — carry over whatever's already typed.
document.getElementById('retro').addEventListener('change', (e) => {
  if (e.target.checked) {
    saveFields();
    window.location.href = '/retro';
  }
});

document.getElementById('gen-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  saveFields();
  const payload = {
    character_name: document.getElementById('character_name').value,
    character_description: document.getElementById('character_description').value,
    background_description: document.getElementById('background_description').value,
    persona: document.getElementById('persona').value,
    context: document.getElementById('context').value,
    genre: document.getElementById('genre').value,
    retro: false,
  };

  showOnly(statusCard);
  statusMessage.textContent = 'リクエストを送信しています…';

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
    showError(String(err));
    return;
  }

  poll(job_id);
});

async function poll(job_id) {
  const messages = [
    'シーンを考えています…',
    '声を録音しています…',
    'イラストを描いています…',
    '音楽を作っています…',
    '最後の仕上げをしています…',
  ];
  let i = 0;
  const interval = setInterval(() => {
    statusMessage.textContent = messages[i % messages.length] + '（数分かかります）';
    i++;
  }, 4000);
  statusMessage.textContent = messages[0];

  try {
    while (true) {
      await new Promise(r => setTimeout(r, 3000));
      const res = await fetch(`/api/status/${job_id}`);
      if (!res.ok) throw new Error('サーバーエラー: ' + res.status);
      const data = await res.json();

      if (data.status === 'queued') {
        statusMessage.textContent = `順番待ちです…（あと ${data.queue_position} 件）`;
      } else if (data.status === 'done') {
        clearInterval(interval);
        showResult(data.video_url);
        return;
      } else if (data.status === 'error') {
        clearInterval(interval);
        showError(data.error || '不明なエラーが発生しました。');
        return;
      }
    }
  } catch (err) {
    clearInterval(interval);
    showError(String(err));
  }
}

function showResult(video_url) {
  const video = document.getElementById('result-video');
  video.src = video_url;
  document.getElementById('download-link').href = video_url;
  showOnly(resultCard);
}

function showError(message) {
  document.getElementById('error-message').textContent = message;
  showOnly(errorCard);
}

document.getElementById('again-btn').addEventListener('click', () => showOnly(formCard));
document.getElementById('retry-btn').addEventListener('click', () => showOnly(formCard));
