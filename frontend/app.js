/**
 * app.js  —  Frontend WebSocket Client
 *
 * Handles:
 *   - WebSocket lifecycle (connect, reconnect, send)
 *   - Rendering all message types (messages, tool calls, status…)
 *   - Permission confirmation modal
 *   - Status indicators (Claude API, Ollama)
 *   - Textarea auto-resize and Enter-to-send
 *   - Settings panel (gear icon) with all controls:
 *       · LOCAL / ONLINE mode switch
 *       · Primary Claude model dropdown
 *       · Local agent model dropdown
 *       · Prompt optimizer toggle
 *       · Advanced: all context / config knobs
 *   - Model display bar — shows active model name, updates on mode change
 *   - Project tree sidebar
 */

'use strict';

// ============================================================
// DOM references — core
// ============================================================
const chatArea   = document.getElementById('chat-area');
const userInput  = document.getElementById('user-input');
const btnSend    = document.getElementById('btn-send');
const btnClear   = document.getElementById('btn-clear');
const statusLine = document.getElementById('status-line');
const statusText = document.getElementById('status-text');

// Model bar
const modelDisplay   = document.getElementById('model-display');
const optimizerBadge = document.getElementById('optimizer-badge');
const localModeBadge = document.getElementById('local-mode-badge');

// Optimizer indicator (footer spinner)
const optimizerIndicator = document.getElementById('optimizer-indicator');

// Status pills
const statusClaude = document.getElementById('status-claude');
const statusOllama = document.getElementById('status-ollama');

// Confirmation modal
const confirmModal  = document.getElementById('confirm-modal');
const modalToolName = document.getElementById('modal-tool-name');
const modalDetails  = document.getElementById('modal-details');
const btnApprove    = document.getElementById('btn-approve');
const btnDeny       = document.getElementById('btn-deny');

// Tree sidebar
const treeContent = document.getElementById('tree-content');

// ============================================================
// DOM references — settings panel
// ============================================================
const settingsPanel    = document.getElementById('settings-panel');
const settingsBackdrop = document.getElementById('settings-backdrop');
const btnSettingsOpen  = document.getElementById('btn-settings');
const btnSettingsClose = document.getElementById('btn-settings-close');

const btnModeOnline  = document.getElementById('btn-mode-online');
const btnModeLocal   = document.getElementById('btn-mode-local');
const modeModelLabel = document.getElementById('mode-active-model-name');

const selPrimaryModel    = document.getElementById('sel-primary-model');
const selLocalAgentModel = document.getElementById('sel-local-agent-model');

const toggleOptimizer      = document.getElementById('toggle-optimizer');
const toggleOptimizerLabel = document.getElementById('toggle-optimizer-label');

const advancedHeader = document.getElementById('advanced-header');
const advancedBody   = document.getElementById('advanced-body');

const inpRecentTurns      = document.getElementById('inp-recent-turns');
const inpSummaryThreshold = document.getElementById('inp-summary-threshold');
const inpRetrievalN       = document.getElementById('inp-retrieval-n');
const inpSimilarity       = document.getElementById('inp-similarity');
const valSimilarity       = document.getElementById('val-similarity');
const inpMaxHistory       = document.getElementById('inp-max-history');
const inpMaxIterations    = document.getElementById('inp-max-iterations');
const inpLocalTimeout     = document.getElementById('inp-local-timeout');
const toggleEmbeddings    = document.getElementById('toggle-embeddings');
const toggleEmbeddingsLbl = document.getElementById('toggle-embeddings-label');
const inpTreeRoot         = document.getElementById('inp-tree-root');

// ============================================================
// State
// ============================================================
let ws = null;
let pendingConfirmationId = null;
let isWaiting             = false;
let optimizerEnabled      = false;
let localModeEnabled      = false;
let currentPrimaryModel   = 'claude-haiku-4-5';
let currentLocalModel     = 'qwen2.5:14b';

// ============================================================
// WebSocket
// ============================================================

function connectWS() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${location.host}/ws`);

  ws.addEventListener('open',    ()  => { console.log('[ws] Connected.'); setWaiting(false); });
  ws.addEventListener('message', (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { console.error('[ws] Bad JSON:', e.data); return; }
    handleServerEvent(msg.type, msg.data);
  });
  ws.addEventListener('close', () => { console.warn('[ws] Disconnected. Reconnecting…'); setTimeout(connectWS, 2000); });
  ws.addEventListener('error', (e) => console.error('[ws] Error:', e));
}

function sendWS(payload) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(payload));
  } else {
    console.warn('[ws] Not connected, message dropped.');
  }
}

// ============================================================
// Server event dispatcher
// ============================================================

function handleServerEvent(type, data) {
  switch (type) {
    case 'status':
      showStatus(data.text);
      break;

    case 'prompt_optimized':
      hideOptimizerIndicator();
      if (data) appendOptimizerPill(data.original, data.optimized);
      break;

    case 'tool_call':
      appendToolEvent(data.tool, data.input, null);
      break;

    case 'tool_result':
      updateLastToolEvent(data.tool, data.success, data.result);
      break;

    case 'tool_denied':
      appendToolDenied(data.tool);
      break;

    case 'confirm_required':
      openConfirmModal(data.confirmation_id, data.tool, data.input);
      break;

    case 'message':
      hideStatus();
      hideOptimizerIndicator();
      appendAgentMessage(data.text, data.source);
      setWaiting(false);
      break;

    case 'error':
      hideStatus();
      hideOptimizerIndicator();
      appendErrorMessage(data.text);
      setWaiting(false);
      break;

    case 'cleared':
      clearChat();
      break;

    case 'optimizer_status':
      setOptimizerState(data.enabled);
      break;

    case 'tree_update':
      updateTreePanel(data.tree);
      break;

    case 'local_mode_status':
      applyLocalMode(data.enabled);
      break;

    case 'model_status':
      // Primary model confirmed by backend
      currentPrimaryModel = data.model;
      updateModelDisplay();
      flashAck('ack-primary-model');
      break;

    case 'local_agent_model_status':
      // Local agent model confirmed
      currentLocalModel = data.model;
      updateModelDisplay();
      flashAck('ack-local-agent-model');
      break;

    case 'config_ack':
      // A set_config was accepted — flash the matching ack dot
      flashAckForKey(data.key);
      break;

    default:
      console.warn('[ws] Unknown event type:', type);
  }
}

// ============================================================
// Send user message
// ============================================================

function sendMessage() {
  const text = userInput.value.trim();
  if (!text || isWaiting) return;

  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();

  appendUserMessage(text);
  sendWS({ type: 'message', text });

  userInput.value = '';
  autoResize(userInput);
  setWaiting(true);
  showOptimizerIndicator();
  showStatus('Sending…');
}

// ============================================================
// Message row builders
// ============================================================

function appendUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'msg-row user-row';
  row.innerHTML = `
    <span class="msg-role">you</span>
    <div class="msg-bubble">${escapeHtml(text)}</div>
  `;
  chatArea.appendChild(row);
  scrollToBottom();
}

function appendAgentMessage(text, source = 'claude') {
  const row = document.createElement('div');
  row.className = 'msg-row agent-row';
  const localClass  = source === 'local' ? ' local-source' : '';
  const sourceLabel = source === 'local' ? 'local' : 'agent';
  row.innerHTML = `
    <span class="msg-role">${escapeHtml(sourceLabel)}</span>
    <div class="msg-bubble${localClass}">${renderMarkdown(text)}</div>
  `;
  chatArea.appendChild(row);
  scrollToBottom();
}

function appendErrorMessage(text) {
  const row = document.createElement('div');
  row.className = 'msg-row agent-row';
  row.innerHTML = `
    <span class="msg-role" style="color:var(--red)">err</span>
    <div class="msg-bubble" style="border-color:var(--red);color:var(--red);">${escapeHtml(text)}</div>
  `;
  chatArea.appendChild(row);
  scrollToBottom();
}

// ============================================================
// Tool event builders
// ============================================================

const lastToolEventEl = {};

function appendToolEvent(toolName, input) {
  const icon    = toolIcon(toolName);
  const wrapper = document.createElement('div');
  wrapper.className = 'tool-event';
  wrapper.dataset.tool = toolName;
  wrapper.innerHTML = `
    <div class="tool-header" onclick="toggleToolBody(this)">
      <span class="tool-icon">${icon}</span>
      <span class="tool-name">${escapeHtml(toolName)}</span>
      <span class="tool-status">…</span>
      <span class="tool-arrow">▶</span>
    </div>
    <div class="tool-body">
      <div class="tool-result-status">
        <span class="tool-label" style="color:var(--text-dim)">INPUT</span>
      </div>
      <pre>${escapeHtml(JSON.stringify(input, null, 2))}</pre>
      <div class="tool-result-body" style="margin-top:8px"></div>
    </div>
  `;
  chatArea.appendChild(wrapper);
  lastToolEventEl[toolName] = wrapper;
  scrollToBottom();
}

function updateLastToolEvent(toolName, success, result) {
  const el = lastToolEventEl[toolName];
  if (!el) return;

  const statusEl   = el.querySelector('.tool-status');
  const resultBody = el.querySelector('.tool-result-body');

  if (statusEl) {
    statusEl.textContent = success ? ' ✓' : ' ✗';
    statusEl.style.color = success ? 'var(--green)' : 'var(--red)';
  }

  if (resultBody) {
    const displayResult = { ...result };
    delete displayResult.tree;  // tree shown in sidebar, not inline
    resultBody.innerHTML = `
      <div class="tool-result-status">
        <span class="${success ? 'result-ok' : 'result-fail'}">${success ? '✓ success' : '✗ failed'}</span>
      </div>
      <pre>${escapeHtml(JSON.stringify(displayResult, null, 2))}</pre>
    `;
  }
  scrollToBottom();
}

function appendToolDenied(toolName) {
  const el = document.createElement('div');
  el.className = 'tool-denied-msg';
  el.textContent = `✗ ${toolName}: denied by user`;
  chatArea.appendChild(el);
  scrollToBottom();
}

window.toggleToolBody = function(header) {
  const expanded = header.classList.toggle('expanded');
  header.nextElementSibling.classList.toggle('visible', expanded);
};

function toolIcon(name) {
  const icons = { read_file: '📄', write_file: '✏️', list_directory: '📁', list_capabilities: '🤖' };
  return icons[name] || '🔧';
}

// ============================================================
// Project tree sidebar
// ============================================================

function updateTreePanel(treeText) {
  if (!treeText) return;
  const ph = document.getElementById('tree-placeholder');
  if (ph) ph.remove();
  treeContent.textContent = treeText;
}

window.toggleTreePanel = function() {
  const panel = document.getElementById('tree-panel');
  const btn   = document.getElementById('tree-collapse-btn');
  const col   = panel.classList.toggle('collapsed');
  btn.textContent = col ? '▶' : '◀';
};

// ============================================================
// Optimizer indicator (footer spinner)
// ============================================================

function showOptimizerIndicator() {
  if (optimizerEnabled) optimizerIndicator.classList.remove('hidden');
}
function hideOptimizerIndicator() {
  optimizerIndicator.classList.add('hidden');
}

function appendOptimizerPill(original, optimized) {
  const pill = document.createElement('div');
  pill.className = 'optimizer-pill';
  pill.style.position = 'relative';
  pill.innerHTML = `
    ⚙ prompt optimized
    <div class="optimizer-detail">
      <strong style="color:var(--text-dim)">Original:</strong><br>${escapeHtml(original)}
      <br><br>
      <strong style="color:var(--amber-dim)">Optimized:</strong><br>${escapeHtml(optimized)}
    </div>
  `;
  chatArea.appendChild(pill);
  scrollToBottom();
}

// ============================================================
// Optimizer state (toggled from settings panel)
// ============================================================

/**
 * Apply the optimizer state to:
 *   - JS flag
 *   - The toggle widget in the settings panel
 *   - The read-only badge in the model bar
 */
function setOptimizerState(enabled) {
  optimizerEnabled = enabled;

  // Panel toggle
  toggleOptimizer.classList.toggle('on', enabled);
  toggleOptimizerLabel.textContent = enabled ? 'on' : 'off';

  // Model bar badge
  optimizerBadge.textContent = enabled ? '⚙ optimizer: on' : '⚙ optimizer: off';
  optimizerBadge.classList.remove('hidden');
}

toggleOptimizer.addEventListener('click', () => {
  const next = !optimizerEnabled;
  sendWS({ type: 'set_optimizer', data: { enabled: next } });
  setOptimizerState(next);  // optimistic
});

// ============================================================
// Local mode state
// ============================================================

/**
 * Apply local mode to all relevant UI:
 *   - Mode switch buttons (ONLINE/LOCAL highlight)
 *   - Model display bar
 *   - Claude status pill label
 *   - Primary model dropdown (disabled in local mode)
 *   - Model bar local-mode badge
 */
function applyLocalMode(enabled) {
  localModeEnabled = enabled;

  // Mode switch highlight
  btnModeOnline.classList.toggle('active-online', !enabled);
  btnModeLocal.classList.toggle('active-local',   enabled);

  // Primary model dropdown — grayed out when local is active
  selPrimaryModel.disabled = enabled;
  const primaryRow = document.getElementById('row-primary-model');
  if (primaryRow) primaryRow.style.opacity = enabled ? '0.4' : '1';

  // Model bar
  updateModelDisplay();

  // Claude pill label
  const claudeLabel = statusClaude.querySelector('.status-label');
  if (claudeLabel) claudeLabel.textContent = enabled ? 'local' : 'claude';

  // Local-mode badge in model bar
  localModeBadge.textContent = enabled ? '🖥 local: on' : '🖥 local: off';
  localModeBadge.classList.remove('hidden');
}

/**
 * Update the model-display bar to show which model is active.
 * - Online mode: shows the Claude primary model
 * - Local mode: shows the Ollama local-agent model
 */
function updateModelDisplay() {
  if (localModeEnabled) {
    modelDisplay.textContent = `local: ${currentLocalModel}`;
  } else {
    modelDisplay.textContent = `claude: ${currentPrimaryModel}`;
  }

  // Also update the sub-label inside the settings panel mode toggle
  modeModelLabel.textContent = localModeEnabled ? currentLocalModel : currentPrimaryModel;
}

// Mode switch click handlers
btnModeOnline.addEventListener('click', () => {
  if (localModeEnabled) {
    sendWS({ type: 'set_local_mode', data: { enabled: false } });
    applyLocalMode(false);
  }
});

btnModeLocal.addEventListener('click', () => {
  if (!localModeEnabled) {
    sendWS({ type: 'set_local_mode', data: { enabled: true } });
    applyLocalMode(true);
  }
});

// ============================================================
// Settings panel — open / close
// ============================================================

function openSettings() {
  settingsPanel.classList.add('open');
  settingsBackdrop.classList.add('visible');
}
function closeSettings() {
  settingsPanel.classList.remove('open');
  settingsBackdrop.classList.remove('visible');
}

btnSettingsOpen.addEventListener('click',  openSettings);
btnSettingsClose.addEventListener('click', closeSettings);
settingsBackdrop.addEventListener('click', closeSettings);

// Keyboard: Escape closes the panel
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && settingsPanel.classList.contains('open')) closeSettings();
});

// ============================================================
// Settings panel — Advanced section toggle
// ============================================================

window.toggleAdvanced = function() {
  advancedHeader.classList.toggle('open');
  advancedBody.classList.toggle('open');
};

// ============================================================
// Settings panel — model dropdowns
// ============================================================

selPrimaryModel.addEventListener('change', () => {
  const model = selPrimaryModel.value;
  currentPrimaryModel = model;
  sendWS({ type: 'set_model', data: { model } });
  if (!localModeEnabled) updateModelDisplay();
  // Ack flash will come from model_status server event
});

selLocalAgentModel.addEventListener('change', () => {
  const model = selLocalAgentModel.value;
  currentLocalModel = model;
  sendWS({ type: 'set_local_agent_model', data: { model } });
  if (localModeEnabled) updateModelDisplay();
  // Ack flash will come from local_agent_model_status server event
});

// ============================================================
// Settings panel — embeddings toggle
// ============================================================

toggleEmbeddings.addEventListener('click', () => {
  const wasOn = toggleEmbeddings.classList.contains('on');
  const next  = !wasOn;
  toggleEmbeddings.classList.toggle('on', next);
  toggleEmbeddingsLbl.textContent = next ? 'on' : 'off';
  sendWS({ type: 'set_config', data: { key: 'embeddings.enabled', value: next } });
});

// ============================================================
// Settings panel — number inputs (debounced)
// ============================================================

// Map: input element ID → config key
const numberInputMap = [
  { id: 'inp-recent-turns',      key: 'context.recent_turns',           parse: parseInt  },
  { id: 'inp-summary-threshold', key: 'context.summary_threshold',      parse: parseInt  },
  { id: 'inp-retrieval-n',       key: 'context.retrieval_n',            parse: parseInt  },
  { id: 'inp-max-history',       key: 'context.max_history_turns',      parse: parseInt  },
  { id: 'inp-max-iterations',    key: 'context.max_iterations_per_turn', parse: parseInt  },
  { id: 'inp-local-timeout',     key: 'local_agent_timeout',            parse: parseFloat },
];

const _debounceTimers = {};

numberInputMap.forEach(({ id, key, parse }) => {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener('input', () => {
    clearTimeout(_debounceTimers[id]);
    _debounceTimers[id] = setTimeout(() => {
      const v = parse(el.value);
      if (!isNaN(v)) sendWS({ type: 'set_config', data: { key, value: v } });
    }, 500);
  });
});

// Similarity cutoff slider — send immediately on change (sliders feel sluggish with debounce)
inpSimilarity.addEventListener('input', () => {
  const v = parseFloat(inpSimilarity.value);
  valSimilarity.textContent = v.toFixed(2);
  clearTimeout(_debounceTimers['similarity']);
  _debounceTimers['similarity'] = setTimeout(() => {
    sendWS({ type: 'set_config', data: { key: 'context.similarity_cutoff', value: v } });
  }, 300);
});

// Tree root — send on blur or Enter
function sendTreeRoot() {
  const v = inpTreeRoot.value.trim();
  if (v) sendWS({ type: 'set_config', data: { key: 'tree_root', value: v } });
}
inpTreeRoot.addEventListener('blur', sendTreeRoot);
inpTreeRoot.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); sendTreeRoot(); } });

// ============================================================
// ACK flash helpers
// ============================================================

// Map from config key → ack element ID
const keyToAckId = {
  'context.recent_turns':            'ack-recent-turns',
  'context.summary_threshold':       'ack-summary-threshold',
  'context.retrieval_n':             'ack-retrieval-n',
  'context.similarity_cutoff':       'ack-similarity',
  'context.max_history_turns':       'ack-max-history',
  'context.max_iterations_per_turn': 'ack-max-iterations',
  'local_agent_timeout':             'ack-local-timeout',
  'embeddings.enabled':              null,   // toggle — no dot needed
  'tree_root':                       'ack-tree-root',
};

function flashAck(ackId) {
  if (!ackId) return;
  const el = document.getElementById(ackId);
  if (!el) return;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 1200);
}

function flashAckForKey(key) {
  flashAck(keyToAckId[key] || null);
}

// ============================================================
// Populate settings panel from /status data
// ============================================================

function populateSettingsFromStatus(data) {
  // Primary model dropdown
  if (data.primary_model) {
    currentPrimaryModel = data.primary_model;
    if (selPrimaryModel) {
      // Select matching option, or add it if not in list
      const existing = [...selPrimaryModel.options].find(o => o.value === data.primary_model);
      if (existing) existing.selected = true;
    }
  }

  // Local agent model dropdown
  if (data.local_agent_model) {
    currentLocalModel = data.local_agent_model;
    if (selLocalAgentModel) {
      const existing = [...selLocalAgentModel.options].find(o => o.value === data.local_agent_model);
      if (existing) existing.selected = true;
    }
  }

  // Optimizer toggle
  setOptimizerState(!!data.use_prompt_optimizer);

  // Local mode
  applyLocalMode(!!data.local_mode);

  // Advanced fields
  const ctx = data.context || {};
  if (inpRecentTurns      && ctx.recent_turns      != null) inpRecentTurns.value      = ctx.recent_turns;
  if (inpSummaryThreshold && ctx.summary_threshold  != null) inpSummaryThreshold.value = ctx.summary_threshold;
  if (inpRetrievalN       && ctx.retrieval_n        != null) inpRetrievalN.value       = ctx.retrieval_n;
  if (inpSimilarity       && ctx.similarity_cutoff  != null) {
    inpSimilarity.value  = ctx.similarity_cutoff;
    valSimilarity.textContent = Number(ctx.similarity_cutoff).toFixed(2);
  }
  if (inpMaxHistory    && ctx.max_history_turns         != null) inpMaxHistory.value    = ctx.max_history_turns;
  if (inpMaxIterations && ctx.max_iterations_per_turn   != null) inpMaxIterations.value = ctx.max_iterations_per_turn;
  if (inpLocalTimeout  && data.local_agent_timeout      != null) inpLocalTimeout.value  = data.local_agent_timeout;

  const emb = data.embeddings || {};
  if (emb.enabled != null) {
    toggleEmbeddings.classList.toggle('on', !!emb.enabled);
    toggleEmbeddingsLbl.textContent = emb.enabled ? 'on' : 'off';
  }

  if (inpTreeRoot && data.tree_root != null) inpTreeRoot.value = data.tree_root;
}

// ============================================================
// Confirmation modal
// ============================================================

function openConfirmModal(confirmationId, toolName, input) {
  pendingConfirmationId = confirmationId;
  modalToolName.textContent = `Tool: ${toolName}`;
  modalDetails.textContent  = JSON.stringify(input, null, 2);
  confirmModal.classList.remove('hidden');
  btnApprove.focus();
}
function closeConfirmModal() {
  confirmModal.classList.add('hidden');
  pendingConfirmationId = null;
}

btnApprove.addEventListener('click', () => {
  if (pendingConfirmationId) sendWS({ type: 'confirm', confirmation_id: pendingConfirmationId, approved: true });
  closeConfirmModal();
});
btnDeny.addEventListener('click', () => {
  if (pendingConfirmationId) sendWS({ type: 'confirm', confirmation_id: pendingConfirmationId, approved: false });
  closeConfirmModal();
});
confirmModal.addEventListener('click', (e) => { if (e.target === confirmModal) btnDeny.click(); });

// ============================================================
// Status + waiting state
// ============================================================

function showStatus(text) { statusText.textContent = text; statusLine.classList.remove('hidden'); }
function hideStatus()      { statusLine.classList.add('hidden'); statusText.textContent = ''; }

function setWaiting(waiting) {
  isWaiting = waiting;
  btnSend.disabled   = waiting;
  userInput.disabled = waiting;
  if (!waiting) hideStatus();
}

// ============================================================
// Clear conversation
// ============================================================

function clearChat() {
  chatArea.querySelectorAll('.msg-row, .tool-event, .tool-denied-msg, .optimizer-pill')
          .forEach(el => el.remove());
  Object.keys(lastToolEventEl).forEach(k => delete lastToolEventEl[k]);

  const welcome = document.createElement('div');
  welcome.id = 'welcome';
  welcome.className = 'welcome-block';
  welcome.innerHTML = `
    <div class="welcome-title">Conversation cleared.</div>
    <div class="welcome-body">Start a new conversation below.</div>
  `;
  chatArea.appendChild(welcome);
}

btnClear.addEventListener('click', () => {
  if (confirm('Clear the conversation history?')) sendWS({ type: 'clear' });
});

// ============================================================
// Status polling — /status endpoint
// ============================================================

async function pollStatus() {
  try {
    const res  = await fetch('/status');
    const data = await res.json();

    // Status pills
    statusClaude.classList.toggle('online',  !!data.claude_api);
    statusClaude.classList.toggle('offline', !data.claude_api);
    statusOllama.classList.toggle('online',  !!data.ollama);
    statusOllama.classList.toggle('offline', !data.ollama);

    // Populate all settings panel controls from server state
    populateSettingsFromStatus(data);

  } catch (e) {
    console.warn('[status] Could not fetch /status:', e);
  }
}

// ============================================================
// Input handling
// ============================================================

btnSend.addEventListener('click', sendMessage);

userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
userInput.addEventListener('input', () => autoResize(userInput));

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

// ============================================================
// Helpers
// ============================================================

function scrollToBottom() { chatArea.scrollTop = chatArea.scrollHeight; }

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function renderMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
}

// ============================================================
// Init
// ============================================================

connectWS();
pollStatus();
setInterval(pollStatus, 30_000);
