/**
 * app.js  —  Frontend WebSocket Client
 *
 * Handles:
 *   - WebSocket lifecycle (connect, reconnect, send)
 *   - Rendering all message types from the server (messages, tool calls, status…)
 *   - Permission confirmation modal
 *   - Status indicators (Claude API, Ollama)
 *   - Textarea auto-resize and Enter-to-send
 *   - Optimizer toggle button (sends set_optimizer, handles optimizer_status)
 */

'use strict';

// ============================================================
// DOM references
// ============================================================
const chatArea    = document.getElementById('chat-area');
const userInput   = document.getElementById('user-input');
const btnSend     = document.getElementById('btn-send');
const btnClear    = document.getElementById('btn-clear');
const statusLine  = document.getElementById('status-line');
const statusText  = document.getElementById('status-text');

// Model bar elements
const modelDisplay       = document.getElementById('model-display');
const optimizerBadge     = document.getElementById('optimizer-badge');       // static label
const btnOptimizerToggle = document.getElementById('btn-optimizer-toggle');  // ON/OFF button

// Input-bar processing indicator (shown while local LLM is optimizing the prompt)
// NOTE: The element ID is "optimizer-indicator" — distinct from "optimizer-badge"
//   optimizer-badge     → model bar label, reflects persistent on/off state
//   optimizer-indicator → footer spinner, shown only during active optimization
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

// ============================================================
// State
// ============================================================
let ws = null;
let pendingConfirmationId = null;  // The confirmation_id we're waiting to resolve
let isWaiting = false;             // True while the agent is processing
let optimizerEnabled = false;      // Mirrors agent.use_prompt_optimizer on the backend

// ============================================================
// WebSocket connection
// ============================================================

function connectWS() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${location.host}/ws`);

  ws.addEventListener('open', () => {
    console.log('[ws] Connected.');
    setWaiting(false);
  });

  ws.addEventListener('message', (event) => {
    let msg;
    try { msg = JSON.parse(event.data); }
    catch { console.error('[ws] Bad JSON:', event.data); return; }
    handleServerEvent(msg.type, msg.data);
  });

  ws.addEventListener('close', () => {
    console.warn('[ws] Disconnected. Reconnecting in 2s…');
    setTimeout(connectWS, 2000);
  });

  ws.addEventListener('error', (e) => {
    console.error('[ws] Error:', e);
  });
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
      // Processing status text shown above the input bar
      showStatus(data.text);
      break;

    case 'prompt_optimized':
      // data is null (no change) or {original, optimized}
      hideOptimizerIndicator();
      if (data) {
        // Show a small pill in the chat showing the optimizer changed the prompt
        appendOptimizerPill(data.original, data.optimized);
      }
      break;

    case 'tool_call':
      // Claude is calling a tool — show it inline in the chat
      appendToolEvent(data.tool, data.input, null);
      break;

    case 'tool_result':
      // Tool finished — update the last tool event card with the result
      updateLastToolEvent(data.tool, data.success, data.result);
      break;

    case 'tool_denied':
      appendToolDenied(data.tool);
      break;

    case 'confirm_required':
      // Agent wants to run a destructive tool — show the modal
      openConfirmModal(data.confirmation_id, data.tool, data.input);
      break;

    case 'message':
      // Final answer from the agent
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
      // Backend confirmed the new optimizer state after a set_optimizer message
      setOptimizerUI(data.enabled);
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

  // Hide welcome block on first message
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
// DOM builders — message rows
// ============================================================

/**
 * Append a user message bubble.
 */
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

/**
 * Append an agent message bubble. source is "claude" or "local".
 */
function appendAgentMessage(text, source = 'claude') {
  const row = document.createElement('div');
  row.className = 'msg-row agent-row';
  const localClass = source === 'local' ? ' local-source' : '';
  const sourceLabel = source === 'local' ? 'local' : 'agent';
  row.innerHTML = `
    <span class="msg-role">${escapeHtml(sourceLabel)}</span>
    <div class="msg-bubble${localClass}">${renderMarkdown(text)}</div>
  `;
  chatArea.appendChild(row);
  scrollToBottom();
}

/**
 * Append a red error notice.
 */
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
// DOM builders — tool events
// ============================================================

// Track the last tool-event element per tool name so we can update it with the result
const lastToolEventEl = {};

/**
 * Append a collapsible tool-call card. Result is filled in later by updateLastToolEvent.
 */
function appendToolEvent(toolName, input, result) {
  const icon = toolIcon(toolName);

  const wrapper = document.createElement('div');
  wrapper.className = 'tool-event';
  wrapper.dataset.tool = toolName;

  const inputJson = JSON.stringify(input, null, 2);

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
      <pre>${escapeHtml(inputJson)}</pre>
      <div class="tool-result-body" style="margin-top:8px"></div>
    </div>
  `;

  chatArea.appendChild(wrapper);
  lastToolEventEl[toolName] = wrapper;
  scrollToBottom();
}

/**
 * Fill in the result section of the most recent tool card for this tool.
 */
function updateLastToolEvent(toolName, success, result) {
  const el = lastToolEventEl[toolName];
  if (!el) return;

  const statusEl = el.querySelector('.tool-status');
  const resultBody = el.querySelector('.tool-result-body');

  if (statusEl) {
    statusEl.textContent = success ? ' ✓' : ' ✗';
    statusEl.style.color = success ? 'var(--green)' : 'var(--red)';
  }

  if (resultBody) {
    const resultJson = JSON.stringify(result, null, 2);
    resultBody.innerHTML = `
      <div class="tool-result-status">
        <span class="${success ? 'result-ok' : 'result-fail'}">${success ? '✓ success' : '✗ failed'}</span>
      </div>
      <pre>${escapeHtml(resultJson)}</pre>
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

/**
 * Toggle tool body expansion on header click.
 */
window.toggleToolBody = function(header) {
  const body = header.nextElementSibling;
  const expanded = header.classList.toggle('expanded');
  body.classList.toggle('visible', expanded);
};

/**
 * Return a small emoji for known tool names.
 */
function toolIcon(name) {
  const icons = {
    read_file:      '📄',
    write_file:     '✏️',
    list_directory: '📁',
  };
  return icons[name] || '🔧';
}

// ============================================================
// Prompt optimizer indicator (footer spinner)
// ============================================================

/**
 * Show the "Optimizing…" footer spinner — only when the optimizer is actually on.
 * Uses the `optimizerEnabled` state flag, NOT the badge visibility, so the two
 * are decoupled and can be toggled independently at runtime.
 */
function showOptimizerIndicator() {
  if (optimizerEnabled) {
    optimizerIndicator.classList.remove('hidden');
  }
}

function hideOptimizerIndicator() {
  optimizerIndicator.classList.add('hidden');
}

/**
 * Append a small amber pill showing that the prompt was rewritten.
 * Hovering reveals the before/after.
 */
function appendOptimizerPill(original, optimized) {
  const pill = document.createElement('div');
  pill.className = 'optimizer-pill';
  pill.style.position = 'relative';
  pill.innerHTML = `
    ⚙ prompt optimized
    <div class="optimizer-detail">
      <strong style="color:var(--text-dim)">Original:</strong><br>
      ${escapeHtml(original)}
      <br><br>
      <strong style="color:var(--amber-dim)">Optimized:</strong><br>
      ${escapeHtml(optimized)}
    </div>
  `;
  chatArea.appendChild(pill);
  scrollToBottom();
}

// ============================================================
// Optimizer toggle
// ============================================================

/**
 * Update all optimizer-related UI to reflect `enabled`.
 * Called both from pollStatus (initial load) and from the optimizer_status
 * server event (after a runtime toggle).
 *
 * @param {boolean} enabled
 */
function setOptimizerUI(enabled) {
  optimizerEnabled = enabled;

  // Show / hide both the label badge and the toggle button together
  optimizerBadge.classList.toggle('hidden', false);        // always visible once known
  btnOptimizerToggle.classList.toggle('hidden', false);    // always visible once known

  // Toggle button: label and strikethrough style
  btnOptimizerToggle.textContent = enabled ? 'on' : 'off';
  btnOptimizerToggle.classList.toggle('optimizer-off', !enabled);
  btnOptimizerToggle.title = enabled
    ? 'Optimizer is ON — click to disable'
    : 'Optimizer is OFF — click to enable';
}

/**
 * Click handler for the optimizer toggle button.
 * Sends the new desired state to the backend over WebSocket.
 * The backend echoes back an optimizer_status event which calls setOptimizerUI().
 */
btnOptimizerToggle.addEventListener('click', () => {
  const newState = !optimizerEnabled;
  sendWS({ type: 'set_optimizer', data: { enabled: newState } });
  // Optimistic UI update: reflect immediately, will be confirmed by server event
  setOptimizerUI(newState);
});

// ============================================================
// Confirmation modal
// ============================================================

function openConfirmModal(confirmationId, toolName, input) {
  pendingConfirmationId = confirmationId;
  modalToolName.textContent = `Tool: ${toolName}`;
  modalDetails.textContent = JSON.stringify(input, null, 2);
  confirmModal.classList.remove('hidden');
  btnApprove.focus();
}

function closeConfirmModal() {
  confirmModal.classList.add('hidden');
  pendingConfirmationId = null;
}

btnApprove.addEventListener('click', () => {
  if (pendingConfirmationId) {
    sendWS({ type: 'confirm', confirmation_id: pendingConfirmationId, approved: true });
  }
  closeConfirmModal();
});

btnDeny.addEventListener('click', () => {
  if (pendingConfirmationId) {
    sendWS({ type: 'confirm', confirmation_id: pendingConfirmationId, approved: false });
  }
  closeConfirmModal();
});

// Dismiss modal on backdrop click
confirmModal.addEventListener('click', (e) => {
  if (e.target === confirmModal) {
    btnDeny.click();
  }
});

// ============================================================
// Status + waiting state
// ============================================================

function showStatus(text) {
  statusText.textContent = text;
  statusLine.classList.remove('hidden');
}

function hideStatus() {
  statusLine.classList.add('hidden');
  statusText.textContent = '';
}

function setWaiting(waiting) {
  isWaiting = waiting;
  btnSend.disabled = waiting;
  userInput.disabled = waiting;
  if (!waiting) hideStatus();
}

// ============================================================
// Clear conversation
// ============================================================

function clearChat() {
  // Remove all message rows, tool events, optimizer pills
  const toRemove = chatArea.querySelectorAll(
    '.msg-row, .tool-event, .tool-denied-msg, .optimizer-pill'
  );
  toRemove.forEach(el => el.remove());
  Object.keys(lastToolEventEl).forEach(k => delete lastToolEventEl[k]);

  // Restore welcome block
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
  if (confirm('Clear the conversation history?')) {
    sendWS({ type: 'clear' });
  }
});

// ============================================================
// Status polling — /status endpoint
// ============================================================

async function pollStatus() {
  try {
    const res = await fetch('/status');
    const data = await res.json();

    // Claude API key indicator
    statusClaude.classList.toggle('online',  !!data.claude_api);
    statusClaude.classList.toggle('offline', !data.claude_api);

    // Ollama indicator
    statusOllama.classList.toggle('online',  !!data.ollama);
    statusOllama.classList.toggle('offline', !data.ollama);

    // Model info in the model bar
    if (data.models) {
      modelDisplay.textContent =
        `primary: ${data.models.primary || '?'}  ·  local: ${data.models.local || '?'}`;
    }

    // Initialise the optimizer toggle UI from the server's reported state.
    // setOptimizerUI handles both showing/hiding and labelling the controls.
    setOptimizerUI(!!data.use_prompt_optimizer);

  } catch (e) {
    console.warn('[status] Could not fetch /status:', e);
  }
}

// ============================================================
// Input handling
// ============================================================

btnSend.addEventListener('click', sendMessage);

userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();   // Don't insert newline — use Shift+Enter for that
    sendMessage();
  }
});

// Auto-resize the textarea as user types
userInput.addEventListener('input', () => autoResize(userInput));

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

// ============================================================
// Helpers
// ============================================================

function scrollToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

/**
 * Escape HTML special chars to prevent XSS when inserting untrusted text.
 * Always use this before setting innerHTML with user/agent content.
 */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Minimal markdown renderer for agent responses:
 * Handles **bold**, `code`, and newlines.
 * Not a full parser — Phase 3 could add a real lib like marked.js.
 */
function renderMarkdown(text) {
  return escapeHtml(text)
    // **bold**
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // `inline code`
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Newlines → <br>
    .replace(/\n/g, '<br>');
}

// ============================================================
// Init
// ============================================================

connectWS();
pollStatus();
// Re-check status every 30s (Ollama can be started/stopped while agent is running)
setInterval(pollStatus, 30_000);
