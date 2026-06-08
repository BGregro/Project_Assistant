/**
 * app.js  —  Phase 3 Frontend WebSocket Client
 *
 * Phase 3 additions over Phase 2:
 *   - Two-panel layout: right panel with Tasks / Files / Memory tabs
 *   - Right panel collapse/expand
 *   - task_progress WebSocket event → live step timeline in Tasks tab
 *   - task_stopped event → stop button, badge update, step cancellation
 *   - Stop button (sends stop_task, disables itself)
 *   - Task status badge (idle / thinking / executing / waiting)
 *   - Tool calls/results wrapped in collapsible <details class="tool-block">
 *     tracked by tool-use ID so result attaches to the correct block
 *   - execute_code tool: code input rendered as syntax-highlighted block
 *   - Thinking indicator: shows below last message, updates label per tool
 *   - Memory tab: recent history rows + vector store count from /status
 *   - Copy-to-clipboard buttons on code blocks
 *
 * All Phase 2 functionality is preserved exactly.
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

// Optimizer indicator (footer)
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

// Phase 3: task controls
const btnStop         = document.getElementById('btn-stop');
const taskStatusBadge = document.getElementById('task-status-badge');
const taskBadgeLabel  = document.getElementById('task-badge-label');
const thinkingIndicator = document.getElementById('thinking-indicator');
const thinkingLabel   = document.getElementById('thinking-label');

// Files tab tree
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

// Phase 3 state
let activeTaskSteps = {};    // stepN → DOM element
let pendingToolBlocks = {};  // tool_use_id → <details> element
let isTaskRunning = false;

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
      updateTaskBadge('thinking');
      break;

    case 'prompt_optimized':
      hideOptimizerIndicator();
      if (data) appendOptimizerPill(data.original, data.optimized);
      break;

    case 'tool_call':
      // Update thinking label to reflect active tool
      setThinkingLabel(toolActionLabel(data.tool));
      appendToolBlock(data.tool, data.input, data.tool_use_id);
      updateTaskBadge('executing');
      break;

    case 'tool_result':
      updateToolBlock(data.tool_use_id || data.tool, data.success, data.result);
      updateTaskBadge('thinking');
      break;

    case 'tool_denied':
      appendToolDenied(data.tool);
      break;

    case 'confirm_required':
      openConfirmModal(data.confirmation_id, data.tool, data.input);
      updateTaskBadge('waiting');
      break;

    case 'message':
      hideStatus();
      hideOptimizerIndicator();
      hideThinkingIndicator();
      appendAgentMessage(data.text, data.source);
      setWaiting(false);
      updateTaskBadge('idle');
      stopTaskRunning();
      updateMemoryHistory(data.text, 'agent');
      break;

    case 'error':
      hideStatus();
      hideOptimizerIndicator();
      hideThinkingIndicator();
      appendErrorMessage(data.text);
      setWaiting(false);
      updateTaskBadge('idle');
      stopTaskRunning();
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
      currentPrimaryModel = data.model;
      updateModelDisplay();
      flashAck('ack-primary-model');
      break;

    case 'local_agent_model_status':
      currentLocalModel = data.model;
      updateModelDisplay();
      flashAck('ack-local-agent-model');
      break;

    case 'config_ack':
      flashAckForKey(data.key);
      break;

    // --- Phase 3 new events ---

    case 'task_started':
      // The backend confirmed a task is now running. Show the Stop button
      // and switch the badge to 'running'.  The frontend already called
      // startTaskRunning() in sendMessage(), so this is a belt-and-braces
      // confirmation that both sides agree a task is active.
      handleTaskStarted(data);
      break;

    case 'task_progress':
      handleTaskProgress(data);
      break;

    case 'task_stopped':
      handleTaskStopped(data);
      break;

    // Phase 3e — plan approval card
    case 'task_plan':
      handleTaskPlan(data);
      break;

    // Phase 3h — structured research mode
    case 'research_started':
      handleResearchStarted(data);
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
  if (!text) return;

  // ── Phase 3b: mid-task message injection ─────────────────────────────────
  // If the Stop button is visible, a task is running.  Show a muted "queued"
  // annotation instead of a normal user bubble and let the backend inject the
  // message at the next safe checkpoint.
  if (isTaskRunning) {
    appendQueuedMessage(text);
    sendWS({ type: 'message', text });
    userInput.value = '';
    autoResize(userInput);
    return;
  }
  // ─────────────────────────────────────────────────────────────────────────

  if (isWaiting) return;

  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();

  appendUserMessage(text);
  updateMemoryHistory(text, 'user');
  sendWS({ type: 'message', text });

  userInput.value = '';
  autoResize(userInput);
  setWaiting(true);
  startTaskRunning();
  showOptimizerIndicator();
  showStatus('Sending…');
  showThinkingIndicator('Thinking…');
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
    <div class="msg-bubble${localClass}">${renderMarkdownWithCode(text)}</div>
  `;
  chatArea.appendChild(row);
  // Attach copy buttons to code blocks
  row.querySelectorAll('.code-block').forEach(attachCopyBtn);
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

/**
 * Muted italic annotation shown when the user sends a message while a task
 * is already running.  The backend queues it; the agent reads it at the next
 * checkpoint.
 */
function appendQueuedMessage(text) {
  const row = document.createElement('div');
  row.className = 'msg-row queued-row';
  row.innerHTML = `
    <span class="msg-role" style="opacity:0.45">you</span>
    <div class="msg-bubble queued-bubble">↪ Instruction queued for agent: <em>${escapeHtml(text)}</em></div>
  `;
  chatArea.appendChild(row);
  scrollToBottom();
}

// ============================================================
// Tool blocks — collapsible <details>
// Tracked by tool_use_id (Phase 3) with fallback to tool name (Phase 2 compat)
// ============================================================

/**
 * Map tool name → display category for border color.
 */
function toolCategory(name) {
  if (['search_web', 'fetch_page'].includes(name)) return 'web';
  if (['read_file', 'write_file', 'list_directory', 'analyze_file'].includes(name)) return 'filesystem';
  if (['execute_code'].includes(name)) return 'code';
  if (['get_system_info'].includes(name)) return 'system';
  if (['list_capabilities'].includes(name)) return 'other';
  return 'other';
}

function toolIcon(name) {
  const icons = {
    read_file: '📄', write_file: '✏️', list_directory: '📁',
    analyze_file: '📊', list_capabilities: '🤖',
    search_web: '🔍', fetch_page: '🌐',
    execute_code: '⚡', get_system_info: '💻',
  };
  return icons[name] || '🔧';
}

function toolActionLabel(name) {
  const labels = {
    search_web: 'Searching web…', fetch_page: 'Fetching page…',
    read_file: 'Reading file…', write_file: 'Writing file…',
    list_directory: 'Listing directory…', execute_code: 'Running code…',
    get_system_info: 'Getting system info…', analyze_file: 'Analyzing file…',
  };
  return labels[name] || `Using ${name}…`;
}

/**
 * Render a compact parameters preview string (first 60 chars of first value).
 */
function paramsPreview(input) {
  if (!input) return '';
  const vals = Object.values(input);
  if (!vals.length) return '';
  const first = String(vals[0]);
  const preview = first.length > 60 ? first.slice(0, 57) + '…' : first;
  return `(${preview})`;
}

/**
 * Render the input body of a tool block.
 * For execute_code: highlight the `code` field as a code block.
 */
function renderToolInput(toolName, input) {
  if (toolName === 'execute_code' && input && input.code) {
    const lang = input.language || 'python';
    const restInput = { ...input };
    delete restInput.code;
    const header = Object.keys(restInput).length
      ? `<pre>${escapeHtml(JSON.stringify(restInput, null, 2))}</pre>` : '';
    return `
      ${header}
      <div class="tb-section-label">code (${escapeHtml(lang)})</div>
      <div class="tb-code-block"><pre>${escapeHtml(input.code)}</pre></div>
    `;
  }
  return `<pre>${escapeHtml(JSON.stringify(input, null, 2))}</pre>`;
}

/**
 * Append a new tool block. Called on tool_call event.
 * @param {string} toolName
 * @param {object} input
 * @param {string|null} toolUseId  — may be null for older backend versions
 */
function appendToolBlock(toolName, input, toolUseId) {
  const cat    = toolCategory(toolName);
  const icon   = toolIcon(toolName);
  const params = paramsPreview(input);
  const start  = Date.now();

  const el = document.createElement('details');
  el.className = 'tool-block';
  el.dataset.cat     = cat;
  el.dataset.tool    = toolName;
  el.dataset.startMs = start;
  if (toolUseId) el.dataset.toolUseId = toolUseId;

  el.innerHTML = `
    <summary>
      <span class="tb-arrow">▶</span>
      <span class="tb-icon">${icon}</span>
      <span class="tb-name">${escapeHtml(toolName)}</span>
      <span class="tb-params">${escapeHtml(params)}</span>
      <span class="tb-status">⟳</span>
      <span class="tb-time"></span>
    </summary>
    <div class="tb-body">
      <div class="tb-section-label">INPUT</div>
      ${renderToolInput(toolName, input)}
      <div class="tb-result-area" style="margin-top:8px"></div>
    </div>
  `;

  chatArea.appendChild(el);

  // Register by ID or by tool name as fallback
  if (toolUseId) {
    pendingToolBlocks[toolUseId] = el;
  }
  // Always keep a by-name ref for compat (last wins)
  pendingToolBlocks[toolName] = el;

  scrollToBottom();
  return el;
}

/**
 * Update a tool block with its result. Called on tool_result event.
 * Looks up by tool_use_id first, then by tool name.
 */
function updateToolBlock(idOrName, success, result) {
  const el = pendingToolBlocks[idOrName];
  if (!el) return;

  const elapsed = Date.now() - parseInt(el.dataset.startMs || '0', 10);
  const statusEl = el.querySelector('.tb-status');
  const timeEl   = el.querySelector('.tb-time');
  const resultArea = el.querySelector('.tb-result-area');

  if (statusEl) {
    statusEl.textContent = success ? ' ✓' : ' ✗';
    statusEl.style.color = success ? 'var(--green)' : 'var(--red)';
  }
  if (timeEl) {
    timeEl.textContent = elapsed < 1000 ? `${elapsed}ms` : `${(elapsed/1000).toFixed(1)}s`;
  }

  if (resultArea) {
    const displayResult = { ...result };
    delete displayResult.tree; // tree shown in Files tab
    resultArea.innerHTML = `
      <div class="tb-section-label">RESULT</div>
      <div class="tb-result-row">
        <span class="${success ? 'tb-ok' : 'tb-fail'}">${success ? '✓ success' : '✗ failed'}</span>
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

// ============================================================
// Task progress — right panel Tasks tab
// ============================================================

function handleTaskStarted(data) {
  // Belt-and-braces: ensure Stop button and badge reflect running state.
  // sendMessage() already called startTaskRunning(), but the backend
  // confirmation is the canonical signal that a task is truly active.
  isTaskRunning = true;
  btnStop.classList.remove('hidden');
  btnStop.disabled = false;
  updateTaskBadge('thinking');
}

// ============================================================
// Phase 3h: Research progress tracking
// ============================================================

// State for the active research run
let _researchTotal   = 0;
let _researchCurrent = 0;

function handleResearchStarted(data) {
  // Agent can send { total_questions: N } to initialise the bar early
  const total = data && data.total_questions ? parseInt(data.total_questions) : 0;
  if (total > 0) {
    _researchTotal   = total;
    _researchCurrent = 0;
    _ensureResearchBar(total);
  }
}

/**
 * Create (or return) the research progress bar container inside the Tasks panel.
 * The bar sits above the step timeline so it is always visible during a run.
 */
function _ensureResearchBar(total) {
  const panel = document.getElementById('task-progress-panel');
  if (!panel) return null;
  let bar = document.getElementById('research-progress-bar');
  if (!bar) {
    bar = document.createElement('div');
    bar.id = 'research-progress-bar';
    bar.className = 'research-progress-container';
    bar.innerHTML = `
      <div class="research-progress-label">
        <span id="research-progress-text">Research: 0 / ${total}</span>
      </div>
      <div class="research-progress-track">
        <div class="research-progress-fill" id="research-progress-fill" style="width:0%"></div>
      </div>
    `;
    if (panel.firstChild) {
      panel.insertBefore(bar, panel.firstChild);
    } else {
      panel.appendChild(bar);
    }
  }
  return bar;
}

function _updateResearchBar(current, total) {
  _ensureResearchBar(total);
  const pct  = total > 0 ? Math.round((current / total) * 100) : 0;
  const fill = document.getElementById('research-progress-fill');
  const text = document.getElementById('research-progress-text');
  if (fill) fill.style.width = pct + '%';
  if (text) text.textContent = `Research: ${current} / ${total}`;
}

function _removeResearchBar() {
  const bar = document.getElementById('research-progress-bar');
  if (bar) bar.remove();
  _researchTotal   = 0;
  _researchCurrent = 0;
}

function handleTaskProgress(data) {
  // data: { step: N, label: "...", status: "running|done|failed", elapsed_ms: N }
  const { step, label, status, elapsed_ms } = data;
  const panel = document.getElementById('task-progress-panel');
  const placeholder = document.getElementById('task-placeholder');

  // Remove placeholder on first step
  if (placeholder) placeholder.remove();

  const stepKey = `step-${step}`;
  let stepEl = activeTaskSteps[stepKey];

  if (!stepEl) {
    // Create new step row
    stepEl = document.createElement('div');
    stepEl.className = 'task-step pending';
    stepEl.id = stepKey;
    stepEl.innerHTML = `
      <div class="step-dot"></div>
      <div class="step-body">
        <div><span class="step-num">${step}.</span><span class="step-label">${escapeHtml(label)}</span></div>
        <div class="step-time"></div>
      </div>
    `;
    panel.appendChild(stepEl);
    activeTaskSteps[stepKey] = stepEl;
  }

  // Update status class
  stepEl.className = `task-step ${status}`;

  // Update elapsed time
  const timeEl = stepEl.querySelector('.step-time');
  if (timeEl && elapsed_ms != null) {
    timeEl.textContent = elapsed_ms < 1000
      ? `${elapsed_ms}ms`
      : `${(elapsed_ms / 1000).toFixed(1)}s`;
  }

  // Phase 3h: detect research progress labels ("Researching: N/T")
  if (label && /^Researching:\s*\d+\/\d+/i.test(label)) {
    const match = label.match(/(\d+)\/(\d+)/);
    if (match) {
      _researchCurrent = parseInt(match[1]);
      _researchTotal   = parseInt(match[2]);
      _updateResearchBar(_researchCurrent, _researchTotal);
    }
  }

  // Switch to tasks tab if not already visible
  const tasksTab = document.getElementById('tab-tasks');
  if (tasksTab && tasksTab.classList.contains('hidden')) {
    switchTab('tasks');
  }
}

function handleTaskStopped(data) {
  _removeResearchBar();  // Phase 3h
  // Mark last running step as cancelled/failed
  Object.values(activeTaskSteps).forEach(el => {
    if (el.classList.contains('running')) {
      el.className = 'task-step cancelled';
    }
  });

  stopTaskRunning();
  updateTaskBadge('idle');
  hideThinkingIndicator();
  setWaiting(false);

  const reason = data && data.reason;

  if (reason === 'user_cancelled') {
    // Append a muted annotation in chat
    const row = document.createElement('div');
    row.className = 'msg-row agent-row';
    row.innerHTML = `
      <span class="msg-role" style="opacity:0.45">sys</span>
      <div class="msg-bubble" style="opacity:0.6;font-style:italic;">↪ Task cancelled.</div>
    `;
    chatArea.appendChild(row);
    scrollToBottom();

  } else if (reason === 'error') {
    const errText = (data && data.error) ? data.error : 'Unknown error.';
    const row = document.createElement('div');
    row.className = 'msg-row agent-row';
    row.innerHTML = `
      <span class="msg-role" style="color:var(--red)">sys</span>
      <div class="msg-bubble" style="border-color:var(--red);color:var(--red);">↪ Task failed: ${escapeHtml(errText)}</div>
    `;
    chatArea.appendChild(row);
    scrollToBottom();
  }
  // reason === 'complete': agent already sent its final answer via 'message' event
}

// ============================================================
// Phase 3e — Plan approval card
// ============================================================

function handleTaskPlan(data) {
  const { plan_id, steps } = data;
  if (!steps || !steps.length) return;

  const card = document.createElement('div');
  card.className = 'plan-card';
  card.dataset.planId = plan_id;

  // Header
  const header = document.createElement('div');
  header.className = 'plan-card-header';
  header.innerHTML = `
    <span class="plan-card-icon">📋</span>
    <span class="plan-card-title">Proposed Plan</span>
    <span class="plan-card-count">${steps.length} step${steps.length !== 1 ? 's' : ''}</span>
  `;
  card.appendChild(header);

  // Steps list
  const stepsList = document.createElement('div');
  stepsList.className = 'plan-steps';

  steps.forEach((s) => {
    const row = document.createElement('div');
    row.className = 'plan-step-row';
    row.dataset.step = s.step;
    row.innerHTML = `
      <div class="plan-step-num">${s.step}</div>
      <div class="plan-step-body">
        <div class="plan-step-action">
          <span class="plan-step-label" contenteditable="true" spellcheck="false"
                data-step="${s.step}" data-field="action">${escapeHtml(s.action)}</span>
        </div>
        <div class="plan-step-details">${escapeHtml(s.details)}</div>
      </div>
    `;
    stepsList.appendChild(row);
  });
  card.appendChild(stepsList);

  // Actions row
  const actions = document.createElement('div');
  actions.className = 'plan-card-actions';

  const btnRun = document.createElement('button');
  btnRun.className = 'plan-btn plan-btn-run';
  btnRun.innerHTML = '▶ Run Plan';

  const btnCancel = document.createElement('button');
  btnCancel.className = 'plan-btn plan-btn-cancel';
  btnCancel.innerHTML = '✕ Cancel';

  actions.appendChild(btnCancel);
  actions.appendChild(btnRun);
  card.appendChild(actions);

  // Run Plan click — collect (possibly edited) actions
  btnRun.addEventListener('click', () => {
    btnRun.disabled = true;
    btnCancel.disabled = true;

    const editedSteps = steps.map((s) => {
      const labelEl = card.querySelector(`[data-step="${s.step}"][data-field="action"]`);
      const editedAction = labelEl ? labelEl.textContent.trim() : s.action;
      // Only include if the user actually changed something
      return { ...s, action: editedAction };
    });

    // Check if any step was actually edited
    const wasEdited = editedSteps.some((s, i) => s.action !== steps[i].action);

    sendWS({
      type: 'plan_response',
      plan_id,
      approved: true,
      edited_steps: wasEdited ? editedSteps : null,
    });

    // Collapse to a muted confirmation line
    actions.innerHTML = '<span class="plan-submitted-note">Plan approved ✓ — running…</span>';
    // Make steps non-editable
    card.querySelectorAll('[contenteditable]').forEach(el => {
      el.contentEditable = 'false';
    });
  });

  // Cancel click
  btnCancel.addEventListener('click', () => {
    btnRun.disabled = true;
    btnCancel.disabled = true;

    sendWS({ type: 'plan_response', plan_id, approved: false, edited_steps: null });

    actions.innerHTML = '<span class="plan-submitted-note plan-cancelled-note">Plan cancelled</span>';
    card.querySelectorAll('[contenteditable]').forEach(el => {
      el.contentEditable = 'false';
    });
  });

  chatArea.appendChild(card);
  scrollToBottom();
}

function resetTaskPanel() {
  activeTaskSteps = {};
  const panel = document.getElementById('task-progress-panel');
  if (panel) {
    panel.innerHTML = '<div id="task-placeholder" class="tab-placeholder">No active task.</div>';
  }
}

// ============================================================
// Task running state (controls Stop button + badge)
// ============================================================

function startTaskRunning() {
  isTaskRunning = true;
  btnStop.classList.remove('hidden');
  btnStop.disabled = false;
  updateTaskBadge('thinking');
  // Reset task panel for new task
  resetTaskPanel();
  pendingToolBlocks = {};
}

function stopTaskRunning() {
  isTaskRunning = false;
  btnStop.classList.add('hidden');
  btnStop.disabled = false;
}

// ============================================================
// Stop button
// ============================================================

btnStop.addEventListener('click', () => {
  btnStop.disabled = true; // prevent double-send
  sendWS({ type: 'stop_task' });
});

// ============================================================
// Task status badge
// ============================================================

function updateTaskBadge(state) {
  if (state === 'idle') {
    taskStatusBadge.classList.add('hidden');
    taskStatusBadge.dataset.state = 'idle';
    return;
  }
  taskStatusBadge.classList.remove('hidden');
  taskStatusBadge.dataset.state = state;
  taskBadgeLabel.textContent = state;
}

// ============================================================
// Thinking indicator
// ============================================================

function showThinkingIndicator(label) {
  thinkingLabel.textContent = label || 'Thinking…';
  thinkingIndicator.classList.remove('hidden');
  scrollToBottom();
}

function hideThinkingIndicator() {
  thinkingIndicator.classList.add('hidden');
}

function setThinkingLabel(label) {
  thinkingLabel.textContent = label;
  showThinkingIndicator(label);
}

// ============================================================
// Right panel — collapse / expand + tab switching
// ============================================================

window.toggleRightPanel = function() {
  const panel = document.getElementById('right-panel');
  const btn   = document.getElementById('panel-toggle-btn');
  const appBody = document.getElementById('app-body');
  const collapsed = panel.classList.toggle('collapsed');
  btn.textContent = collapsed ? '▶' : '◀';
  // Adjust grid on the parent
  appBody.style.gridTemplateColumns = collapsed
    ? `1fr ${getComputedStyle(document.documentElement).getPropertyValue('--panel-toggle-w').trim()} 0px`
    : '';
};

window.switchTab = function(tabName) {
  // Hide all tabs
  document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

  // Show selected
  const tab = document.getElementById(`tab-${tabName}`);
  if (tab) tab.classList.remove('hidden');
  const btn = document.querySelector(`.tab-btn[data-tab="${tabName}"]`);
  if (btn) btn.classList.add('active');
};

window.expandAndTab = function(tabName) {
  const panel = document.getElementById('right-panel');
  if (panel.classList.contains('collapsed')) {
    toggleRightPanel();
  }
  switchTab(tabName);
};

// ============================================================
// Memory tab — update recent history
// ============================================================

const recentHistory = []; // { role, text }[] last 5

function updateMemoryHistory(text, role) {
  recentHistory.push({ role, text });
  if (recentHistory.length > 5) recentHistory.shift();
  renderMemoryHistory();
}

function renderMemoryHistory() {
  const list = document.getElementById('memory-history-list');
  if (!list) return;
  if (!recentHistory.length) {
    list.innerHTML = '<div class="tab-placeholder">No history yet.</div>';
    return;
  }
  list.innerHTML = recentHistory.map(({ role, text }) => {
    const snippet = text.length > 80 ? text.slice(0, 77) + '…' : text;
    return `
      <div class="memory-history-row">
        <span class="memory-role ${role}">${escapeHtml(role)}</span>
        <span class="memory-snippet">${escapeHtml(snippet)}</span>
      </div>
    `;
  }).join('');
}

// ============================================================
// Files tab — project tree
// ============================================================

function updateTreePanel(treeText) {
  if (!treeText) return;
  const ph = document.getElementById('tree-placeholder');
  if (ph) ph.remove();
  treeContent.textContent = treeText;
  // Optionally switch to Files tab on first update
  // switchTab('files');
}

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
// Optimizer state
// ============================================================

function setOptimizerState(enabled) {
  optimizerEnabled = enabled;
  toggleOptimizer.classList.toggle('on', enabled);
  toggleOptimizerLabel.textContent = enabled ? 'on' : 'off';
  optimizerBadge.textContent = enabled ? '⚙ optimizer: on' : '⚙ optimizer: off';
  optimizerBadge.classList.remove('hidden');
}

toggleOptimizer.addEventListener('click', () => {
  const next = !optimizerEnabled;
  sendWS({ type: 'set_optimizer', data: { enabled: next } });
  setOptimizerState(next);
});

// ============================================================
// Local mode state
// ============================================================

function applyLocalMode(enabled) {
  localModeEnabled = enabled;
  btnModeOnline.classList.toggle('active-online', !enabled);
  btnModeLocal.classList.toggle('active-local',   enabled);
  selPrimaryModel.disabled = enabled;
  const primaryRow = document.getElementById('row-primary-model');
  if (primaryRow) primaryRow.style.opacity = enabled ? '0.4' : '1';
  updateModelDisplay();
  const claudeLabel = statusClaude.querySelector('.status-label');
  if (claudeLabel) claudeLabel.textContent = enabled ? 'local' : 'claude';
  localModeBadge.textContent = enabled ? '🖥 local: on' : '🖥 local: off';
  localModeBadge.classList.remove('hidden');
}

function updateModelDisplay() {
  if (localModeEnabled) {
    modelDisplay.textContent = `local: ${currentLocalModel}`;
  } else {
    modelDisplay.textContent = `claude: ${currentPrimaryModel}`;
  }
  modeModelLabel.textContent = localModeEnabled ? currentLocalModel : currentPrimaryModel;
}

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

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && settingsPanel.classList.contains('open')) closeSettings();
});

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
});

selLocalAgentModel.addEventListener('change', () => {
  const model = selLocalAgentModel.value;
  currentLocalModel = model;
  sendWS({ type: 'set_local_agent_model', data: { model } });
  if (localModeEnabled) updateModelDisplay();
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

const numberInputMap = [
  { id: 'inp-recent-turns',      key: 'context.recent_turns',            parse: parseInt  },
  { id: 'inp-summary-threshold', key: 'context.summary_threshold',       parse: parseInt  },
  { id: 'inp-retrieval-n',       key: 'context.retrieval_n',             parse: parseInt  },
  { id: 'inp-max-history',       key: 'context.max_history_turns',       parse: parseInt  },
  { id: 'inp-max-iterations',    key: 'context.max_iterations_per_turn', parse: parseInt  },
  { id: 'inp-local-timeout',     key: 'local_agent_timeout',             parse: parseFloat },
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

inpSimilarity.addEventListener('input', () => {
  const v = parseFloat(inpSimilarity.value);
  valSimilarity.textContent = v.toFixed(2);
  clearTimeout(_debounceTimers['similarity']);
  _debounceTimers['similarity'] = setTimeout(() => {
    sendWS({ type: 'set_config', data: { key: 'context.similarity_cutoff', value: v } });
  }, 300);
});

function sendTreeRoot() {
  const v = inpTreeRoot.value.trim();
  if (v) sendWS({ type: 'set_config', data: { key: 'tree_root', value: v } });
}
inpTreeRoot.addEventListener('blur', sendTreeRoot);
inpTreeRoot.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); sendTreeRoot(); } });

// ============================================================
// ACK flash helpers
// ============================================================

const keyToAckId = {
  'context.recent_turns':            'ack-recent-turns',
  'context.summary_threshold':       'ack-summary-threshold',
  'context.retrieval_n':             'ack-retrieval-n',
  'context.similarity_cutoff':       'ack-similarity',
  'context.max_history_turns':       'ack-max-history',
  'context.max_iterations_per_turn': 'ack-max-iterations',
  'local_agent_timeout':             'ack-local-timeout',
  'embeddings.enabled':              null,
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
  if (data.primary_model) {
    currentPrimaryModel = data.primary_model;
    if (selPrimaryModel) {
      const existing = [...selPrimaryModel.options].find(o => o.value === data.primary_model);
      if (existing) existing.selected = true;
    }
  }
  if (data.local_agent_model) {
    currentLocalModel = data.local_agent_model;
    if (selLocalAgentModel) {
      const existing = [...selLocalAgentModel.options].find(o => o.value === data.local_agent_model);
      if (existing) existing.selected = true;
    }
  }

  setOptimizerState(!!data.use_prompt_optimizer);
  applyLocalMode(!!data.local_mode);

  const ctx = data.context || {};
  if (inpRecentTurns      && ctx.recent_turns      != null) inpRecentTurns.value      = ctx.recent_turns;
  if (inpSummaryThreshold && ctx.summary_threshold  != null) inpSummaryThreshold.value = ctx.summary_threshold;
  if (inpRetrievalN       && ctx.retrieval_n        != null) inpRetrievalN.value       = ctx.retrieval_n;
  if (inpSimilarity       && ctx.similarity_cutoff  != null) {
    inpSimilarity.value = ctx.similarity_cutoff;
    valSimilarity.textContent = Number(ctx.similarity_cutoff).toFixed(2);
  }
  if (inpMaxHistory    && ctx.max_history_turns          != null) inpMaxHistory.value    = ctx.max_history_turns;
  if (inpMaxIterations && ctx.max_iterations_per_turn    != null) inpMaxIterations.value = ctx.max_iterations_per_turn;
  if (inpLocalTimeout  && data.local_agent_timeout       != null) inpLocalTimeout.value  = data.local_agent_timeout;

  const emb = data.embeddings || {};
  if (emb.enabled != null) {
    toggleEmbeddings.classList.toggle('on', !!emb.enabled);
    toggleEmbeddingsLbl.textContent = emb.enabled ? 'on' : 'off';
  }
  if (inpTreeRoot && data.tree_root != null) inpTreeRoot.value = data.tree_root;

  // Phase 3: populate memory tab vector count
  if (data.embeddings_count != null) {
    const el = document.getElementById('memory-vector-status');
    if (el) el.textContent = `Semantic memory: ${data.embeddings_count} entries`;
  }

  // Phase 3g: show user profile load status in memory tab
  const profileEl = document.getElementById('mem-profile-status');
  if (profileEl) {
    if (data.profile_loaded === true) {
      profileEl.textContent = 'Profile: loaded ✓';
      profileEl.style.color = 'var(--green)';
    } else {
      profileEl.textContent = 'Profile: not found ⚠';
      profileEl.style.color = 'var(--yellow)';
    }
  }
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
  chatArea.querySelectorAll(
    '.msg-row, .tool-block, .tool-denied-msg, .optimizer-pill'
  ).forEach(el => el.remove());
  pendingToolBlocks = {};
  activeTaskSteps   = {};
  recentHistory.length = 0;
  renderMemoryHistory();
  resetTaskPanel();

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
// Phase 3f: Long-term memory counts from /memory endpoint
// ============================================================

async function fetchMemoryCounts() {
  try {
    const res  = await fetch('/memory');
    const data = await res.json();
    const tasksEl    = document.getElementById('mem-tasks-count');
    const factsEl    = document.getElementById('mem-facts-count');
    const researchEl = document.getElementById('mem-research-count');
    if (tasksEl)    tasksEl.textContent    = `Tasks logged: ${(data.tasks    || []).length}`;
    if (factsEl)    factsEl.textContent    = `Facts stored: ${(data.facts    || []).length}`;
    if (researchEl) researchEl.textContent = `Research entries: ${(data.research || []).length}`;
  } catch (e) {
    console.warn('[memory] Could not fetch /memory:', e);
  }
}

// ============================================================
// Status polling — /status endpoint
// ============================================================

async function pollStatus() {
  try {
    const res  = await fetch('/status');
    const data = await res.json();

    statusClaude.classList.toggle('online',  !!data.claude_api);
    statusClaude.classList.toggle('offline', !data.claude_api);
    statusOllama.classList.toggle('online',  !!data.ollama);
    statusOllama.classList.toggle('offline', !data.ollama);

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
// Copy-to-clipboard for code blocks
// ============================================================

function attachCopyBtn(codeBlockEl) {
  const btn = document.createElement('button');
  btn.className = 'copy-btn';
  btn.textContent = 'copy';
  btn.addEventListener('click', () => {
    const pre = codeBlockEl.querySelector('pre');
    if (!pre) return;
    navigator.clipboard.writeText(pre.textContent).then(() => {
      btn.textContent = 'copied!';
      setTimeout(() => { btn.textContent = 'copy'; }, 1500);
    });
  });
  codeBlockEl.appendChild(btn);
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

/**
 * Minimal markdown renderer that also handles fenced code blocks.
 * Code blocks become .code-block divs with copy button support.
 */
function renderMarkdownWithCode(text) {
  // Split on fenced code blocks first
  const parts = text.split(/(```[\s\S]*?```)/g);
  return parts.map(part => {
    if (part.startsWith('```')) {
      const lines  = part.slice(3, -3).split('\n');
      const lang   = lines[0].trim() || 'code';
      const code   = lines.slice(1).join('\n');
      return `<div class="code-block"><pre>${escapeHtml(code)}</pre></div>`;
    }
    // Inline markdown
    return escapeHtml(part)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>');
  }).join('');
}

// ============================================================
// Init
// ============================================================

async function init() {
  connectWS();
  pollStatus();
  setInterval(pollStatus, 30_000);

  // Phase 3f: fetch long-term memory counts on load and every 60 seconds.
  fetchMemoryCounts();
  setInterval(fetchMemoryCounts, 60_000);

  // Phase 3b: on page load, check whether the last task was interrupted.
  // Show a dismissible warning banner so the user knows they can resume
  // or start fresh — we never auto-resume because the agent might have
  // been mid-destructive-operation.
  try {
    const res  = await fetch('/task');
    const task = await res.json();
    if (task && task.status === 'running') {
      showInterruptedBanner(task);
    }
  } catch (e) {
    // /task unavailable (server not yet ready on first load) — ignore
  }
}

/**
 * Show a yellow dismissible banner at the top of the chat area when a
 * previously running task was interrupted (server was restarted while a
 * task was active, leaving status="running" on disk).
 */
function showInterruptedBanner(task) {
  const existing = document.getElementById('interrupted-banner');
  if (existing) return;  // don't show twice

  const banner = document.createElement('div');
  banner.id = 'interrupted-banner';
  banner.className = 'interrupted-banner';

  const preview = task.initial_message
    ? ` ("${escapeHtml(task.initial_message.slice(0, 60))}${task.initial_message.length > 60 ? '…' : ''}")`
    : '';

  banner.innerHTML = `
    <span>⚠ A previous task was interrupted${preview}. Ask the agent to continue or start fresh.</span>
    <button class="banner-dismiss" onclick="document.getElementById('interrupted-banner').remove()">✕</button>
  `;

  // Insert before the first child of chatArea, or append if empty
  if (chatArea.firstChild) {
    chatArea.insertBefore(banner, chatArea.firstChild);
  } else {
    chatArea.appendChild(banner);
  }
}

init();
