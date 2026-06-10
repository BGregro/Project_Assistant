/**
 * app.js  —  Phase 4.5 Frontend
 *
 * Changes from Phase 3 / 4:
 *   - Header toolbar: connection dot (green/amber/red), icon buttons only
 *   - Status bar (below input): shows last status event, auto-clears in 8s
 *     Status events no longer render as chat bubbles
 *   - Task-run containers: tool blocks + final reply grouped under a
 *     collapsible header (goal / badge / step-count / elapsed)
 *   - Task history: loaded from GET /task?history=10, shown in Tasks tab
 *   - State restore on reconnect: last complete/cancelled task restores
 *     step timeline; interrupted-banner for status="running"
 *   - Settings panel gains "Active model" read-only display
 *   - All Phase 3 / 4 functionality fully preserved
 */

'use strict';

// ============================================================
// DOM references — core
// ============================================================
const chatArea   = document.getElementById('chat-area');
const userInput  = document.getElementById('user-input');
const btnSend    = document.getElementById('btn-send');
const btnClear   = document.getElementById('btn-clear');

// Connection dot (new header)
const connDot = document.getElementById('conn-dot');

// Status bar (new, below input)
const statusBarDot  = document.getElementById('status-bar-dot');
const statusBarText = document.getElementById('status-bar-text');

// Stop button (now in header toolbar)
const btnStop = document.getElementById('btn-stop');

// Legacy refs kept so nothing crashes
const statusClaude = document.getElementById('status-claude') || {};
const statusOllama = document.getElementById('status-ollama') || {};

// Optimizer indicator (chat area)
const optimizerIndicator = document.getElementById('optimizer-indicator');

// Thinking indicator
const thinkingIndicator = document.getElementById('thinking-indicator');
const thinkingLabel     = document.getElementById('thinking-label');

// Files tab tree
const treeContent = document.getElementById('tree-content');

// Settings panel
const settingsPanel    = document.getElementById('settings-panel');
const settingsBackdrop = document.getElementById('settings-backdrop');
const btnSettingsOpen  = document.getElementById('btn-settings');
const btnSettingsClose = document.getElementById('btn-settings-close');

const btnModeOnline  = document.getElementById('btn-mode-online');
const btnModeLocal   = document.getElementById('btn-mode-local');
const modeModelLabel = document.getElementById('mode-active-model-name');
const activeModelDisplay = document.getElementById('active-model-display');

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
let pendingConfirmationId = null;
let isWaiting             = false;
let optimizerEnabled      = false;
let localModeEnabled      = false;
let currentPrimaryModel   = 'claude-haiku-4-5';
let currentLocalModel     = 'qwen2.5:14b';

// Phase 3 state
let activeTaskSteps  = {};   // stepN → DOM element (right-panel timeline)
let pendingToolBlocks = {};  // tool_use_id → <details> element
let isTaskRunning    = false;

// Phase 4.5 — task run container tracking
let activeTaskContainer  = null;   // current .task-run-container DOM element
let taskRunStartMs       = 0;      // Date.now() when task started
let taskRunStepCount     = 0;      // total steps received for current run
let currentTaskGoal      = '';     // first 60 chars of triggering message
let statusBarTimer       = null;   // auto-clear timer for status bar

// Phase 3h — research progress
let _researchTotal   = 0;
let _researchCurrent = 0;

// ============================================================
// WebSocket — connection + reconnect
// ============================================================

function connectWS() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${location.host}/ws`);

  setConnDot('reconnecting');

  ws.addEventListener('open', () => {
    console.log('[ws] Connected.');
    setConnDot('connected');
    setWaiting(false);
    if (!isTaskRunning) setStatusBar(STATUS_IDLE_TEXT, 'idle');
  });
  ws.addEventListener('message', (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { console.error('[ws] Bad JSON:', e.data); return; }
    handleServerEvent(msg.type, msg.data);
  });
  ws.addEventListener('close', () => {
    console.warn('[ws] Disconnected. Reconnecting…');
    setConnDot('reconnecting');
    setTimeout(connectWS, 2000);
  });
  ws.addEventListener('error', (e) => {
    console.error('[ws] Error:', e);
    setConnDot('disconnected');
  });
}

function sendWS(payload) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(payload));
  } else {
    console.warn('[ws] Not connected, message dropped.');
  }
}

// Connection dot: 'connected' | 'reconnecting' | 'disconnected'
function setConnDot(state) {
  if (!connDot) return;
  connDot.className = 'conn-dot ' + state;
}

// ============================================================
// Status bar — replaces inline status bubbles
// ============================================================

const STATUS_IDLE_TEXT = 'Idle';
const STATUS_IDLE_MS   = 60000;

function setStatusBar(text, mode = 'active') {
  // mode: 'active' (amber), 'error' (red), 'done' (green), 'idle'
  if (!statusBarDot || !statusBarText) return;
  clearTimeout(statusBarTimer);
  statusBarText.textContent = text || STATUS_IDLE_TEXT;
  statusBarDot.className = 'status-bar-dot ' + (mode === 'idle' ? '' : mode);
  if (mode !== 'idle') {
    statusBarTimer = setTimeout(() => setStatusBar(STATUS_IDLE_TEXT, 'idle'), STATUS_IDLE_MS);
  }
}

function clearStatusBar() {
  setStatusBar(STATUS_IDLE_TEXT, 'idle');
}

// ============================================================
// Server event dispatcher
// ============================================================

function handleServerEvent(type, data) {
  switch (type) {

    // Status events → status bar only, never a chat bubble
    case 'status':
      setStatusBar(data.text, 'active');
      break;

    case 'prompt_optimized':
      hideOptimizerIndicator();
      if (data) appendOptimizerPill(data.original, data.optimized);
      break;

    case 'tool_call':
      setThinkingLabel(toolActionLabel(data.tool));
      setStatusBar(toolActionLabel(data.tool), 'active');
      appendToolBlock(data.tool, data.input, data.tool_use_id);
      break;

    case 'tool_result':
      updateToolBlock(data.tool_use_id || data.tool, data.success, data.result);
      break;

    case 'execution_output':
      appendExecutionOutputLine(data.line, data.stream);
      break;

    case 'tool_denied':
      appendToolDenied(data.tool);
      break;

    case 'confirm_required':
      openConfirmModal(data.confirmation_id, data.tool, data.input);
      setStatusBar('Waiting for approval…', 'active');
      break;

    case 'message':
      hideOptimizerIndicator();
      hideThinkingIndicator();
      appendAgentMessage(data.text, data.source);
      setWaiting(false);
      closeTaskContainer('complete');
      updateMemoryHistory(data.text, 'agent');
      setStatusBar('Done', 'done');
      break;

    case 'error':
      hideOptimizerIndicator();
      hideThinkingIndicator();
      appendErrorMessage(data.text);
      setWaiting(false);
      closeTaskContainer('failed');
      setStatusBar('Error: ' + (data.text || '').slice(0, 80), 'error');
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

    case 'task_started':
      handleTaskStarted(data);
      break;

    case 'task_progress':
      handleTaskProgress(data);
      break;

    case 'task_stopped':
      handleTaskStopped(data);
      break;

    case 'task_plan':
      handleTaskPlan(data);
      break;

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

  // Mid-task message injection
  if (isTaskRunning) {
    appendQueuedMessage(text);
    sendWS({ type: 'message', text });
    userInput.value = '';
    autoResize(userInput);
    return;
  }

  if (isWaiting) return;

  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();

  appendUserMessage(text);
  updateMemoryHistory(text, 'user');
  sendWS({ type: 'message', text });

  userInput.value = '';
  autoResize(userInput);
  setWaiting(true);
  currentTaskGoal = text.slice(0, 60) + (text.length > 60 ? '…' : '');
  taskRunStartMs  = Date.now();
  taskRunStepCount = 0;
  startTaskRunning();
  showOptimizerIndicator();
  showThinkingIndicator('Thinking…');
  setStatusBar('Sending…', 'active');
}

// ============================================================
// Message row builders
// ============================================================

function appendUserMessage(text) {
  const target = _currentChatTarget();
  const row = document.createElement('div');
  row.className = 'msg-row user-row';
  row.innerHTML = `
    <span class="msg-role">you</span>
    <div class="msg-bubble">${escapeHtml(text)}</div>
  `;
  target.appendChild(row);
  scrollToBottom();
}

function appendAgentMessage(text, source = 'claude') {
  // Final agent replies always go directly to chatArea — never inside a
  // task container body.  The task body uses display:none when collapsed,
  // so replies buried there become invisible.  Appending to chatArea
  // places the reply below the task container and keeps it always visible.
  const row = document.createElement('div');
  row.className = 'msg-row agent-row';
  const localClass  = source === 'local' ? ' local-source' : '';
  const sourceLabel = source === 'local' ? 'local' : 'agent';
  row.innerHTML = `
    <span class="msg-role">${escapeHtml(sourceLabel)}</span>
    <div class="msg-bubble${localClass}">${renderMarkdownWithCode(text)}</div>
  `;
  chatArea.appendChild(row);
  row.querySelectorAll('.code-block').forEach(attachCopyBtn);
  scrollToBottom();
}

function appendErrorMessage(text) {
  const target = _currentChatTarget();
  const row = document.createElement('div');
  row.className = 'msg-row agent-row';
  row.innerHTML = `
    <span class="msg-role" style="color:var(--red)">err</span>
    <div class="msg-bubble" style="border-color:var(--red);color:var(--red);">${escapeHtml(text)}</div>
  `;
  target.appendChild(row);
  scrollToBottom();
}

function appendQueuedMessage(text) {
  // Mid-task queued messages go inside the active task container if open
  const target = activeTaskContainer
    ? activeTaskContainer.querySelector('.task-run-body')
    : chatArea;
  const row = document.createElement('div');
  row.className = 'msg-row queued-row';
  row.innerHTML = `
    <span class="msg-role" style="opacity:0.45">you</span>
    <div class="msg-bubble queued-bubble">↪ Queued: <em>${escapeHtml(text)}</em></div>
  `;
  target.appendChild(row);
  scrollToBottom();
}

/**
 * Return the DOM node where new chat content should be appended.
 * During an active task, content goes inside the task container's body.
 * Outside a task (simple Q&A), content goes directly into chatArea.
 */
function _currentChatTarget() {
  if (activeTaskContainer) {
    return activeTaskContainer.querySelector('.task-run-body');
  }
  return chatArea;
}

// ============================================================
// Task-run containers
// Phase 4.5: group tool blocks + final reply under a collapsible header.
// ============================================================

/**
 * Create a new task-run container and append it to chatArea.
 * Called when task_started is received.
 */
function createTaskContainer(goal) {
  const el = document.createElement('div');
  el.className = 'task-run-container open';

  // Build header
  const header = document.createElement('div');
  header.className = 'task-run-header';
  header.innerHTML = `
    <span class="task-run-arrow">▶</span>
    <span class="task-run-goal">${escapeHtml(goal)}</span>
    <span class="task-run-badge running">running</span>
    <span class="task-run-meta">
      <span class="task-run-steps">0 steps</span>
    </span>
  `;
  header.addEventListener('click', () => toggleTaskContainer(el));
  el.appendChild(header);

  // Body
  const body = document.createElement('div');
  body.className = 'task-run-body';
  el.appendChild(body);

  chatArea.appendChild(el);
  activeTaskContainer = el;
  scrollToBottom();
  return el;
}

function toggleTaskContainer(el) {
  el.classList.toggle('open');
}

/**
 * Update the container's step count and elapsed time display.
 */
function updateTaskContainerMeta() {
  if (!activeTaskContainer) return;
  const stepsEl = activeTaskContainer.querySelector('.task-run-steps');
  if (stepsEl) {
    stepsEl.textContent = `${taskRunStepCount} step${taskRunStepCount !== 1 ? 's' : ''}`;
  }
  // Also update elapsed (shown while running)
  const elapsed = Date.now() - taskRunStartMs;
  const elapsedStr = elapsed < 60000
    ? `${(elapsed / 1000).toFixed(0)}s`
    : `${Math.floor(elapsed / 60000)}m ${Math.floor((elapsed % 60000) / 1000)}s`;
  let metaEl = activeTaskContainer.querySelector('.task-run-elapsed');
  if (!metaEl) {
    metaEl = document.createElement('span');
    metaEl.className = 'task-run-elapsed';
    const metaContainer = activeTaskContainer.querySelector('.task-run-meta');
    if (metaContainer) metaContainer.appendChild(metaEl);
  }
  metaEl.textContent = elapsedStr;
}

/**
 * Close the active task container with a final status.
 * status: 'complete' | 'failed' | 'cancelled'
 * Called when 'message', 'error', or 'task_stopped' arrives.
 */
function closeTaskContainer(status) {
  if (!activeTaskContainer) return;

  const badge = activeTaskContainer.querySelector('.task-run-badge');
  if (badge) {
    badge.className = 'task-run-badge ' + status;
    badge.textContent = status;
  }

  // Freeze elapsed time
  updateTaskContainerMeta();

  // Collapse after completion (10s delay so user can read the final output)
  if (status === 'complete') {
    setTimeout(() => {
      if (activeTaskContainer) activeTaskContainer.classList.remove('open');
    }, 10000);
  }

  stopTaskRunning();
  activeTaskContainer = null;
}

// ============================================================
// Tool blocks
// ============================================================

function toolCategory(name) {
  if (['search_web', 'fetch_page'].includes(name)) return 'web';
  if (['read_file', 'write_file', 'list_directory', 'analyze_file',
       'list_outputs', 'patch_file'].includes(name)) return 'filesystem';
  if (['execute_code', 'install_package'].includes(name)) return 'code';
  if (['get_system_info'].includes(name)) return 'system';
  if (['log_research', 'recall_memory', 'log_fact', 'recall_projects',
       'read_user_profile', 'update_user_profile'].includes(name)) return 'memory';
  if (['list_capabilities'].includes(name)) return 'other';
  return 'other';
}

function toolIcon(name) {
  const icons = {
    read_file: '📄', write_file: '✏️', list_directory: '📁', patch_file: '🩹',
    analyze_file: '📊', list_capabilities: '🤖', list_outputs: '📦',
    search_web: '🔍', fetch_page: '🌐',
    execute_code: '⚡', install_package: '📥', get_system_info: '💻',
    log_research: '💾', recall_memory: '🧠', log_fact: '📌', recall_projects: '🗂',
    read_user_profile: '👤', update_user_profile: '✏',
    scaffold_project: '🏗', get_project_status: '📊', mark_file_complete: '✅',
    run_project_test: '🧪', deep_research: '🔬',
    browser_open: '🌏', browser_read: '📖', browser_screenshot: '📸',
    write_tool: '🔨', reload_tool: '🔄', scan_system: '🖥',
  };
  return icons[name] || '🔧';
}

function toolActionLabel(name) {
  const labels = {
    search_web: 'Searching web…', fetch_page: 'Fetching page…',
    read_file: 'Reading file…', write_file: 'Writing file…',
    list_directory: 'Listing directory…', execute_code: 'Running code…',
    install_package: 'Installing package…',
    get_system_info: 'Getting system info…', analyze_file: 'Analyzing file…',
    patch_file: 'Patching file…', list_outputs: 'Listing outputs…',
    log_research: 'Logging research…', recall_memory: 'Recalling memory…',
    log_fact: 'Storing fact…', scaffold_project: 'Scaffolding project…',
    get_project_status: 'Checking project status…',
    mark_file_complete: 'Marking file complete…',
    run_project_test: 'Running project test…',
    browser_open: 'Opening browser…', browser_read: 'Reading page…',
    deep_research: 'Planning research…',
  };
  return labels[name] || `Using ${name}…`;
}

function paramsPreview(input) {
  if (!input) return '';
  const vals = Object.values(input);
  if (!vals.length) return '';
  const first = String(vals[0]);
  const preview = first.length > 55 ? first.slice(0, 52) + '…' : first;
  return `(${preview})`;
}

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
 * Append a tool block.
 * When inside a task run, append to the container body.
 * When standalone (simple Q&A), append directly to chatArea.
 */
function appendToolBlock(toolName, input, toolUseId) {
  const cat    = toolCategory(toolName);
  const icon   = toolIcon(toolName);
  const params = paramsPreview(input);
  const start  = Date.now();

  taskRunStepCount++;
  updateTaskContainerMeta();

  const el = document.createElement('details');
  el.className = 'tool-block';
  el.dataset.cat     = cat;
  el.dataset.tool    = toolName;
  el.dataset.startMs = start;
  if (toolUseId) el.dataset.toolUseId = toolUseId;

  el.innerHTML = `
    <summary>
      <span class="tb-arrow">▶</span>
      <span class="tb-cat-dot"></span>
      <span class="tb-name">${escapeHtml(toolName)}</span>
      <span class="tb-params">${escapeHtml(params)}</span>
      <span class="tb-status">⟳</span>
      <span class="tb-time"></span>
    </summary>
    <div class="tb-body">
      <div class="tb-section-label">INPUT</div>
      ${renderToolInput(toolName, input)}
      <div class="tb-result-area" style="margin-top:6px"></div>
    </div>
  `;

  const target = _currentChatTarget();
  target.appendChild(el);

  if (toolUseId) pendingToolBlocks[toolUseId] = el;
  pendingToolBlocks[toolName] = el;  // by-name fallback

  scrollToBottom();
  return el;
}

function updateToolBlock(idOrName, success, result) {
  const el = pendingToolBlocks[idOrName];
  if (!el) return;

  const elapsed   = Date.now() - parseInt(el.dataset.startMs || '0', 10);
  const statusEl  = el.querySelector('.tb-status');
  const timeEl    = el.querySelector('.tb-time');
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
    delete displayResult.tree;
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
  const target = _currentChatTarget();
  const el = document.createElement('div');
  el.className = 'tool-denied-msg';
  el.textContent = `✗ ${toolName}: denied by user`;
  target.appendChild(el);
  scrollToBottom();
}

// Phase 4.5 — execution output streaming
function appendExecutionOutputLine(line, stream) {
  const block = pendingToolBlocks['execute_code'];
  if (!block) return;

  const body = block.querySelector('.tb-body');
  if (!body) return;

  let streamArea = body.querySelector('.tb-stream-area');
  if (!streamArea) {
    const label = document.createElement('div');
    label.className = 'tb-section-label';
    label.textContent = 'LIVE OUTPUT';
    body.appendChild(label);

    streamArea = document.createElement('pre');
    streamArea.className = 'tb-stream-area';
    body.appendChild(streamArea);
  }

  streamArea.textContent += line + '\n';
  streamArea.scrollTop = streamArea.scrollHeight;
  block.open = true;
  scrollToBottom();
}

// ============================================================
// Task lifecycle
// ============================================================

function handleTaskStarted(data) {
  isTaskRunning = true;
  btnStop.classList.remove('hidden');
  btnStop.disabled = false;

  // Create the task-run container in the chat area
  if (!activeTaskContainer) {
    createTaskContainer(currentTaskGoal);
  }

  // Reset right-panel timeline
  resetTaskPanel();
}

function handleTaskProgress(data) {
  const { step, label, status, elapsed_ms } = data;
  const panel = document.getElementById('task-progress-panel');
  const placeholder = document.getElementById('task-placeholder');
  if (placeholder) placeholder.remove();

  const stepKey = `step-${step}`;
  let stepEl = activeTaskSteps[stepKey];

  if (!stepEl) {
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

  stepEl.className = `task-step ${status}`;

  const timeEl = stepEl.querySelector('.step-time');
  if (timeEl && elapsed_ms != null) {
    timeEl.textContent = elapsed_ms < 1000
      ? `${elapsed_ms}ms`
      : `${(elapsed_ms / 1000).toFixed(1)}s`;
  }

  // Phase 3h: research progress labels
  if (label && /^Researching:\s*\d+\/\d+/i.test(label)) {
    const match = label.match(/(\d+)\/(\d+)/);
    if (match) {
      _researchCurrent = parseInt(match[1]);
      _researchTotal   = parseInt(match[2]);
      _updateResearchBar(_researchCurrent, _researchTotal);
    }
  }

  // Switch to tasks tab if hidden
  const tasksTab = document.getElementById('tab-tasks');
  if (tasksTab && tasksTab.classList.contains('hidden')) switchTab('tasks');
}

function handleTaskStopped(data) {
  _removeResearchBar();

  Object.values(activeTaskSteps).forEach(el => {
    if (el.classList.contains('running')) el.className = 'task-step cancelled';
  });

  const reason = data && data.reason;

  if (reason === 'user_cancelled') {
    // Append cancellation note inside the task container
    const target = activeTaskContainer
      ? activeTaskContainer.querySelector('.task-run-body')
      : chatArea;
    const row = document.createElement('div');
    row.className = 'msg-row agent-row';
    row.innerHTML = `
      <span class="msg-role" style="opacity:0.45">sys</span>
      <div class="msg-bubble" style="opacity:0.6;font-style:italic;">↪ Task cancelled.</div>
    `;
    target.appendChild(row);
    scrollToBottom();
    closeTaskContainer('cancelled');

  } else if (reason === 'error') {
    const errText = (data && data.error) ? data.error : 'Unknown error.';
    const target = activeTaskContainer
      ? activeTaskContainer.querySelector('.task-run-body')
      : chatArea;
    const row = document.createElement('div');
    row.className = 'msg-row agent-row';
    row.innerHTML = `
      <span class="msg-role" style="color:var(--red)">sys</span>
      <div class="msg-bubble" style="border-color:var(--red);color:var(--red);">↪ Task failed: ${escapeHtml(errText)}</div>
    `;
    target.appendChild(row);
    scrollToBottom();
    closeTaskContainer('failed');

  } else {
    // 'complete' — message event already handled by appendAgentMessage + closeTaskContainer
    closeTaskContainer('complete');
  }

  hideThinkingIndicator();
  setWaiting(false);
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

  const header = document.createElement('div');
  header.className = 'plan-card-header';
  header.innerHTML = `
    <span class="plan-card-icon">📋</span>
    <span class="plan-card-title">Proposed Plan</span>
    <span class="plan-card-count">${steps.length} step${steps.length !== 1 ? 's' : ''}</span>
  `;
  card.appendChild(header);

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

  const actions = document.createElement('div');
  actions.className = 'plan-card-actions';
  const btnRun    = document.createElement('button');
  const btnCancel = document.createElement('button');
  btnRun.className    = 'plan-btn plan-btn-run';
  btnRun.innerHTML    = '▶ Run Plan';
  btnCancel.className = 'plan-btn plan-btn-cancel';
  btnCancel.innerHTML = '✕ Cancel';
  actions.appendChild(btnCancel);
  actions.appendChild(btnRun);
  card.appendChild(actions);

  btnRun.addEventListener('click', () => {
    btnRun.disabled = true; btnCancel.disabled = true;
    const editedSteps = steps.map((s) => {
      const labelEl = card.querySelector(`[data-step="${s.step}"][data-field="action"]`);
      return { ...s, action: labelEl ? labelEl.textContent.trim() : s.action };
    });
    const wasEdited = editedSteps.some((s, i) => s.action !== steps[i].action);
    sendWS({ type: 'plan_response', plan_id, approved: true, edited_steps: wasEdited ? editedSteps : null });
    actions.innerHTML = '<span class="plan-submitted-note">Plan approved ✓ — running…</span>';
    card.querySelectorAll('[contenteditable]').forEach(el => { el.contentEditable = 'false'; });
  });

  btnCancel.addEventListener('click', () => {
    btnRun.disabled = true; btnCancel.disabled = true;
    sendWS({ type: 'plan_response', plan_id, approved: false, edited_steps: null });
    actions.innerHTML = '<span class="plan-submitted-note plan-cancelled-note">Plan cancelled</span>';
    card.querySelectorAll('[contenteditable]').forEach(el => { el.contentEditable = 'false'; });
  });

  // Plan card goes into current chat target
  const target = _currentChatTarget();
  target.appendChild(card);
  scrollToBottom();
}

// ============================================================
// Task running state
// ============================================================

function startTaskRunning() {
  isTaskRunning = true;
  btnStop.classList.remove('hidden');
  btnStop.disabled = false;
  resetTaskPanel();
  pendingToolBlocks = {};
}

function stopTaskRunning() {
  isTaskRunning = false;
  btnStop.classList.add('hidden');
  btnStop.disabled = false;
}

function resetTaskPanel() {
  activeTaskSteps = {};
  const panel = document.getElementById('task-progress-panel');
  if (panel) {
    panel.innerHTML = '<div id="task-placeholder" class="tab-placeholder">No active task.</div>';
  }
}

// Stop button
btnStop.addEventListener('click', () => {
  btnStop.disabled = true;
  sendWS({ type: 'stop_task' });
});

// ============================================================
// Phase 3h — Research progress bar (Tasks tab)
// ============================================================

function handleResearchStarted(data) {
  const total = data && data.total_questions ? parseInt(data.total_questions) : 0;
  if (total > 0) {
    _researchTotal = total;
    _researchCurrent = 0;
    _ensureResearchBar(total);
  }
}

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
    panel.insertBefore(bar, panel.firstChild);
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
  _researchTotal = 0;
  _researchCurrent = 0;
}

// ============================================================
// Task history — Tasks tab
// Phase 4.5: load from GET /task?history=10 on connect
// ============================================================

async function loadTaskHistory() {
  try {
    const res  = await fetch('/task?history=10');
    const data = await res.json();

    // Check for interrupted (running) task
    if (data && data.status === 'running') {
      showInterruptedBanner(data);
    } else if (data && (data.status === 'complete' || data.status === 'cancelled')) {
      // Restore last task step timeline in right panel
      restoreLastTaskTimeline(data);
    }

    // Render task history list
    const history = (data && data.history) ? data.history : [];
    renderTaskHistoryList(history);

  } catch (e) {
    console.warn('[task] Could not load task history:', e);
    const phEl = document.getElementById('task-history-placeholder');
    if (phEl) phEl.textContent = 'Could not load history.';
  }
}

/**
 * Restore the last task's step timeline in the right panel.
 * Called on page load when the last task was completed or cancelled.
 */
function restoreLastTaskTimeline(task) {
  const panel = document.getElementById('task-progress-panel');
  const placeholder = document.getElementById('task-placeholder');
  if (placeholder) placeholder.remove();

  const steps = task.steps || [];
  if (!steps.length) {
    const panel = document.getElementById('task-progress-panel');
    if (!panel) return;
    const el = document.createElement('div');
    el.className = 'task-step done';
    el.innerHTML = `<span class="step-dot done"></span>
        <span class="step-label">${escapeHtml((task.initial_message || 'Unknown task').slice(0, 80))}</span>
        <span class="step-status" style="margin-left:auto;font-size:0.75em;opacity:0.6">✓ ${task.status || 'complete'}</span>`;
    panel.appendChild(el);
    return;
  }

  steps.forEach((s) => {
    const stepKey = `step-${s.step}`;
    if (activeTaskSteps[stepKey]) return;

    const stepEl = document.createElement('div');
    stepEl.className = `task-step ${s.status || 'done'}`;
    stepEl.id = stepKey;
    const elapsed = s.elapsed_ms != null
      ? (s.elapsed_ms < 1000 ? `${s.elapsed_ms}ms` : `${(s.elapsed_ms/1000).toFixed(1)}s`)
      : '';
    stepEl.innerHTML = `
      <div class="step-dot"></div>
      <div class="step-body">
        <div><span class="step-num">${s.step}.</span><span class="step-label">${escapeHtml(s.tool || '')}</span></div>
        ${elapsed ? `<div class="step-time">${elapsed}</div>` : ''}
      </div>
    `;
    panel.appendChild(stepEl);
    activeTaskSteps[stepKey] = stepEl;
  });
}

/**
 * Render the task history list in the Tasks tab.
 * Each item: goal, outcome badge, duration, timestamp. Expandable to show steps.
 */
function renderTaskHistoryList(history) {
  const list = document.getElementById('task-history-list');
  if (!list) return;

  const placeholder = document.getElementById('task-history-placeholder');
  if (placeholder) placeholder.remove();

  if (!history || !history.length) {
    list.innerHTML = '<div class="tab-placeholder">No completed tasks yet.</div>';
    return;
  }

  // Show most recent first
  const items = [...history].reverse();

  items.forEach((task) => {
    const item = document.createElement('div');
    item.className = 'task-history-item';

    const outcome  = task.outcome || 'unknown';
    const goal     = (task.goal || '').slice(0, 50) + (task.goal && task.goal.length > 50 ? '…' : '');
    const durSec   = task.duration_seconds;
    const durStr   = durSec != null
      ? (durSec < 60 ? `${durSec}s` : `${Math.floor(durSec/60)}m ${durSec%60}s`)
      : '';
    const ts       = task.timestamp
      ? new Date(task.timestamp).toLocaleDateString(undefined, { month:'short', day:'numeric' })
      : '';

    const header = document.createElement('div');
    header.className = 'task-history-header';
    header.innerHTML = `
      <span class="th-outcome ${outcome}">${outcome}</span>
      <span class="th-goal">${escapeHtml(goal)}</span>
      <span class="th-meta">
        ${durStr ? `<span>${durStr}</span>` : ''}
        ${ts     ? `<span>${ts}</span>`     : ''}
      </span>
    `;
    item.appendChild(header);

    // Steps section (hidden until expanded)
    const stepsDiv = document.createElement('div');
    stepsDiv.className = 'task-history-steps';

    const toolsUsed = task.tools_used || [];
    const summary   = task.summary   || '';
    if (summary) {
      const p = document.createElement('div');
      p.style.cssText = 'font-size:10.5px;color:var(--text-dim);margin-bottom:5px;';
      p.textContent = summary;
      stepsDiv.appendChild(p);
    }
    toolsUsed.forEach((tool) => {
      const row = document.createElement('div');
      row.className = 'th-step-row';
      row.innerHTML = `
        <span class="th-step-dot"></span>
        <span>${escapeHtml(tool)}</span>
      `;
      stepsDiv.appendChild(row);
    });

    item.appendChild(stepsDiv);

    // Toggle on header click
    header.addEventListener('click', () => {
      item.classList.toggle('expanded');
    });

    list.appendChild(item);
  });
}

// ============================================================
// Right panel — collapse / expand / tabs
// ============================================================

window.toggleRightPanel = function() {
  const panel   = document.getElementById('right-panel');
  const btn     = document.getElementById('panel-toggle-btn');
  const appBody = document.getElementById('app-body');
  const collapsed = panel.classList.toggle('collapsed');
  btn.textContent = collapsed ? '▶' : '◀';
  appBody.style.gridTemplateColumns = collapsed
    ? `1fr ${getComputedStyle(document.documentElement).getPropertyValue('--panel-toggle-w').trim()} 0px`
    : '';
};

window.switchTab = function(tabName) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  const tab = document.getElementById(`tab-${tabName}`);
  if (tab) tab.classList.remove('hidden');
  const btn = document.querySelector(`.tab-btn[data-tab="${tabName}"]`);
  if (btn) btn.classList.add('active');
};

window.expandAndTab = function(tabName) {
  const panel = document.getElementById('right-panel');
  if (panel.classList.contains('collapsed')) toggleRightPanel();
  switchTab(tabName);
};

// ============================================================
// Memory tab
// ============================================================

const recentHistory = [];

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
}

// ============================================================
// Optimizer indicator (chat area spinner)
// ============================================================

function showOptimizerIndicator() {
  if (optimizerEnabled && optimizerIndicator) optimizerIndicator.classList.remove('hidden');
}
function hideOptimizerIndicator() {
  if (optimizerIndicator) optimizerIndicator.classList.add('hidden');
}

function appendOptimizerPill(original, optimized) {
  const pill = document.createElement('div');
  pill.className = 'optimizer-pill';
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
  if (toggleOptimizer) toggleOptimizer.classList.toggle('on', enabled);
  if (toggleOptimizerLabel) toggleOptimizerLabel.textContent = enabled ? 'on' : 'off';
}

if (toggleOptimizer) {
  toggleOptimizer.addEventListener('click', () => {
    const next = !optimizerEnabled;
    sendWS({ type: 'set_optimizer', data: { enabled: next } });
    setOptimizerState(next);
  });
}

// ============================================================
// Local mode state
// ============================================================

function applyLocalMode(enabled) {
  localModeEnabled = enabled;
  if (btnModeOnline) btnModeOnline.classList.toggle('active-online', !enabled);
  if (btnModeLocal)  btnModeLocal.classList.toggle('active-local',   enabled);
  if (selPrimaryModel) selPrimaryModel.disabled = enabled;
  const primaryRow = document.getElementById('row-primary-model');
  if (primaryRow) primaryRow.style.opacity = enabled ? '0.4' : '1';
  updateModelDisplay();
}

function updateModelDisplay() {
  const modelStr = localModeEnabled
    ? `local: ${currentLocalModel}`
    : `${currentPrimaryModel}`;

  if (modeModelLabel) modeModelLabel.textContent = localModeEnabled ? currentLocalModel : currentPrimaryModel;
  if (activeModelDisplay) activeModelDisplay.textContent = modelStr;
}

if (btnModeOnline) {
  btnModeOnline.addEventListener('click', () => {
    if (localModeEnabled) {
      sendWS({ type: 'set_local_mode', data: { enabled: false } });
      applyLocalMode(false);
    }
  });
}
if (btnModeLocal) {
  btnModeLocal.addEventListener('click', () => {
    if (!localModeEnabled) {
      sendWS({ type: 'set_local_mode', data: { enabled: true } });
      applyLocalMode(true);
    }
  });
}

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

if (btnSettingsOpen)  btnSettingsOpen.addEventListener('click',  openSettings);
if (btnSettingsClose) btnSettingsClose.addEventListener('click', closeSettings);
if (settingsBackdrop) settingsBackdrop.addEventListener('click', closeSettings);

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && settingsPanel.classList.contains('open')) closeSettings();
});

window.toggleAdvanced = function() {
  advancedHeader.classList.toggle('open');
  advancedBody.classList.toggle('open');
};

// ============================================================
// Settings — model dropdowns
// ============================================================

if (selPrimaryModel) {
  selPrimaryModel.addEventListener('change', () => {
    const model = selPrimaryModel.value;
    currentPrimaryModel = model;
    sendWS({ type: 'set_model', data: { model } });
    if (!localModeEnabled) updateModelDisplay();
  });
}

if (selLocalAgentModel) {
  selLocalAgentModel.addEventListener('change', () => {
    const model = selLocalAgentModel.value;
    currentLocalModel = model;
    sendWS({ type: 'set_local_agent_model', data: { model } });
    if (localModeEnabled) updateModelDisplay();
  });
}

// ============================================================
// Settings — embeddings toggle
// ============================================================

if (toggleEmbeddings) {
  toggleEmbeddings.addEventListener('click', () => {
    const wasOn = toggleEmbeddings.classList.contains('on');
    const next  = !wasOn;
    toggleEmbeddings.classList.toggle('on', next);
    if (toggleEmbeddingsLbl) toggleEmbeddingsLbl.textContent = next ? 'on' : 'off';
    sendWS({ type: 'set_config', data: { key: 'embeddings.enabled', value: next } });
  });
}

// ============================================================
// Settings — number inputs (debounced)
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

if (inpSimilarity) {
  inpSimilarity.addEventListener('input', () => {
    const v = parseFloat(inpSimilarity.value);
    if (valSimilarity) valSimilarity.textContent = v.toFixed(2);
    clearTimeout(_debounceTimers['similarity']);
    _debounceTimers['similarity'] = setTimeout(() => {
      sendWS({ type: 'set_config', data: { key: 'context.similarity_cutoff', value: v } });
    }, 300);
  });
}

function sendTreeRoot() {
  if (!inpTreeRoot) return;
  const v = inpTreeRoot.value.trim();
  if (v) sendWS({ type: 'set_config', data: { key: 'tree_root', value: v } });
}
if (inpTreeRoot) {
  inpTreeRoot.addEventListener('blur', sendTreeRoot);
  inpTreeRoot.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); sendTreeRoot(); } });
}

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

function flashAckForKey(key) { flashAck(keyToAckId[key] || null); }

// ============================================================
// Populate settings from /status
// ============================================================

function populateSettingsFromStatus(data) {
  if (data.primary_model) {
    currentPrimaryModel = data.primary_model;
    if (selPrimaryModel) {
      const opt = [...selPrimaryModel.options].find(o => o.value === data.primary_model);
      if (opt) opt.selected = true;
    }
  }
  if (data.local_agent_model) {
    currentLocalModel = data.local_agent_model;
    if (selLocalAgentModel) {
      const opt = [...selLocalAgentModel.options].find(o => o.value === data.local_agent_model);
      if (opt) opt.selected = true;
    }
  }

  setOptimizerState(!!data.use_prompt_optimizer);
  applyLocalMode(!!data.local_mode);
  updateModelDisplay();

  const ctx = data.context || {};
  if (inpRecentTurns      && ctx.recent_turns      != null) inpRecentTurns.value      = ctx.recent_turns;
  if (inpSummaryThreshold && ctx.summary_threshold  != null) inpSummaryThreshold.value = ctx.summary_threshold;
  if (inpRetrievalN       && ctx.retrieval_n        != null) inpRetrievalN.value       = ctx.retrieval_n;
  if (inpSimilarity       && ctx.similarity_cutoff  != null) {
    inpSimilarity.value = ctx.similarity_cutoff;
    if (valSimilarity) valSimilarity.textContent = Number(ctx.similarity_cutoff).toFixed(2);
  }
  if (inpMaxHistory    && ctx.max_history_turns          != null) inpMaxHistory.value    = ctx.max_history_turns;
  if (inpMaxIterations && ctx.max_iterations_per_turn    != null) inpMaxIterations.value = ctx.max_iterations_per_turn;
  if (inpLocalTimeout  && data.local_agent_timeout       != null) inpLocalTimeout.value  = data.local_agent_timeout;

  const emb = data.embeddings || {};
  if (emb.enabled != null && toggleEmbeddings) {
    toggleEmbeddings.classList.toggle('on', !!emb.enabled);
    if (toggleEmbeddingsLbl) toggleEmbeddingsLbl.textContent = emb.enabled ? 'on' : 'off';
  }
  if (inpTreeRoot && data.tree_root != null) inpTreeRoot.value = data.tree_root;

  if (data.embeddings_count != null) {
    const el = document.getElementById('memory-vector-status');
    if (el) el.textContent = `Semantic memory: ${data.embeddings_count} entries`;
  }

  updateProfileStatus(data.profile_loaded);

  // Update legacy status-pill state (kept for /status polling logic)
  if (statusClaude && statusClaude.classList) {
    statusClaude.classList.toggle('online',  !!data.claude_api);
    statusClaude.classList.toggle('offline', !data.claude_api);
  }
  if (statusOllama && statusOllama.classList) {
    statusOllama.classList.toggle('online',  !!data.ollama);
    statusOllama.classList.toggle('offline', !data.ollama);
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
// Waiting state
// ============================================================

function setWaiting(waiting) {
  isWaiting = waiting;
  btnSend.disabled   = waiting;
  userInput.disabled = waiting;
}

// ============================================================
// Thinking indicator
// ============================================================

function showThinkingIndicator(label) {
  if (thinkingLabel) thinkingLabel.textContent = label || 'Thinking…';
  if (thinkingIndicator) thinkingIndicator.classList.remove('hidden');
  scrollToBottom();
}
function hideThinkingIndicator() {
  if (thinkingIndicator) thinkingIndicator.classList.add('hidden');
}
function setThinkingLabel(label) {
  if (thinkingLabel) thinkingLabel.textContent = label;
  showThinkingIndicator(label);
}

// ============================================================
// Clear conversation
// ============================================================

function clearChat() {
  chatArea.querySelectorAll(
    '.msg-row, .tool-block, .tool-denied-msg, .optimizer-pill, .task-run-container, .plan-card'
  ).forEach(el => el.remove());
  pendingToolBlocks  = {};
  activeTaskSteps    = {};
  activeTaskContainer = null;
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

if (btnClear) {
  btnClear.addEventListener('click', () => {
    if (confirm('Clear the conversation history?')) sendWS({ type: 'clear' });
  });
}

// ============================================================
// Long-term memory counts
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
// Status polling — /status
// ============================================================

function updateProfileStatus(loaded) {
  const el = document.getElementById('mem-profile-status');
  if (!el) return;
  el.textContent = loaded ? 'Profile: loaded ✓' : 'Profile: not found ⚠';
  el.style.color = loaded ? 'var(--accent-green)' : 'var(--accent-amber)';
}

async function pollStatus() {
  try {
    const res  = await fetch('/status');
    const data = await res.json();
    populateSettingsFromStatus(data);
  } catch (e) {
    console.warn('[status] Could not fetch /status:', e);
  }
}

// ============================================================
// Interrupted banner  (Phase 3b)
// ============================================================

function showInterruptedBanner(task) {
  const existing = document.getElementById('interrupted-banner');
  if (existing) return;

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

  if (chatArea.firstChild) {
    chatArea.insertBefore(banner, chatArea.firstChild);
  } else {
    chatArea.appendChild(banner);
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
  el.style.height = Math.min(el.scrollHeight, 130) + 'px';
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

function renderMarkdownWithCode(text) {
  const parts = text.split(/(```[\s\S]*?```)/g);
  return parts.map(part => {
    if (part.startsWith('```')) {
      const lines = part.slice(3, -3).split('\n');
      const code  = lines.slice(1).join('\n');
      return `<div class="code-block"><pre>${escapeHtml(code)}</pre></div>`;
    }
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

  fetchMemoryCounts();
  setInterval(fetchMemoryCounts, 60_000);

  // Load task history (and check for interrupted task)
  loadTaskHistory();

  // Standalone profile status update — fires once shortly after load to
  // ensure the indicator is correct even if pollStatus races with the server.
  setTimeout(async () => {
    try {
      const r = await fetch('/status');
      const d = await r.json();
      updateProfileStatus(d.profile_loaded);
    } catch(e) {}
  }, 500);
}

init();
