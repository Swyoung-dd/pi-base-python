"use strict";

const state = {
  sessions: [],
  selectedSessionId: null,
  messages: [],
  models: [],
  selectedModel: null,
  thinking: "off",
  streaming: false,
  liveAssistantIndex: null,
  liveTools: [],
  railTab: "sessions",
  directoryPath: "",
  directoryParent: null,
  selectedFile: null,
  selectedFileContent: null,
};

const elements = {
  workspace: document.querySelector(".workspace"),
  projectPath: document.querySelector("#projectPath"),
  emptyPath: document.querySelector("#emptyPath"),
  runStatus: document.querySelector("#runStatus"),
  statusCopy: document.querySelector("#runStatus .status-copy"),
  themeButton: document.querySelector("#themeButton"),
  inspectorButton: document.querySelector("#inspectorButton"),
  closeInspectorButton: document.querySelector("#closeInspectorButton"),
  mobileMenuButton: document.querySelector("#mobileMenuButton"),
  scrim: document.querySelector("#scrim"),
  sessionsPanel: document.querySelector("#sessionsPanel"),
  filesPanel: document.querySelector("#filesPanel"),
  sessionSearch: document.querySelector("#sessionSearch"),
  sessionList: document.querySelector("#sessionList"),
  newSessionButton: document.querySelector("#newSessionButton"),
  refreshSessionsButton: document.querySelector("#refreshSessionsButton"),
  fileList: document.querySelector("#fileList"),
  fileUpButton: document.querySelector("#fileUpButton"),
  filePathButton: document.querySelector("#filePathButton"),
  refreshFilesButton: document.querySelector("#refreshFilesButton"),
  sessionTitle: document.querySelector("#sessionTitle"),
  sessionMeta: document.querySelector("#sessionMeta"),
  modelSelect: document.querySelector("#modelSelect"),
  thinkingSelect: document.querySelector("#thinkingSelect"),
  messageViewport: document.querySelector("#messageViewport"),
  emptyState: document.querySelector("#emptyState"),
  messageList: document.querySelector("#messageList"),
  promptInput: document.querySelector("#promptInput"),
  sendButton: document.querySelector("#sendButton"),
  composerState: document.querySelector("#composerState"),
  characterCount: document.querySelector("#characterCount"),
  inspectorTitle: document.querySelector("#inspectorTitle"),
  signalInspector: document.querySelector("#signalInspector"),
  fileInspector: document.querySelector("#fileInspector"),
  signalChannel: document.querySelector("#signalChannel"),
  signalModel: document.querySelector("#signalModel"),
  signalThinking: document.querySelector("#signalThinking"),
  signalMessages: document.querySelector("#signalMessages"),
  signalState: document.querySelector("#signalState"),
  scopeLabel: document.querySelector("#scopeLabel"),
  activityList: document.querySelector("#activityList"),
  fileInspectorPath: document.querySelector("#fileInspectorPath"),
  filePreview: document.querySelector("#filePreview"),
  copyFileButton: document.querySelector("#copyFileButton"),
  toastRegion: document.querySelector("#toastRegion"),
};

function icon(name) {
  return `<svg aria-hidden="true"><use href="#i-${name}"></use></svg>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function inlineMarkdown(value) {
  let text = escapeHtml(value);
  text = text.replace(/`([^`]+)`/g, "<code>$1</code>");
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  text = text.replace(/\[([^\]]+)]\((https?:\/\/[^\s)]+)\)/g, (_match, label, url) => {
    return `<a href="${url}" target="_blank" rel="noreferrer">${label}</a>`;
  });
  return text;
}

function markdownToHtml(value) {
  const source = String(value || "").replaceAll("\r\n", "\n");
  const parts = source.split(/(```[\s\S]*?```)/g);
  return parts.map((part) => {
    if (part.startsWith("```")) {
      const body = part.slice(3, -3);
      const newline = body.indexOf("\n");
      const code = newline >= 0 ? body.slice(newline + 1) : body;
      return `<pre><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`;
    }
    const lines = part.split("\n");
    const output = [];
    let paragraph = [];
    let listType = null;
    let listItems = [];

    const flushParagraph = () => {
      if (paragraph.length) {
        output.push(`<p>${inlineMarkdown(paragraph.join(" "))}</p>`);
        paragraph = [];
      }
    };
    const flushList = () => {
      if (listType && listItems.length) {
        output.push(`<${listType}>${listItems.map((item) => `<li>${inlineMarkdown(item)}</li>`).join("")}</${listType}>`);
      }
      listType = null;
      listItems = [];
    };

    for (const line of lines) {
      const heading = line.match(/^(#{1,3})\s+(.+)$/);
      const unordered = line.match(/^\s*[-*]\s+(.+)$/);
      const ordered = line.match(/^\s*\d+\.\s+(.+)$/);
      const quote = line.match(/^>\s?(.*)$/);
      if (!line.trim()) {
        flushParagraph();
        flushList();
      } else if (heading) {
        flushParagraph();
        flushList();
        output.push(`<h${heading[1].length}>${inlineMarkdown(heading[2])}</h${heading[1].length}>`);
      } else if (unordered || ordered) {
        flushParagraph();
        const nextType = unordered ? "ul" : "ol";
        if (listType && listType !== nextType) flushList();
        listType = nextType;
        listItems.push((unordered || ordered)[1]);
      } else if (quote) {
        flushParagraph();
        flushList();
        output.push(`<blockquote>${inlineMarkdown(quote[1])}</blockquote>`);
      } else {
        flushList();
        paragraph.push(line.trim());
      }
    }
    flushParagraph();
    flushList();
    return output.join("");
  }).join("");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: options.body ? { "Content-Type": "application/json", ...(options.headers || {}) } : options.headers,
  });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    throw new Error(payload?.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  elements.toastRegion.append(toast);
  window.setTimeout(() => toast.remove(), 3600);
}

function setRunState(next, label) {
  state.streaming = next === "running";
  elements.runStatus.dataset.state = next;
  elements.statusCopy.textContent = label;
  elements.signalState.textContent = label;
  elements.scopeLabel.textContent = next === "running" ? "LIVE SIGNAL" : next === "error" ? "FAULT SIGNAL" : "IDLE SIGNAL";
  elements.signalInspector.classList.toggle("is-running", state.streaming);
  elements.sendButton.classList.toggle("is-running", state.streaming);
  elements.sendButton.setAttribute("aria-label", state.streaming ? "停止生成" : "发送消息");
  elements.composerState.textContent = state.streaming ? "INPUT · AGENT RUNNING" : "INPUT · READY";
  elements.modelSelect.disabled = state.streaming;
  elements.thinkingSelect.disabled = state.streaming;
}

function addActivity(message) {
  const entry = document.createElement("div");
  entry.className = "activity-entry";
  const time = document.createElement("time");
  time.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const copy = document.createElement("span");
  copy.textContent = message;
  entry.append(time, copy);
  elements.activityList.prepend(entry);
  while (elements.activityList.children.length > 9) {
    elements.activityList.lastElementChild.remove();
  }
}

function relativeTime(value) {
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 1000));
  if (seconds < 60) return "NOW";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}M`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}H`;
  return `${Math.floor(hours / 24)}D`;
}

function sessionDisplayTitle(session) {
  return session?.preview?.trim() || (session?.id ? `Channel ${session.id.slice(0, 8)}` : "新会话");
}

function renderSessions() {
  const query = elements.sessionSearch.value.trim().toLowerCase();
  const sessions = state.sessions.filter((session) => {
    return !query || sessionDisplayTitle(session).toLowerCase().includes(query) || session.id.toLowerCase().includes(query);
  });
  elements.sessionList.replaceChildren();
  if (!sessions.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = query ? "NO MATCHING CHANNELS" : "NO SAVED CHANNELS";
    elements.sessionList.append(empty);
    return;
  }
  for (const session of sessions) {
    const row = document.createElement("div");
    row.className = "session-row";
    row.classList.toggle("is-active", session.id === state.selectedSessionId);

    const open = document.createElement("button");
    open.className = "session-open";
    open.type = "button";
    open.setAttribute("aria-label", `打开会话：${sessionDisplayTitle(session)}`);
    open.innerHTML = `<span class="session-node"></span><span class="session-copy"><span class="session-title">${escapeHtml(sessionDisplayTitle(session))}</span><span class="session-meta"><span>${relativeTime(session.updatedAt)}</span><span>${session.messageCount} MSG</span></span></span>`;
    open.addEventListener("click", () => selectSession(session.id));

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "bare-icon session-delete";
    remove.setAttribute("aria-label", "删除会话");
    remove.innerHTML = icon("trash");
    remove.addEventListener("click", () => deleteSession(session));
    row.append(open, remove);
    elements.sessionList.append(row);
  }
}

async function loadSessions() {
  elements.sessionList.innerHTML = '<div class="loading-list">SCANNING CHANNELS</div>';
  const payload = await api("/api/sessions");
  state.sessions = payload.sessions;
  renderSessions();
}

function messageText(message) {
  if (typeof message.content === "string") return message.content;
  return (message.content || [])
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("\n");
}

function renderContentBlocks(message, container, isStreamingMessage) {
  for (const block of message.content || []) {
    if (block.type === "text") {
      const markdown = document.createElement("div");
      markdown.className = `markdown${isStreamingMessage ? " typing-cursor" : ""}`;
      markdown.innerHTML = markdownToHtml(block.text || "");
      container.append(markdown);
    } else if (block.type === "thinking") {
      const details = document.createElement("details");
      details.className = "thinking-block";
      const summary = document.createElement("summary");
      summary.textContent = isStreamingMessage ? "THINKING · LIVE" : "THINKING TRACE";
      const content = document.createElement("div");
      content.className = "thinking-content";
      content.textContent = block.thinking || "";
      details.append(summary, content);
      container.append(details);
    } else if (block.type === "toolCall") {
      container.append(createToolBlock(block.name, block.arguments, "done"));
    }
  }
  if (message.error_message) {
    const error = document.createElement("div");
    error.className = "message-error";
    error.textContent = message.error_message;
    container.append(error);
  }
}

function createToolBlock(name, detail, toolState) {
  const block = document.createElement("div");
  block.className = "tool-block";
  block.dataset.state = toolState;
  const heading = document.createElement("div");
  heading.className = "tool-heading";
  const status = document.createElement("span");
  status.className = "tool-status";
  const label = document.createElement("span");
  label.textContent = `${name || "tool"} · ${toolState === "running" ? "RUNNING" : toolState === "error" ? "FAILED" : "COMPLETE"}`;
  heading.append(status, label);
  const content = document.createElement("div");
  content.className = "tool-content";
  content.textContent = typeof detail === "string" ? detail : JSON.stringify(detail || {}, null, 2);
  block.append(heading, content);
  return block;
}

function renderMessages() {
  elements.messageList.replaceChildren();
  const hasMessages = state.messages.length > 0 || state.liveTools.length > 0;
  elements.emptyState.hidden = hasMessages;
  elements.messageList.hidden = !hasMessages;
  state.messages.forEach((message, index) => {
    if (message.role === "toolResult") return;
    const article = document.createElement("article");
    article.className = `message ${message.role}`;
    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = message.role === "user" ? "YOU" : "piY";
    const body = document.createElement("div");
    body.className = "message-body";
    if (message.role === "user") {
      body.textContent = messageText(message);
    } else {
      renderContentBlocks(message, body, state.streaming && index === state.liveAssistantIndex);
    }
    article.append(label, body);
    elements.messageList.append(article);
  });
  for (const tool of state.liveTools) {
    const article = document.createElement("article");
    article.className = "message assistant";
    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = "TOOL";
    const body = document.createElement("div");
    body.className = "message-body";
    body.append(createToolBlock(tool.name, tool.detail, tool.state));
    article.append(label, body);
    elements.messageList.append(article);
  }
  elements.signalMessages.textContent = String(state.messages.length);
  elements.sessionMeta.textContent = `LOCAL CHANNEL · ${state.messages.length} MESSAGES`;
}

function updateSessionIdentity() {
  const session = state.sessions.find((item) => item.id === state.selectedSessionId);
  elements.sessionTitle.textContent = sessionDisplayTitle(session);
  elements.signalChannel.textContent = state.selectedSessionId ? state.selectedSessionId.slice(0, 10).toUpperCase() : "NEW";
  elements.sessionMeta.textContent = `LOCAL CHANNEL · ${state.messages.length} MESSAGES`;
}

async function selectSession(sessionId, closeMobile = true) {
  if (state.streaming) {
    showToast("当前会话仍在运行，请先停止或等待完成", "error");
    return;
  }
  const payload = await api(`/api/sessions/${encodeURIComponent(sessionId)}`);
  state.selectedSessionId = sessionId;
  state.messages = payload.messages || [];
  state.liveTools = [];
  state.liveAssistantIndex = null;
  if (payload.model) setSelectedModel(payload.model.provider, payload.model.id);
  updateSessionIdentity();
  renderSessions();
  renderMessages();
  showSignalInspector();
  if (closeMobile) closeNavigation();
  requestAnimationFrame(() => {
    elements.messageViewport.scrollTop = elements.messageViewport.scrollHeight;
  });
}

async function newSession() {
  if (state.streaming) {
    showToast("当前会话仍在运行，请先停止或等待完成", "error");
    return;
  }
  const payload = await api("/api/sessions", { method: "POST", body: "{}" });
  state.selectedSessionId = payload.id;
  state.messages = [];
  state.liveTools = [];
  state.liveAssistantIndex = null;
  elements.sessionTitle.textContent = "新会话";
  elements.sessionMeta.textContent = "LOCAL CHANNEL · 0 MESSAGES";
  elements.signalChannel.textContent = payload.id.slice(0, 10).toUpperCase();
  renderSessions();
  renderMessages();
  showSignalInspector();
  closeNavigation();
  elements.promptInput.focus();
  addActivity("New channel allocated");
}

async function deleteSession(session) {
  if (state.streaming) {
    showToast("运行期间不能删除会话", "error");
    return;
  }
  const confirmed = window.confirm(`删除会话“${sessionDisplayTitle(session)}”？此操作不可撤销。`);
  if (!confirmed) return;
  await api(`/api/sessions/${encodeURIComponent(session.id)}`, { method: "DELETE" });
  addActivity(`Deleted ${session.id.slice(0, 8)}`);
  await loadSessions();
  if (state.selectedSessionId === session.id) {
    if (state.sessions.length) await selectSession(state.sessions[0].id, false);
    else await newSession();
  }
}

function setSelectedModel(provider, id) {
  const value = `${provider}::${id}`;
  const exists = Array.from(elements.modelSelect.options).some((option) => option.value === value);
  if (exists) elements.modelSelect.value = value;
  state.selectedModel = { provider, id };
  elements.signalModel.textContent = `${provider}/${id}`;
}

async function loadModels() {
  const payload = await api("/api/models");
  state.models = payload.models;
  state.thinking = payload.thinking || "off";
  elements.thinkingSelect.value = state.thinking;
  elements.modelSelect.replaceChildren();
  const groups = new Map();
  for (const model of state.models) {
    if (!groups.has(model.provider)) groups.set(model.provider, []);
    groups.get(model.provider).push(model);
  }
  for (const [provider, models] of groups) {
    const group = document.createElement("optgroup");
    group.label = provider;
    for (const model of models) {
      const option = document.createElement("option");
      option.value = `${model.provider}::${model.id}`;
      option.textContent = model.name || model.id;
      group.append(option);
    }
    elements.modelSelect.append(group);
  }
  setSelectedModel(payload.selected.provider, payload.selected.id);
  elements.signalThinking.textContent = state.thinking.toUpperCase();
}

async function changeModel() {
  const [provider, model] = elements.modelSelect.value.split("::");
  await api("/api/model", {
    method: "POST",
    body: JSON.stringify({ provider, model, sessionId: state.selectedSessionId }),
  });
  setSelectedModel(provider, model);
  addActivity(`Model set to ${provider}/${model}`);
}

async function changeThinking() {
  if (!state.selectedSessionId) await newSession();
  const level = elements.thinkingSelect.value;
  await api("/api/thinking", {
    method: "POST",
    body: JSON.stringify({ level, sessionId: state.selectedSessionId }),
  });
  state.thinking = level;
  elements.signalThinking.textContent = level.toUpperCase();
  addActivity(`Reasoning set to ${level}`);
}

function ensureLiveAssistant(message = null) {
  if (state.liveAssistantIndex !== null && state.messages[state.liveAssistantIndex]) {
    return state.messages[state.liveAssistantIndex];
  }
  const assistant = message || {
    role: "assistant",
    content: [{ type: "text", text: "" }],
    error_message: null,
  };
  state.messages.push(assistant);
  state.liveAssistantIndex = state.messages.length - 1;
  return assistant;
}

function appendDelta(type, delta) {
  const assistant = ensureLiveAssistant();
  const blockType = type === "thinking_delta" ? "thinking" : "text";
  const field = blockType === "thinking" ? "thinking" : "text";
  let block = assistant.content?.findLast((item) => item.type === blockType);
  if (!block) {
    block = { type: blockType, [field]: "" };
    assistant.content = [...(assistant.content || []), block];
  }
  block[field] = `${block[field] || ""}${delta}`;
}

function handleAgentEvent(event) {
  if (event.type === "message_start") {
    const assistant = ensureLiveAssistant(event.message);
    if (!assistant.content?.length) assistant.content = [{ type: "text", text: "" }];
  } else if (event.type === "text_delta" || event.type === "thinking_delta") {
    appendDelta(event.type, event.delta || "");
  } else if (event.type === "message_end") {
    ensureLiveAssistant();
    state.messages[state.liveAssistantIndex] = event.message;
  } else if (event.type === "tool_execution_start") {
    state.liveTools.push({
      id: event.tool_call_id,
      name: event.tool_name,
      detail: event.arguments,
      state: "running",
    });
    addActivity(`${event.tool_name} started`);
  } else if (event.type === "tool_execution_end") {
    const tool = state.liveTools.find((item) => item.id === event.tool_call_id);
    if (tool) {
      tool.state = event.result?.is_error ? "error" : "done";
      tool.detail = event.result?.content?.map((block) => block.text || "").join("\n") || tool.detail;
    }
    addActivity(`${event.tool_name} ${event.result?.is_error ? "failed" : "complete"}`);
  } else if (event.type === "provider_retry") {
    addActivity(`Provider retry ${event.attempt}/${event.max_retries}`);
  } else if (event.type === "context_compacted") {
    addActivity(`Context compacted · ${event.dropped_messages} dropped`);
  }
  renderMessages();
  elements.messageViewport.scrollTop = elements.messageViewport.scrollHeight;
}

async function sendPrompt() {
  if (state.streaming) {
    await abortPrompt();
    return;
  }
  const message = elements.promptInput.value.trim();
  if (!message) return;
  if (!state.selectedSessionId) await newSession();
  elements.promptInput.value = "";
  resizePrompt();
  state.messages.push({ role: "user", content: message, timestamp: Date.now() });
  state.liveAssistantIndex = null;
  state.liveTools = [];
  renderMessages();
  setRunState("running", "RUNNING");
  addActivity("Prompt accepted");
  elements.messageViewport.scrollTop = elements.messageViewport.scrollHeight;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId: state.selectedSessionId, message }),
    });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.error || `Request failed: ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const packet = JSON.parse(line);
        if (packet.type === "event") handleAgentEvent(packet.event);
        if (packet.type === "error") throw new Error(packet.error);
      }
      if (done) break;
    }
    if (buffer.trim()) {
      const packet = JSON.parse(buffer);
      if (packet.type === "event") handleAgentEvent(packet.event);
      if (packet.type === "error") throw new Error(packet.error);
    }
    state.liveTools = [];
    state.liveAssistantIndex = null;
    setRunState("ready", "READY");
    await loadSessions();
    await selectSession(state.selectedSessionId, false);
    addActivity("Agent response complete");
  } catch (error) {
    setRunState("error", "FAULT");
    showToast(error.message || String(error), "error");
    addActivity("Agent response failed");
    window.setTimeout(() => setRunState("ready", "READY"), 1800);
  }
}

async function abortPrompt() {
  if (!state.selectedSessionId || !state.streaming) return;
  await api("/api/abort", {
    method: "POST",
    body: JSON.stringify({ sessionId: state.selectedSessionId }),
  });
  addActivity("Abort requested");
}

function resizePrompt() {
  elements.promptInput.style.height = "auto";
  elements.promptInput.style.height = `${Math.min(150, Math.max(34, elements.promptInput.scrollHeight))}px`;
  elements.characterCount.textContent = String(elements.promptInput.value.length);
}

function setRailTab(tab) {
  state.railTab = tab;
  document.querySelectorAll("[data-rail-tab]").forEach((button) => {
    const active = button.dataset.railTab === tab;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  elements.sessionsPanel.hidden = tab !== "sessions";
  elements.filesPanel.hidden = tab !== "files";
  if (tab === "files" && !elements.fileList.children.length) loadDirectory("");
}

async function loadDirectory(path) {
  elements.fileList.innerHTML = '<div class="loading-list">READING WORKSPACE</div>';
  const payload = await api(`/api/files?path=${encodeURIComponent(path)}`);
  state.directoryPath = payload.path;
  state.directoryParent = payload.parent;
  elements.filePathButton.textContent = payload.path ? `/${payload.path}` : "/";
  elements.filePathButton.title = payload.path ? `/${payload.path}` : "/";
  elements.fileUpButton.disabled = payload.parent === null;
  elements.fileList.replaceChildren();
  if (!payload.entries.length) {
    elements.fileList.innerHTML = '<div class="empty-list">EMPTY DIRECTORY</div>';
    return;
  }
  for (const entry of payload.entries) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "file-row";
    row.dataset.kind = entry.kind;
    row.classList.toggle("is-active", entry.path === state.selectedFile);
    const kindLabel = entry.kind === "directory" ? "DIR" : (entry.name.split(".").pop() || "FILE").toUpperCase().slice(0, 5);
    row.innerHTML = `${icon(entry.kind === "directory" ? "folder" : "file")}<span class="file-name">${escapeHtml(entry.name)}</span><span class="file-kind">${escapeHtml(kindLabel)}</span>`;
    row.addEventListener("click", () => {
      if (entry.kind === "directory") loadDirectory(entry.path);
      else openFile(entry.path);
    });
    elements.fileList.append(row);
  }
}

async function openFile(path) {
  state.selectedFile = path;
  state.selectedFileContent = null;
  showInspector();
  elements.signalInspector.hidden = true;
  elements.fileInspector.hidden = false;
  elements.inspectorTitle.textContent = path.split("/").pop();
  elements.fileInspectorPath.textContent = path;
  elements.filePreview.innerHTML = '<div class="loading-list">LOADING FILE</div>';
  elements.copyFileButton.hidden = true;
  renderFileSelection();
  try {
    const payload = await api(`/api/file?path=${encodeURIComponent(path)}`);
    elements.filePreview.replaceChildren();
    if (payload.kind === "text") {
      state.selectedFileContent = payload.content;
      const pre = document.createElement("pre");
      pre.textContent = payload.content;
      elements.filePreview.append(pre);
      elements.copyFileButton.hidden = false;
    } else if (payload.kind === "image") {
      const image = document.createElement("img");
      image.src = payload.rawUrl;
      image.alt = payload.name;
      elements.filePreview.append(image);
    } else if (payload.kind === "pdf") {
      const frame = document.createElement("iframe");
      frame.src = payload.rawUrl;
      frame.title = payload.name;
      elements.filePreview.append(frame);
    }
    addActivity(`Opened ${path}`);
  } catch (error) {
    elements.filePreview.innerHTML = `<div class="file-preview-error">${escapeHtml(error.message || String(error))}</div>`;
  }
}

function renderFileSelection() {
  elements.fileList.querySelectorAll(".file-row").forEach((row) => {
    const name = row.querySelector(".file-name")?.textContent;
    const fullPath = state.directoryPath ? `${state.directoryPath}/${name}` : name;
    row.classList.toggle("is-active", fullPath === state.selectedFile);
  });
}

function showSignalInspector() {
  elements.fileInspector.hidden = true;
  elements.signalInspector.hidden = false;
  elements.inspectorTitle.textContent = "运行信号";
}

function showInspector() {
  elements.workspace.classList.remove("inspector-closed");
  elements.workspace.classList.add("inspector-open");
}

function hideInspector() {
  elements.workspace.classList.remove("inspector-open");
  elements.workspace.classList.add("inspector-closed");
}

function toggleInspector() {
  const open = elements.workspace.classList.contains("inspector-open") || !elements.workspace.classList.contains("inspector-closed");
  if (open) hideInspector();
  else showInspector();
}

function closeNavigation() {
  elements.workspace.classList.remove("nav-open");
  elements.scrim.hidden = true;
}

function toggleNavigation() {
  const next = !elements.workspace.classList.contains("nav-open");
  elements.workspace.classList.toggle("nav-open", next);
  elements.scrim.hidden = !next;
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("piy-web-theme", theme);
}

function initializeTheme() {
  const saved = localStorage.getItem("piy-web-theme");
  const preferred = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  applyTheme(saved || preferred);
}

function bindEvents() {
  elements.themeButton.addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });
  elements.inspectorButton.addEventListener("click", toggleInspector);
  elements.closeInspectorButton.addEventListener("click", hideInspector);
  elements.mobileMenuButton.addEventListener("click", toggleNavigation);
  elements.scrim.addEventListener("click", closeNavigation);
  document.querySelectorAll("[data-rail-tab]").forEach((button) => {
    button.addEventListener("click", () => setRailTab(button.dataset.railTab));
  });
  elements.sessionSearch.addEventListener("input", renderSessions);
  elements.newSessionButton.addEventListener("click", () => newSession().catch(reportError));
  elements.refreshSessionsButton.addEventListener("click", () => loadSessions().catch(reportError));
  elements.refreshFilesButton.addEventListener("click", () => loadDirectory(state.directoryPath).catch(reportError));
  elements.fileUpButton.addEventListener("click", () => {
    if (state.directoryParent !== null) loadDirectory(state.directoryParent).catch(reportError);
  });
  elements.filePathButton.addEventListener("click", () => loadDirectory("").catch(reportError));
  elements.modelSelect.addEventListener("change", () => changeModel().catch(reportError));
  elements.thinkingSelect.addEventListener("change", () => changeThinking().catch(reportError));
  elements.promptInput.addEventListener("input", resizePrompt);
  elements.promptInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      sendPrompt();
    }
  });
  elements.sendButton.addEventListener("click", sendPrompt);
  elements.copyFileButton.addEventListener("click", async () => {
    if (state.selectedFileContent === null) return;
    await navigator.clipboard.writeText(state.selectedFileContent);
    showToast("文件内容已复制");
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeNavigation();
      if (window.innerWidth <= 920) hideInspector();
    }
  });
}

function reportError(error) {
  showToast(error.message || String(error), "error");
}

async function initialize() {
  initializeTheme();
  if (window.innerWidth <= 920) hideInspector();
  else showInspector();
  bindEvents();
  setRunState("ready", "READY");
  const health = await api("/api/health");
  elements.projectPath.textContent = health.project;
  elements.projectPath.title = health.project;
  elements.emptyPath.textContent = health.project;
  await Promise.all([loadModels(), loadSessions(), loadDirectory("")]);
  if (state.sessions.length) await selectSession(state.sessions[0].id, false);
  else await newSession();
}

initialize().catch((error) => {
  setRunState("error", "OFFLINE");
  reportError(error);
});
