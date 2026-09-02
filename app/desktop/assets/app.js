"use strict";

const $ = (id) => document.getElementById(id);
const elements = {
  runtimeChip: $("runtime-chip"), errorNotice: $("error-notice"),
  topicInput: $("topic-input"), selectedAngle: $("selected-angle"),
  lengthSlider: $("length-slider"), lengthOutput: $("length-output"), snapTrack: $("snap-track"),
  generateButton: $("generate-button"), recommendationButton: $("recommendation-button"),
  progressStep: $("progress-step"), progressTitle: $("progress-title"), progressSubtitle: $("progress-subtitle"),
  progressBackButton: $("progress-back-button"), cancelButton: $("cancel-button"),
  activityLine: $("activity-line"), logConsole: $("log-console"),
  progressNote: $("progress-note"), progressRetryButton: $("progress-retry-button"),
  pickerBackButton: $("picker-back-button"), recommendationList: $("recommendation-list"),
  scriptMeta: $("script-meta"), scriptBody: $("script-body"), copyButton: $("copy-button"),
  newScriptButton: $("new-script-button"), evaluateButton: $("evaluate-button"),
  reportBackButton: $("report-back-button"), reportNewButton: $("report-new-button"),
  overallScore: $("overall-score"), reportSummaryText: $("report-summary-text"),
  reportTags: $("report-tags"), dimensionGrid: $("dimension-grid"),
  oralPanel: $("oral-panel"), oralSubscores: $("oral-subscores"),
  groupPanel: $("group-panel"), judgeGroups: $("judge-groups"),
  findingsPanel: $("findings-panel"), findingsList: $("findings-list"), toast: $("toast"),
};

const views = {
  compose: $("compose-view"), recommendation_progress: $("progress-view"),
  recommendation_picker: $("recommendation-picker-view"), generation_progress: $("progress-view"),
  result: $("result-view"), evaluation_progress: $("progress-view"), report: $("report-view"),
};

const state = {
  ready: false, currentStage: "compose", selectedRecommendation: null,
  defaultTopic: HyTopicDefaults.choose(),
  currentRunId: null, currentScript: "", currentReport: null,
  recommendation: { jobId: null, status: "idle", lastSeq: 0, events: [], result: null, error: null, pollTimer: null },
  foreground: { jobId: null, kind: null, lastSeq: 0, pollTimer: null },
  slider: { snapPoints: [280, 450, 700], enterPixels: 14, releasePixels: 24, pointerId: null, attachedPoint: null },
};

elements.topicInput.placeholder = `比如：${state.defaultTopic}`;

const progressLabels = {
  recommendation_progress: ["LIVE SIGNALS", "正在准备推荐选题", "读取公开热榜并生成 20 条创作方向"],
  generation_progress: ["RESEARCH & WRITE", "正在调研并生成文案", "搜索、整理背景并撰写口播正文"],
  evaluation_progress: ["FORMAL REVIEW", "正在评估文案质量", "运行长度规则与 Hy3 七维 Judge"],
};

function show(element) { element.classList.remove("hidden"); }
function hide(element) { element.classList.add("hidden"); }
function clear(element) { while (element.firstChild) element.removeChild(element.firstChild); }
function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}
function showError(message) { elements.errorNotice.textContent = message; show(elements.errorNotice); }
function clearError() { elements.errorNotice.textContent = ""; hide(elements.errorNotice); }

let toastTimer = null;
function toast(message) {
  elements.toast.textContent = message; show(elements.toast);
  if (toastTimer) window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => hide(elements.toast), 2400);
}

function setStage(stage) {
  state.currentStage = stage;
  const target = views[stage];
  for (const view of new Set(Object.values(views))) hide(view);
  show(target);
  document.body.dataset.stage = stage;
  if (stage === "report") target.scrollTop = 0;
  if (stage === "result") elements.scriptBody.scrollTop = 0;
  if (window.pywebview?.api?.set_stage) window.pywebview.api.set_stage(stage).catch(() => {});
}

function updateComposeControls() {
  elements.generateButton.disabled = !state.ready || Boolean(state.foreground.jobId);
  elements.recommendationButton.disabled = !state.ready;
  elements.topicInput.disabled = Boolean(state.foreground.jobId);
  elements.lengthSlider.disabled = Boolean(state.foreground.jobId);
}

function updateRange() {
  const slider = elements.lengthSlider;
  const minimum = Number(slider.min), maximum = Number(slider.max), value = Number(slider.value);
  slider.style.setProperty("--range-progress", `${((value - minimum) / (maximum - minimum)) * 100}%`);
  elements.lengthOutput.textContent = `${value} 字`;
  slider.classList.toggle("magnetized", state.slider.attachedPoint === value);
  for (const button of elements.snapTrack.querySelectorAll(".snap-button")) {
    button.classList.toggle("active", Number(button.dataset.value) === value);
  }
}

function configureRange(config) {
  const slider = elements.lengthSlider;
  slider.min = String(config.min); slider.max = String(config.max); slider.value = String(config.default);
  state.slider.snapPoints = config.snap_points;
  state.slider.enterPixels = config.snap_enter_pixels;
  state.slider.releasePixels = config.snap_release_pixels;
  clear(elements.snapTrack);
  for (const value of state.slider.snapPoints) {
    const button = node("button", "snap-button", value);
    button.type = "button"; button.dataset.value = String(value);
    button.style.left = `${((value - config.min) / (config.max - config.min)) * 100}%`;
    button.addEventListener("click", () => { slider.value = String(value); state.slider.attachedPoint = value; updateRange(); });
    elements.snapTrack.appendChild(button);
  }
  state.slider.attachedPoint = config.default;
  updateRange();
}

function updateSliderFromPointer(event) {
  const slider = elements.lengthSlider, rect = slider.getBoundingClientRect();
  const minimum = Number(slider.min), maximum = Number(slider.max);
  const rawValue = HySlider.pointerValue(event.clientX, rect.left, rect.width, minimum, maximum);
  const result = HySlider.magnetize({
    rawValue, width: rect.width, minimum, maximum, snapPoints: state.slider.snapPoints,
    attachedPoint: state.slider.attachedPoint, enterPixels: state.slider.enterPixels,
    releasePixels: state.slider.releasePixels,
  });
  state.slider.attachedPoint = result.attachedPoint;
  slider.value = String(result.value);
  updateRange();
}

elements.lengthSlider.addEventListener("pointerdown", (event) => {
  if (elements.lengthSlider.disabled) return;
  event.preventDefault(); state.slider.pointerId = event.pointerId;
  try { elements.lengthSlider.setPointerCapture(event.pointerId); } catch (_) {}
  updateSliderFromPointer(event);
});
elements.lengthSlider.addEventListener("pointermove", (event) => {
  if (state.slider.pointerId === event.pointerId) updateSliderFromPointer(event);
});
function finishSliderPointer(event) {
  if (state.slider.pointerId !== event.pointerId) return;
  updateSliderFromPointer(event);
  try { elements.lengthSlider.releasePointerCapture(event.pointerId); } catch (_) {}
  state.slider.pointerId = null;
}
elements.lengthSlider.addEventListener("pointerup", finishSliderPointer);
elements.lengthSlider.addEventListener("pointercancel", finishSliderPointer);
elements.lengthSlider.addEventListener("keydown", () => { state.slider.attachedPoint = null; });
elements.lengthSlider.addEventListener("input", () => {
  if (state.slider.pointerId === null) { state.slider.attachedPoint = null; updateRange(); }
});

function appendEvents(events, target = elements.logConsole) {
  for (const event of events) {
    const line = node("div", `log-line ${event.level}`);
    const time = String(event.timestamp).split("T")[1]?.slice(0, 8) || "--:--:--";
    line.appendChild(node("span", "log-time", time));
    line.appendChild(node("span", "log-level", event.level));
    line.appendChild(node("span", "log-message", event.message));
    target.appendChild(line);
  }
  while (target.children.length > 500) target.removeChild(target.firstChild);
  target.scrollTop = target.scrollHeight;
}

function configureProgress(stage, events = []) {
  const labels = progressLabels[stage];
  elements.progressStep.textContent = labels[0]; elements.progressTitle.textContent = labels[1];
  elements.progressSubtitle.textContent = labels[2]; clear(elements.logConsole); appendEvents(events);
  elements.activityLine.classList.remove("stopped"); hide(elements.progressRetryButton);
  elements.cancelButton.disabled = false;
  if (stage === "recommendation_progress") {
    show(elements.progressBackButton); elements.cancelButton.textContent = "停止推荐";
    elements.progressNote.textContent = "返回填写不会停止后台推荐任务。";
  } else {
    hide(elements.progressBackButton); elements.cancelButton.textContent = "取消任务";
    elements.progressNote.textContent = "已发出的远端请求在取消后仍可能产生费用。";
  }
  setStage(stage);
}

function terminalRecommendation(message, retry) {
  elements.activityLine.classList.add("stopped"); elements.progressNote.textContent = message;
  elements.cancelButton.disabled = true;
  if (retry) show(elements.progressRetryButton);
}

function failRecommendation(message) {
  state.recommendation.status = "failed";
  state.recommendation.error = message;
  if (state.currentStage === "recommendation_progress") {
    terminalRecommendation(message, true);
  }
}

async function startRecommendations() {
  state.recommendation.status = "queued"; state.recommendation.error = null;
  let response;
  try { response = await window.pywebview.api.start_recommendations(); }
  catch (_) { failRecommendation("无法启动推荐任务。"); return false; }
  if (!response.ok) {
    failRecommendation(response.error || "无法启动推荐任务。");
    return false;
  }
  if (state.recommendation.jobId !== response.job_id) {
    state.recommendation.jobId = response.job_id; state.recommendation.lastSeq = 0;
    state.recommendation.events = []; state.recommendation.result = null;
  }
  state.recommendation.status = response.status || "queued";
  scheduleRecommendationPoll(50); return true;
}

function scheduleRecommendationPoll(delay = 350) {
  if (state.recommendation.pollTimer) window.clearTimeout(state.recommendation.pollTimer);
  state.recommendation.pollTimer = window.setTimeout(pollRecommendation, delay);
}

async function pollRecommendation() {
  const rec = state.recommendation;
  if (!rec.jobId) return;
  let response;
  try { response = await window.pywebview.api.poll_job(rec.jobId, rec.lastSeq); }
  catch (_) { scheduleRecommendationPoll(700); return; }
  if (!response.ok) { failRecommendation(response.error || "推荐任务状态不可用。"); return; }
  const events = response.events || [];
  if (events.length) {
    rec.lastSeq = events[events.length - 1].seq; rec.events.push(...events);
    if (state.currentStage === "recommendation_progress") appendEvents(events);
  }
  rec.status = response.status; rec.error = response.error;
  if (["queued", "running", "cancelling"].includes(response.status)) { scheduleRecommendationPoll(); return; }
  if (response.status === "succeeded") {
    rec.result = response.result; renderRecommendations(response.result.recommendations);
    if (state.currentStage === "recommendation_progress") setStage("recommendation_picker");
  } else if (state.currentStage === "recommendation_progress") {
    terminalRecommendation(response.error || "推荐任务已停止。", true);
  }
}

async function openRecommendations() {
  clearError();
  if (state.recommendation.status === "succeeded" && state.recommendation.result) { setStage("recommendation_picker"); return; }
  if (["failed", "cancelled", "idle"].includes(state.recommendation.status)) await startRecommendations();
  configureProgress("recommendation_progress", state.recommendation.events);
  if (["failed", "cancelled"].includes(state.recommendation.status)) terminalRecommendation(state.recommendation.error || "推荐任务未完成。", true);
}

function renderRecommendations(items) {
  clear(elements.recommendationList);
  for (const item of items) {
    const card = node("button", "recommendation-card"); card.type = "button";
    card.appendChild(node("strong", "", item.title));
    card.appendChild(node("p", "", `${item.angle} · ${item.why_now}`));
    const sources = item.sources.slice(0, 3).map((source) => source.title).join(" / ");
    card.appendChild(node("span", "source-line", `信号来源：${sources || "公开热榜"}`));
    card.addEventListener("click", () => {
      state.selectedRecommendation = item; elements.topicInput.value = item.title;
      elements.selectedAngle.textContent = `创作角度：${item.angle}`; show(elements.selectedAngle);
      setStage("compose"); elements.topicInput.focus();
    });
    elements.recommendationList.appendChild(card);
  }
}

async function startForeground(kind, startCall) {
  clearError();
  let response;
  try { response = await startCall(); }
  catch (_) { showError("无法连接桌面任务桥，请重新启动应用。"); return false; }
  if (!response.ok) { showError(response.error || "任务无法开始。"); return false; }
  const stage = kind === "generation" ? "generation_progress" : "evaluation_progress";
  state.foreground = { jobId: response.job_id, kind, lastSeq: 0, pollTimer: null };
  configureProgress(stage); updateComposeControls(); scheduleForegroundPoll(50); return true;
}

function scheduleForegroundPoll(delay = 300) {
  if (state.foreground.pollTimer) window.clearTimeout(state.foreground.pollTimer);
  state.foreground.pollTimer = window.setTimeout(pollForeground, delay);
}

async function pollForeground() {
  const foreground = state.foreground;
  if (!foreground.jobId) return;
  let response;
  try { response = await window.pywebview.api.poll_job(foreground.jobId, foreground.lastSeq); }
  catch (_) { scheduleForegroundPoll(650); return; }
  if (!response.ok) { finishForeground("failed", null, response.error); return; }
  const events = response.events || [];
  if (events.length) { foreground.lastSeq = events[events.length - 1].seq; appendEvents(events); }
  if (["queued", "running", "cancelling"].includes(response.status)) { scheduleForegroundPoll(); return; }
  finishForeground(response.status, response.result, response.error);
}

function finishForeground(status, result, error) {
  const kind = state.foreground.kind;
  if (state.foreground.pollTimer) window.clearTimeout(state.foreground.pollTimer);
  state.foreground = { jobId: null, kind: null, lastSeq: 0, pollTimer: null }; updateComposeControls();
  if (status === "succeeded") {
    if (kind === "generation") renderGeneration(result); else renderReport(result);
    return;
  }
  showError(error || (status === "cancelled" ? "任务已取消。" : "任务执行失败。"));
  setStage(kind === "generation" ? "compose" : "result");
}

async function startGeneration() {
  const typedTopic = elements.topicInput.value.trim();
  const topic = HyTopicDefaults.resolve(typedTopic, state.defaultTopic);
  if (!typedTopic) {
    elements.topicInput.value = topic;
    state.selectedRecommendation = null;
    hide(elements.selectedAngle);
  }
  const angle = state.selectedRecommendation ? state.selectedRecommendation.angle : "";
  await startForeground("generation", () => window.pywebview.api.start_generation({ topic, angle, target_length: Number(elements.lengthSlider.value) }));
}

function renderGeneration(result) {
  state.currentRunId = result.run_id; state.currentScript = result.script_text; state.currentReport = null;
  elements.scriptBody.textContent = result.script_text; clear(elements.scriptMeta);
  elements.scriptMeta.appendChild(node("span", "", result.topic));
  elements.scriptMeta.appendChild(node("span", "", `目标 ${result.target_length} 字`));
  elements.scriptMeta.appendChild(node("span", "", `实际 ${result.character_count} 字`));
  setStage("result");
}

async function startEvaluation() {
  if (state.currentRunId) await startForeground("evaluation", () => window.pywebview.api.start_evaluation(state.currentRunId));
}

async function cancelProgressJob() {
  const jobId = state.currentStage === "recommendation_progress" ? state.recommendation.jobId : state.foreground.jobId;
  if (!jobId) return;
  elements.cancelButton.disabled = true;
  const response = await window.pywebview.api.cancel_job(jobId);
  if (!response.ok) { showError(response.error || "无法取消任务。"); elements.cancelButton.disabled = false; }
}

function addQuotes(parent, label, spans, problem) {
  if (!Array.isArray(spans) || !spans.length) return;
  const section = node("div", "quote-section"); section.appendChild(node("span", "quote-label", label));
  for (const span of spans) section.appendChild(node("blockquote", `script-quote${problem ? " problem" : ""}`, span));
  parent.appendChild(section);
}
function renderDimensions(dimensions) {
  clear(elements.dimensionGrid);
  for (const dimension of dimensions) {
    const card = node("article", "dimension-card"), title = node("div", "dimension-title");
    title.appendChild(node("h3", "", dimension.name)); title.appendChild(node("span", "score-badge", `${dimension.score ?? "—"} / ${dimension.score_max}`));
    card.appendChild(title); card.appendChild(node("p", "dimension-reason", dimension.reason));
    addQuotes(card, "成立原文", dimension.positive_spans, false); addQuotes(card, "问题原文", dimension.problem_spans, true);
    elements.dimensionGrid.appendChild(card);
  }
}
function renderOralSubscores(subscores) {
  clear(elements.oralSubscores); const entries = Object.entries(subscores || {});
  if (!entries.length) { hide(elements.oralPanel); return; }
  for (const [name, item] of entries) {
    const block = node("div", "subscore"), head = node("div", "subscore-head");
    head.appendChild(node("span", "", name)); head.appendChild(node("span", "", `${item.score} / 3`));
    block.appendChild(head); block.appendChild(node("p", "", item.comment));
    addQuotes(block, "成立原文", item.positive_spans, false); addQuotes(block, "问题原文", item.problem_spans, true);
    elements.oralSubscores.appendChild(block);
  }
  show(elements.oralPanel);
}
function renderJudgeGroups(groups) {
  clear(elements.judgeGroups);
  if (!Array.isArray(groups) || !groups.length) { hide(elements.groupPanel); return; }
  for (const group of groups) { const block = node("div", "judge-group"); block.appendChild(node("strong", "", group.name)); block.appendChild(node("p", "", group.summary)); elements.judgeGroups.appendChild(block); }
  show(elements.groupPanel);
}
function renderFindings(findings) {
  clear(elements.findingsList);
  if (!Array.isArray(findings) || !findings.length) { hide(elements.findingsPanel); return; }
  for (const finding of findings) {
    const block = node("div", "finding"), head = node("div", "finding-head");
    head.appendChild(node("span", "", finding.code)); head.appendChild(node("span", `severity ${finding.severity}`, finding.severity));
    block.appendChild(head); block.appendChild(node("p", "", finding.message)); elements.findingsList.appendChild(block);
  }
  show(elements.findingsPanel);
}
function renderReport(report) {
  state.currentReport = report; elements.overallScore.textContent = report.score_percent ?? "—";
  elements.reportSummaryText.textContent = report.summary; clear(elements.reportTags);
  elements.reportTags.appendChild(node("span", "report-tag", `Rubric ${report.rubric_version}`));
  if (report.judge_model) elements.reportTags.appendChild(node("span", "report-tag", report.judge_model));
  elements.reportTags.appendChild(node("span", "report-tag", report.cached ? "已复用缓存" : "本次新评估"));
  if (!report.eligible) elements.reportTags.appendChild(node("span", "report-tag", "存在门控问题"));
  renderDimensions(report.dimensions || []); renderOralSubscores(report.oral_subscores || {});
  renderJudgeGroups(report.judge_groups || []); renderFindings(report.findings || []); setStage("report");
}

async function copyScript() {
  if (!state.currentScript) return;
  try { await navigator.clipboard.writeText(state.currentScript); }
  catch (_) {
    const textarea = document.createElement("textarea"); textarea.value = state.currentScript;
    textarea.style.position = "fixed"; textarea.style.opacity = "0"; document.body.appendChild(textarea);
    textarea.select(); document.execCommand("copy"); textarea.remove();
  }
  toast("文案已复制");
}
function resetForNewScript() {
  state.currentRunId = null; state.currentScript = ""; state.currentReport = null;
  elements.topicInput.value = ""; state.selectedRecommendation = null; hide(elements.selectedAngle);
  clearError(); setStage("compose"); elements.topicInput.focus();
}

elements.recommendationButton.addEventListener("click", openRecommendations);
elements.generateButton.addEventListener("click", startGeneration);
elements.progressBackButton.addEventListener("click", () => setStage("compose"));
elements.cancelButton.addEventListener("click", cancelProgressJob);
elements.progressRetryButton.addEventListener("click", async () => { await startRecommendations(); configureProgress("recommendation_progress", state.recommendation.events); });
elements.pickerBackButton.addEventListener("click", () => setStage("compose"));
elements.copyButton.addEventListener("click", copyScript);
elements.newScriptButton.addEventListener("click", resetForNewScript);
elements.evaluateButton.addEventListener("click", startEvaluation);
elements.reportBackButton.addEventListener("click", () => setStage("result"));
elements.reportNewButton.addEventListener("click", resetForNewScript);
elements.topicInput.addEventListener("input", () => {
  if (state.selectedRecommendation && elements.topicInput.value !== state.selectedRecommendation.title) {
    state.selectedRecommendation = null; hide(elements.selectedAngle);
  }
});

window.addEventListener("pywebviewready", async () => {
  let bootstrap;
  try { bootstrap = await window.pywebview.api.bootstrap(); }
  catch (_) { elements.runtimeChip.textContent = "桌面桥不可用"; showError("无法初始化桌面任务桥，请重新启动应用。"); return; }
  configureRange(bootstrap.length);
  const diagnostic = bootstrap.diagnostic;
  elements.runtimeChip.textContent = `${diagnostic.platform} · ${diagnostic.backend}`;
  elements.runtimeChip.title = diagnostic.message; state.ready = bootstrap.ready;
  if (!bootstrap.ready) { showError(bootstrap.configuration_error || diagnostic.message); updateComposeControls(); return; }
  elements.runtimeChip.classList.add("ready"); updateComposeControls(); setStage("compose");
  startRecommendations();
});
