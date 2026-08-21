(function (root, factory) {
  const api = factory(root);
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.TrainingAssistantUI = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function (root) {
  const labels = {
    collecting_requirements: "需求确认中",
    requirements_review: "待确认需求",
    data_plan_preparing: "数据方案准备中",
    data_plan_ready: "待批准数据构建",
    data_generating: "数据构建中",
    data_review: "数据审查中",
    train_plan_ready: "训练方案已生成",
    preflight_blocked: "预检未通过",
    train_ready: "待批准训练",
    training: "训练中",
    train_failed: "训练失败",
    ab_plan_ready: "待批准 A/B 评测",
    evaluating: "A/B 评测中",
    diagnosis_ready: "评测诊断已完成",
    cancelling: "正在停止",
    cancelled: "已中止",
    completed: "已完成",
  };

  const stepStatusLabels = {
    pending: "未开始",
    active: "进行中",
    needs_confirmation: "待确认",
    blocked: "已阻断",
    failed: "失败",
    cancelling: "正在停止",
    cancelled: "已中止",
    succeeded: "已完成",
    skipped: "已跳过",
  };

  function statusLabel(status) {
    return labels[status] || status;
  }

  function numberText(value) {
    if (!Number.isFinite(Number(value))) return "";
    return Number.isInteger(Number(value))
      ? String(Number(value))
      : String(Math.round(Number(value) * 100) / 100);
  }

  function stepView(step) {
    const progress = step.progress || null;
    let progressText = "";
    if (progress) {
      const current = numberText(progress.current);
      const target = numberText(progress.target);
      const percentage = numberText(progress.percentage);
      if (current && target) progressText = `${current} / ${target}`;
      if (percentage) progressText += `${progressText ? " · " : ""}${percentage}%`;
    }
    return {
      ...step,
      statusText: stepStatusLabels[step.status] || step.status,
      progressText,
      progressPercentage: progress && Number.isFinite(Number(progress.percentage))
        ? Math.max(0, Math.min(100, Number(progress.percentage)))
        : null,
    };
  }

  function workflowTimeline(snapshot) {
    return (snapshot.workflow_steps || []).map(stepView);
  }

  function cancelView(snapshot) {
    const busy = snapshot.state === "cancelling";
    const visible = busy || (snapshot.available_actions || []).includes("cancel");
    return {
      visible,
      busy,
      label: busy ? "正在停止…" : "中止流程",
    };
  }

  function safeApiUrl(value) {
    if (typeof value !== "string" || !value.startsWith("/api/")) return null;
    if (value.includes("\\") || value.includes("..") || value.startsWith("//")) return null;
    return value;
  }

  function artifactView(item) {
    const sha = typeof item.sha256 === "string" && item.sha256
      ? item.sha256.replace(/^sha256:/, "")
      : "";
    return {
      name: item.name || item.kind || "产物",
      kind: item.kind || "artifact",
      downloadUrl: safeApiUrl(item.download_url),
      previewUrl: safeApiUrl(item.preview_url),
      hashLabel: sha ? `sha256:${sha}` : "",
    };
  }

  function requirementFieldView(field) {
    const sourceLabels = {
      user: "用户明确",
      default: "系统默认",
      assistant_assumption: "助手假设",
    };
    return {
      value: field && field.value !== undefined ? field.value : "",
      source: (field && field.source) || "assistant_assumption",
      sourceText: sourceLabels[(field && field.source) || "assistant_assumption"],
      evidenceMessageIds: (field && field.evidence_message_ids) || [],
    };
  }

  function approvalView(item) {
    const titles = {
      confirm_requirements: "确认需求理解",
      start_datagen: "批准并开始数据构建",
      start_training: "批准并开始训练",
      start_evaluation: "批准并开始 A/B 评测",
      retry_scoring: "复用推理产物并重新评分",
      skip_evaluation: "跳过 A/B 评测",
      accept_candidate: "接受当前模型",
      finish_without_accepting: "结束本轮且不接受候选模型",
      start_iteration: "批准下一轮迭代",
      recover_training: "从 checkpoint 恢复训练",
    };
    return {
      title: titles[item.action] || item.action,
      planHash: item.plan_hash,
      summary: item.summary,
    };
  }

  function mergeEvents(oldRows, newRows) {
    const byId = new Map(oldRows.map((row) => [row.event_id, row]));
    newRows.forEach((row) => byId.set(row.event_id, row));
    return Array.from(byId.values()).sort((a, b) => a.event_id - b.event_id);
  }

  function trainingPlanView(plan) {
    if (!plan) return null;
    return {
      config: plan.config,
      decisions: plan.decisions,
      estimated_steps: plan.estimated_steps,
      estimated_vram_gb: plan.estimated_vram_gb,
      estimated_hours_low: plan.estimated_hours_low,
      estimated_hours_high: plan.estimated_hours_high,
      eta_confidence: plan.eta_confidence,
      eta_basis: plan.eta_basis,
      max_training_hours: plan.max_training_hours,
      risks: plan.risks,
    };
  }

  function latestComparison(events) {
    const rows = events || [];
    for (let index = rows.length - 1; index >= 0; index -= 1) {
      const row = rows[index];
      if (row.event_type === "evaluation_completed" && row.payload) {
        return row.payload.comparison || null;
      }
    }
    return null;
  }

  function workflowLabel(workflow) {
    const goal = workflow.objective && workflow.objective.goal
      ? workflow.objective.goal
      : workflow.workflow_id;
    const iteration = Number.isInteger(workflow.iteration) ? workflow.iteration + 1 : 1;
    return `${goal} · 第 ${iteration} 轮 · ${statusLabel(workflow.state)} · ${workflow.updated_at || ""}`;
  }

  function canRerunPreflight(status) {
    return ["train_plan_ready", "preflight_blocked", "train_ready"].includes(status);
  }

  function parseSseFrames(buffer) {
    const events = [];
    let rest = buffer;
    while (true) {
      const separator = rest.match(/\r?\n\r?\n/);
      if (!separator || separator.index === undefined) break;
      const frame = rest.slice(0, separator.index);
      rest = rest.slice(separator.index + separator[0].length);
      if (!frame.trim()) continue;
      let type = "message";
      const dataLines = [];
      frame.split(/\r?\n/).forEach((line) => {
        if (line.startsWith("event:")) type = line.slice(6).trim();
        if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      });
      const rawData = dataLines.join("\n");
      events.push({ type, data: rawData ? JSON.parse(rawData) : {} });
    }
    return { events, rest };
  }

  function assistantStreamPath(workflowId) {
    return workflowId
      ? `/api/assistant/workflows/${workflowId}/messages/stream`
      : "/api/assistant/workflows/stream";
  }

  async function consumeSse(response, onEvent) {
    if (!response.ok) {
      let body = {};
      try { body = await response.json(); } catch (_) { body = {}; }
      throw new Error(body.detail || `请求失败 (${response.status})`);
    }
    if (!response.body || !response.body.getReader) {
      throw new Error("当前浏览器不支持流式响应");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const parsed = parseSseFrames(buffer);
      buffer = parsed.rest;
      for (const event of parsed.events) await onEvent(event);
      if (done) break;
    }
    if (buffer.trim()) throw new Error("流式响应不完整，请重新加载任务状态");
  }

  if (!root || !root.document) {
    return {
      statusLabel,
      approvalView,
      mergeEvents,
      trainingPlanView,
      latestComparison,
      workflowLabel,
      canRerunPreflight,
      parseSseFrames,
      assistantStreamPath,
      consumeSse,
      stepView,
      workflowTimeline,
      cancelView,
      artifactView,
      requirementFieldView,
    };
  }

  const state = {
    currentId: root.localStorage.getItem("lf_assistant_workflow") || null,
    snapshot: null,
    events: [],
    timer: null,
    mounted: false,
    busy: false,
  };

  function element(tag, className, text) {
    const node = root.document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  async function api(path, options) {
    const response = await root.fetch(path, options);
    let body = {};
    try { body = await response.json(); } catch (_) { body = {}; }
    if (!response.ok) throw new Error(body.detail || `请求失败 (${response.status})`);
    return body;
  }

  function showError(error) {
    const host = root.document.getElementById("as-events");
    if (!host) return;
    const row = element("p", "assistant-error", error.message || String(error));
    host.prepend(row);
  }

  function renderWorkflows(workflows) {
    const host = root.document.getElementById("as-workflows");
    host.replaceChildren();
    if (!workflows.length) {
      host.appendChild(element("p", "hint", "暂无训练任务"));
      return;
    }
    workflows.forEach((workflow) => {
      const button = element(
        "button",
        `ghost assistant-workflow-row${workflow.workflow_id === state.currentId ? " active" : ""}`,
        workflowLabel(workflow),
      );
      button.type = "button";
      button.addEventListener("click", () => selectWorkflow(workflow.workflow_id));
      host.appendChild(button);
    });
  }

  async function loadWorkflows() {
    const data = await api("/api/assistant/workflows");
    const workflows = data.workflows || [];
    renderWorkflows(workflows);
    if (state.currentId && !state.snapshot) await loadSnapshot();
  }

  function renderMessages(messages) {
    const host = root.document.getElementById("as-messages");
    host.replaceChildren();
    (messages || []).forEach((message) => {
      const row = element("div", `assistant-message ${message.role}`);
      row.textContent = message.content || "";
      host.appendChild(row);
    });
    host.scrollTop = host.scrollHeight;
  }

  function renderStreamingMessage(userMessage, assistantText, progressText) {
    const messages = [
      ...((state.snapshot && state.snapshot.messages) || []),
      { role: "user", content: userMessage },
      { role: "assistant", content: assistantText || progressText },
    ];
    renderMessages(messages);
  }

  function appendJsonBlock(host, title, value) {
    if (value === null || value === undefined) return;
    const block = element("details", "assistant-tech-details");
    block.appendChild(element("summary", "", `查看技术详情 · ${title}`));
    const pre = element("pre");
    pre.textContent = JSON.stringify(value, null, 2);
    block.appendChild(pre);
    host.appendChild(block);
  }

  function setApprovalBusy(host, busy) {
    host.querySelectorAll("button").forEach((button) => { button.disabled = busy; });
  }

  async function actOnApproval(approval, approve) {
    if (state.busy) return;
    state.busy = true;
    const host = root.document.getElementById("as-current");
    setApprovalBusy(host, true);
    const suffix = approve ? "approve" : "reject";
    const options = { method: "POST", headers: {} };
    if (approve) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify({ plan_hash: approval.plan_hash });
    }
    try {
      state.snapshot = await api(
        `/api/assistant/workflows/${state.currentId}/approvals/${approval.approval_id}/${suffix}`,
        options,
      );
      state.events = state.snapshot.events || state.events;
      renderSnapshot();
      await loadWorkflows();
    } catch (error) {
      showError(error);
    } finally {
      state.busy = false;
      setApprovalBusy(host, false);
    }
  }

  function renderApprovals(host, approvals, preflight) {
    (approvals || []).forEach((approval) => {
      const view = approvalView(approval);
      const card = element("div", "assistant-approval");
      card.appendChild(element("h4", "", view.title));
      card.appendChild(element("p", "", view.summary || ""));
      card.appendChild(element("code", "hint", view.planHash || ""));
      if (approval.action === "start_training") {
        const warnings = ((preflight && preflight.checks) || [])
          .filter((check) => check.status === "warn");
        if (warnings.length) {
          const list = element("ul", "assistant-warning-list");
          warnings.forEach((warning) => {
            list.appendChild(element(
              "li",
              "",
              `${warning.name}: ${warning.summary}${warning.remediation ? `；${warning.remediation}` : ""}`,
            ));
          });
          card.appendChild(list);
        }
      }
      const buttons = element("div", "row");
      const approve = element("button", "", "确认执行");
      approve.type = "button";
      approve.addEventListener("click", () => actOnApproval(approval, true));
      const reject = element("button", "ghost", "拒绝");
      reject.type = "button";
      reject.addEventListener("click", () => actOnApproval(approval, false));
      buttons.append(approve, reject);
      card.appendChild(buttons);
      host.appendChild(card);
    });
  }

  async function rerunPreflight() {
    if (!state.currentId || state.busy) return;
    state.busy = true;
    try {
      state.snapshot = await api(
        `/api/assistant/workflows/${state.currentId}/preflight`,
        { method: "POST" },
      );
      state.events = state.snapshot.events || state.events;
      renderSnapshot();
    } catch (error) {
      showError(error);
    } finally {
      state.busy = false;
    }
  }

  async function updateWorkflow(path, options) {
    if (!state.currentId || state.busy) return;
    state.busy = true;
    const host = root.document.getElementById("as-current");
    setApprovalBusy(host, true);
    try {
      state.snapshot = await api(path, options || { method: "POST" });
      state.events = state.snapshot.events || state.events;
      renderSnapshot();
      await loadWorkflows();
    } catch (error) {
      showError(error);
    } finally {
      state.busy = false;
      setApprovalBusy(host, false);
    }
  }

  async function cancelWorkflow() {
    if (!state.currentId || state.busy) return;
    if (!root.confirm("停止当前流程和外部任务？已有数据、日志和模型产物会保留。")) return;
    await updateWorkflow(
      `/api/assistant/workflows/${state.currentId}/cancel`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason: "用户手动中止" }),
      },
    );
  }

  function renderRequirementEvidence(host, draft) {
    if (!draft) return;
    const section = element("div", "assistant-requirements");
    section.appendChild(element("h4", "", "需求证据"));
    const objective = draft.proposed_objective || {};
    const evidence = draft.field_evidence || {};
    const fields = [
      ["业务场景", draft.scenario],
      ["当前问题", draft.current_problem],
      ["期望行为", draft.desired_behavior],
      ["任务类型", { value: objective.task_types, ...(evidence.task_types || {}) }],
      ["基座模型", { value: objective.base_model_path, ...(evidence.base_model_path || {}) }],
      ["数据来源", { value: objective.data_source, ...(evidence.data_source || {}) }],
      ["成功标准", { value: objective.success_criteria, ...(evidence.success_criteria || {}) }],
    ];
    fields.forEach(([label, raw]) => {
      if (!raw || raw.value === undefined || raw.value === null) return;
      const view = requirementFieldView(raw);
      const row = element("div", "assistant-requirement-row");
      const heading = element("div", "assistant-requirement-heading");
      heading.appendChild(element("strong", "", label));
      heading.appendChild(element("span", `assistant-source ${view.source}`, view.sourceText));
      row.appendChild(heading);
      row.appendChild(element(
        "div",
        "assistant-requirement-value",
        typeof view.value === "string" ? view.value : JSON.stringify(view.value),
      ));
      if (view.evidenceMessageIds.length) {
        row.appendChild(element(
          "small",
          "hint",
          `用户消息 #${view.evidenceMessageIds.join(", #")}`,
        ));
      }
      section.appendChild(row);
    });
    (draft.assumptions || []).forEach((assumption) => {
      const row = element("div", "assistant-requirement-row");
      const heading = element("div", "assistant-requirement-heading");
      heading.appendChild(element("strong", "", "待确认假设"));
      heading.appendChild(element("span", "assistant-source assistant_assumption", "助手假设"));
      row.append(heading, element("div", "assistant-requirement-value", assumption));
      section.appendChild(row);
    });
    host.appendChild(section);
  }

  function renderArtifacts(host, artifacts) {
    if (!(artifacts || []).length) return;
    const list = element("ul", "assistant-artifacts");
    artifacts.forEach((artifact) => {
      const view = artifactView(artifact);
      const row = element("li");
      const url = view.downloadUrl || view.previewUrl;
      if (url) {
        const link = element("a", "", view.name);
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener";
        row.appendChild(link);
      } else {
        row.appendChild(element("span", "", view.name));
      }
      row.appendChild(element("small", "hint", ` ${view.kind}`));
      if (view.hashLabel) row.appendChild(element("code", "assistant-artifact-hash", view.hashLabel));
      list.appendChild(row);
    });
    host.appendChild(list);
  }

  function renderStepActions(host, actions) {
    const supported = {
      revise_requirements: ["修订需求", () => updateWorkflow(
        `/api/assistant/workflows/${state.currentId}/requirements/revise`,
        { method: "POST" },
      )],
      retry_data_plan: ["重试生成数据方案", () => updateWorkflow(
        `/api/assistant/workflows/${state.currentId}/data-plan/retry`,
        { method: "POST" },
      )],
      rerun_preflight: ["重新执行预检", rerunPreflight],
    };
    const buttons = element("div", "assistant-step-actions");
    (actions || []).forEach((action) => {
      if (!supported[action]) return;
      const [label, handler] = supported[action];
      const button = element("button", "ghost", label);
      button.type = "button";
      button.addEventListener("click", handler);
      buttons.appendChild(button);
    });
    if (buttons.childNodes.length) host.appendChild(buttons);
  }

  function renderTimeline(host, snapshot) {
    const timeline = element("div", "assistant-timeline");
    timeline.setAttribute("aria-label", "训练流程八个阶段");
    workflowTimeline(snapshot).forEach((step, index) => {
      const card = element("article", `assistant-step status-${step.status}`);
      const header = element("div", "assistant-step-header");
      header.appendChild(element("span", "assistant-step-number", step.sequence));
      header.appendChild(element("h4", "", step.title));
      header.appendChild(element("span", "assistant-step-status", step.statusText));
      card.appendChild(header);
      if (step.summary) card.appendChild(element("p", "assistant-step-summary", step.summary));
      if (index === 0) renderRequirementEvidence(card, snapshot.requirement_draft);
      if (step.progress) {
        const progressText = element(
          "div",
          "assistant-progress-text",
          step.progressText || "已连接，等待可量化进度",
        );
        progressText.setAttribute("aria-live", "polite");
        card.appendChild(progressText);
        if (step.progressPercentage !== null) {
          const track = element("div", "assistant-progress");
          track.setAttribute("role", "progressbar");
          track.setAttribute("aria-valuemin", "0");
          track.setAttribute("aria-valuemax", "100");
          track.setAttribute("aria-valuenow", String(step.progressPercentage));
          const fill = element("i");
          fill.style.width = `${step.progressPercentage}%`;
          track.appendChild(fill);
          card.appendChild(track);
        }
        if (Number.isFinite(Number(step.progress.eta_seconds))) {
          card.appendChild(element(
            "small", "hint", `预计剩余 ${Math.ceil(Number(step.progress.eta_seconds) / 60)} 分钟`,
          ));
        }
      }
      if ((step.issues || []).length) {
        const issues = element("ul", "assistant-issues");
        step.issues.forEach((issue) => issues.appendChild(element(
          "li", "", issue.message || issue.summary || issue.code || String(issue),
        )));
        card.appendChild(issues);
      }
      if ((step.decisions || []).length) {
        appendJsonBlock(card, "本阶段决策", step.decisions);
      }
      renderArtifacts(card, step.artifacts);
      renderStepActions(card, step.actions);
      timeline.appendChild(card);
    });
    host.appendChild(timeline);
  }

  function renderCurrent(snapshot) {
    const host = root.document.getElementById("as-current");
    host.replaceChildren();
    if (!snapshot) {
      host.appendChild(element("span", "hint", "选择或创建一个训练任务"));
      return;
    }
    host.appendChild(element("h3", "", statusLabel(snapshot.state)));
    const taskMeta = element(
      "p",
      "hint",
      `${snapshot.workflow_id} · 第 ${(snapshot.iteration || 0) + 1} 轮 · ${snapshot.updated_at || ""}`,
    );
    host.appendChild(taskMeta);
    const cancel = cancelView(snapshot);
    if (cancel.visible) {
      const button = element("button", "danger assistant-cancel", cancel.label);
      button.type = "button";
      button.disabled = cancel.busy;
      if (!cancel.busy) button.addEventListener("click", cancelWorkflow);
      host.appendChild(button);
    }
    renderTimeline(host, snapshot);
    renderApprovals(host, snapshot.pending_approvals, snapshot.preflight);
    const technical = element("div", "assistant-technical");
    appendJsonBlock(technical, "训练目标", snapshot.objective);
    appendJsonBlock(technical, "数据方案", snapshot.data_plan);
    appendJsonBlock(technical, "数据画像", snapshot.dataset_profile);
    appendJsonBlock(technical, "训练决策", trainingPlanView(snapshot.training_plan));
    appendJsonBlock(technical, "预检", snapshot.preflight);
    appendJsonBlock(technical, "A/B 方案", snapshot.evaluation_plan);
    appendJsonBlock(technical, "A/B 对照分析", latestComparison(state.events));
    appendJsonBlock(technical, "诊断", snapshot.diagnosis);
    host.appendChild(technical);
  }

  function renderEvents(events) {
    const host = root.document.getElementById("as-events");
    host.replaceChildren();
    host.appendChild(element("h3", "", "运行事件"));
    if (!events.length) {
      host.appendChild(element("p", "hint", "暂无事件"));
      return;
    }
    events.slice(-30).reverse().forEach((event) => {
      const row = element("div", "assistant-block");
      row.appendChild(element("strong", "", event.event_type));
      const details = element("pre");
      details.textContent = JSON.stringify(event.payload || {}, null, 2);
      row.appendChild(details);
      host.appendChild(row);
    });
  }

  function renderSnapshot() {
    renderMessages((state.snapshot && state.snapshot.messages) || []);
    renderCurrent(state.snapshot);
    renderEvents(state.events);
  }

  async function loadSnapshot() {
    if (!state.currentId) return;
    try {
      state.snapshot = await api(`/api/assistant/workflows/${state.currentId}`);
      state.events = state.snapshot.events || [];
      renderSnapshot();
    } catch (error) {
      showError(error);
    }
  }

  async function selectWorkflow(workflowId) {
    state.currentId = workflowId;
    state.snapshot = null;
    state.events = [];
    root.localStorage.setItem("lf_assistant_workflow", workflowId);
    await loadSnapshot();
    await loadWorkflows();
  }

  async function sendMessage() {
    const input = root.document.getElementById("as-input");
    const message = input.value.trim();
    if (!message || state.busy) return;
    state.busy = true;
    root.document.getElementById("as-send").disabled = true;
    let assistantText = "";
    let completedSnapshot = null;
    renderStreamingMessage(message, "", "正在理解需求并生成方案…");
    try {
      const response = await root.fetch(assistantStreamPath(state.currentId), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      await consumeSse(response, async (event) => {
        if (event.type === "progress" && !assistantText) {
          renderStreamingMessage(message, "", event.data.message || "正在处理…");
        } else if (event.type === "assistant_delta") {
          assistantText += event.data.delta || "";
          renderStreamingMessage(message, assistantText, "");
        } else if (event.type === "snapshot") {
          completedSnapshot = event.data;
        } else if (event.type === "error") {
          throw new Error(event.data.detail || "助手处理失败");
        }
      });
      if (!completedSnapshot) throw new Error("流式响应缺少最终任务状态");
      state.snapshot = completedSnapshot;
      state.currentId = completedSnapshot.workflow_id;
      state.events = state.snapshot.events || [];
      root.localStorage.setItem("lf_assistant_workflow", state.currentId);
      input.value = "";
      renderSnapshot();
      await loadWorkflows();
    } catch (error) {
      showError(error);
    } finally {
      state.busy = false;
      root.document.getElementById("as-send").disabled = false;
    }
  }

  async function pollEvents() {
    if (!state.currentId || !isAssistantVisible()) return;
    const afterId = state.events.length
      ? state.events[state.events.length - 1].event_id
      : 0;
    try {
      const data = await api(
        `/api/assistant/workflows/${state.currentId}/events?after_id=${afterId}`,
      );
      const incoming = data.events || [];
      if (incoming.length) {
        state.events = mergeEvents(state.events, incoming);
        await loadSnapshot();
      }
    } catch (error) {
      showError(error);
    }
  }

  function isAssistantVisible() {
    const view = root.document.getElementById("view-assistant");
    return Boolean(view && view.style.display !== "none" && !root.document.hidden);
  }

  function syncPolling() {
    if (state.timer) {
      root.clearInterval(state.timer);
      state.timer = null;
    }
    if (isAssistantVisible()) {
      state.timer = root.setInterval(pollEvents, 10000);
      loadWorkflows().catch(showError);
    }
  }

  function onTabChanged(which) {
    if (!state.mounted) return;
    if (which === "assistant") loadSnapshot();
    syncPolling();
  }

  function newWorkflow() {
    state.currentId = null;
    state.snapshot = null;
    state.events = [];
    root.localStorage.removeItem("lf_assistant_workflow");
    renderSnapshot();
    root.document.getElementById("as-input").focus();
    loadWorkflows().catch(showError);
  }

  function mount() {
    if (state.mounted || !root.document.getElementById("as-send")) return;
    state.mounted = true;
    root.document.getElementById("as-send").addEventListener("click", sendMessage);
    root.document.getElementById("as-new").addEventListener("click", newWorkflow);
    root.document.getElementById("as-input").addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") sendMessage();
    });
    root.document.addEventListener("visibilitychange", syncPolling);
    root.addEventListener("beforeunload", () => {
      if (state.timer) root.clearInterval(state.timer);
    });
    renderSnapshot();
    loadWorkflows().catch(showError);
    if (state.currentId) loadSnapshot();
    syncPolling();
  }

  if (root.document.readyState === "loading") {
    root.document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }

  return {
    statusLabel,
    approvalView,
    mergeEvents,
    trainingPlanView,
    latestComparison,
    workflowLabel,
    canRerunPreflight,
    parseSseFrames,
    assistantStreamPath,
    consumeSse,
    stepView,
    workflowTimeline,
    cancelView,
    artifactView,
    requirementFieldView,
    onTabChanged,
    mount,
  };
});
