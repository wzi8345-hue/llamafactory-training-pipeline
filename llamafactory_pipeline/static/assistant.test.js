const test = require("node:test");
const assert = require("node:assert/strict");
const {
  statusLabel,
  approvalView,
  mergeEvents,
  trainingPlanView,
  latestComparison,
  workflowLabel,
  canRerunPreflight,
  parseSseFrames,
  assistantStreamPath,
  workflowTimeline,
  cancelView,
  artifactView,
  requirementFieldView,
} = require("./assistant.js");

function eightSteps(activeKey) {
  const titles = [
    "需求理解与确认", "数据策略规划", "数据构建", "数据质量复核",
    "训练方案与预检", "模型训练", "A/B 评测", "诊断与下一轮决策",
  ];
  const keys = [
    "requirements", "data_plan", "data_build", "data_review",
    "train_plan", "training", "evaluation", "diagnosis",
  ];
  return keys.map((key, index) => ({
    key,
    sequence: index + 1,
    title: titles[index],
    status: key === activeKey ? "active" : "pending",
    progress: key === "data_build" ? {
      current: 188,
      target: 1000,
      percentage: 18.8,
      eta_seconds: 1260,
      details: {},
    } : null,
    artifacts: [],
    issues: [],
    actions: [],
  }));
}

test("status labels are user readable", () => {
  assert.equal(statusLabel("train_ready"), "待批准训练");
  assert.equal(statusLabel("training"), "训练中");
});

test("approval view preserves plan hash", () => {
  const view = approvalView({
    action: "start_training",
    plan_hash: "sha256:abc",
    summary: "SFT",
  });
  assert.equal(view.planHash, "sha256:abc");
  assert.match(view.title, /训练/);
});

test("recovery approval has a readable title", () => {
  const view = approvalView({
    action: "recover_training",
    plan_hash: "sha256:recovery",
    summary: "resume",
  });
  assert.equal(view.title, "从 checkpoint 恢复训练");
});

test("scoring retry clearly states that inference artifacts are reused", () => {
  const view = approvalView({
    action: "retry_scoring",
    plan_hash: "sha256:score",
    summary: "reuse predictions",
  });
  assert.equal(view.title, "复用推理产物并重新评分");
});

test("failed candidate can finish without unsafe acceptance", () => {
  const view = approvalView({
    action: "finish_without_accepting",
    plan_hash: "sha256:finish",
    summary: "failed gates",
  });
  assert.equal(view.title, "结束本轮且不接受候选模型");
});

test("event merge is idempotent", () => {
  const merged = mergeEvents([{ event_id: 1 }], [{ event_id: 1 }, { event_id: 2 }]);
  assert.deepEqual(merged.map((row) => row.event_id), [1, 2]);
});

test("training view exposes config, vram and eta basis", () => {
  const view = trainingPlanView({
    config: { train: { learning_rate: 0.0001 } },
    decisions: [{ parameter: "cutoff_len", value: 2048 }],
    estimated_vram_gb: 18,
    estimated_steps: 100,
    estimated_hours_low: 1,
    estimated_hours_high: 2,
    eta_confidence: "medium",
    eta_basis: "historical_expanded_2",
    risks: ["risk"],
  });
  assert.equal(view.config.train.learning_rate, 0.0001);
  assert.equal(view.estimated_vram_gb, 18);
  assert.equal(view.eta_basis, "historical_expanded_2");
});

test("latest comparison is extracted from evaluation event", () => {
  const comparison = { paired_comparison: { n: 100, wins: 60, losses: 20 } };
  assert.deepEqual(latestComparison([
    { event_id: 1, event_type: "training_started", payload: {} },
    { event_id: 2, event_type: "evaluation_completed", payload: { comparison } },
  ]), comparison);
});

test("workflow label includes goal, iteration and update time", () => {
  const label = workflowLabel({
    state: "training",
    objective: { goal: "Improve FC routing" },
    iteration: 2,
    updated_at: "2026-08-20T01:02:03+00:00",
  });
  assert.match(label, /Improve FC routing/);
  assert.match(label, /第 3 轮/);
});

test("preflight can only rerun from planning states", () => {
  assert.equal(canRerunPreflight("preflight_blocked"), true);
  assert.equal(canRerunPreflight("training"), false);
});

test("SSE parser preserves partial frames and decodes structured events", () => {
  const partial = 'event: assistant_delta\ndata: {"delta":"训练方';
  const parsed = parseSseFrames(
    'event: progress\ndata: {"stage":"planning","message":"正在生成"}\n\n'
      + partial,
  );
  assert.deepEqual(parsed.events, [{
    type: "progress",
    data: { stage: "planning", message: "正在生成" },
  }]);
  assert.equal(parsed.rest, partial);

  const completed = parseSseFrames(
    `${parsed.rest}案。"}\r\n\r\nevent: done\r\ndata: {}\r\n\r\n`,
  );
  assert.deepEqual(completed.events, [
    { type: "assistant_delta", data: { delta: "训练方案。" } },
    { type: "done", data: {} },
  ]);
  assert.equal(completed.rest, "");
});

test("assistant stream path keeps create and follow-up endpoints distinct", () => {
  assert.equal(assistantStreamPath(null), "/api/assistant/workflows/stream");
  assert.equal(
    assistantStreamPath("wf_1"),
    "/api/assistant/workflows/wf_1/messages/stream",
  );
});

test("timeline keeps all eight stages in backend order", () => {
  const view = workflowTimeline({ workflow_steps: eightSteps("data_build") });
  assert.equal(view.length, 8);
  assert.equal(view[2].title, "数据构建");
  assert.equal(view[2].statusText, "进行中");
  assert.equal(view[2].progressText, "188 / 1000 · 18.8%");
});

test("manual cancellation is only offered for cancellable snapshots", () => {
  assert.equal(cancelView({ state: "training", available_actions: ["cancel"] }).visible, true);
  assert.equal(cancelView({ state: "completed", available_actions: [] }).visible, false);
  assert.equal(cancelView({ state: "cancelling", available_actions: [] }).busy, true);
});

test("artifact view keeps safe API links and hashes", () => {
  const item = artifactView({
    name: "数据报告",
    kind: "report",
    sha256: "a".repeat(64),
    download_url: "/api/datagen/jobs/job_1/report",
  });
  assert.equal(item.downloadUrl, "/api/datagen/jobs/job_1/report");
  assert.match(item.hashLabel, /^sha256:/);
  assert.equal(artifactView({ name: "bad", download_url: "https://evil.test" }).downloadUrl, null);
});

test("requirement evidence labels user defaults and assumptions", () => {
  assert.equal(requirementFieldView({ value: "FC 路由", source: "user" }).sourceText, "用户明确");
  assert.equal(requirementFieldView({ value: "SFT", source: "default" }).sourceText, "系统默认");
  assert.equal(
    requirementFieldView({ value: "Qwen", source: "assistant_assumption" }).sourceText,
    "助手假设",
  );
});
