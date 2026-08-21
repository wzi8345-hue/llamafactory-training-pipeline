(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.DatagenUI = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  function finetuneType(value) {
    return value === 'dpo' ? 'dpo' : 'sft';
  }

  function promptKeys(type, taskType, mode) {
    let gen;
    let judge;
    if (taskType === 'fc') {
      gen = 'default_fc_gen_prompt';
      judge = 'default_fc_judge_prompt';
    } else if (mode === 'single') {
      gen = 'default_qa_gen_prompt';
      judge = 'default_qa_judge_prompt';
    } else {
      gen = mode === 'consensus'
        ? 'default_qa_multi_gen_consensus'
        : 'default_qa_multi_gen_synthesis';
      judge = 'default_qa_multi_judge_prompt';
    }
    const keys = {gen, judge};
    if (finetuneType(type) === 'dpo') {
      keys.rejected = taskType === 'fc'
        ? 'default_fc_dpo_rejected_prompt'
        : 'default_qa_dpo_rejected_prompt';
      keys.pairJudge = 'default_dpo_pair_judge_prompt';
    }
    return keys;
  }

  function numberValue(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function buildConfig(values) {
    const type = finetuneType(values.finetuneType);
    const taskType = values.taskType === 'fc' ? 'fc' : 'qa';
    const mode = values.mode || 'single';
    const multi = taskType === 'qa' && mode !== 'single';
    return {
      finetune_type: type,
      task_type: multi ? 'qa_multi' : taskType,
      count: numberValue(values.count, 100),
      kb_source_dir: (values.kbSourceDir || '').trim() || 'uploads',
      fc_seed_file: (values.fcSeedFile || '').trim(),
      min_len: numberValue(values.minLen, 200),
      temperature: numberValue(values.temperature, 0.9),
      judge_min_score: numberValue(values.judgeMinScore, 4),
      grounding_check: values.groundingCheck !== false && values.groundingCheck !== 'false',
      dedup_threshold: numberValue(values.dedupThreshold, 0.9),
      attempt_multiplier: numberValue(values.attemptMultiplier, 3),
      sub_mode: multi ? mode : 'synthesis',
      collection: (values.collection || '').trim(),
      n_docs: numberValue(values.nDocs, 3),
      neighbor_top_k: numberValue(values.neighborTopK, 20),
      gen_prompt: values.genPrompt || '',
      judge_prompt: values.judgePrompt || '',
      rejected_prompt: type === 'dpo' ? (values.rejectedPrompt || '') : '',
      pair_judge_prompt: type === 'dpo' ? (values.pairJudgePrompt || '') : '',
    };
  }

  function optionLabel(job) {
    const type = finetuneType(job.finetune_type).toUpperCase();
    const id = String(job.job_id || '');
    const shortId = id.length > 15 ? `${id.slice(0, 15)}…` : id;
    return `[${type}] ${shortId} · ${job.task_type || ''} · ${job.accepted || 0}/${job.target || 0} 条`;
  }

  function stageForJob(job) {
    return finetuneType(job && job.finetune_type);
  }

  return {buildConfig, optionLabel, promptKeys, stageForJob};
});
