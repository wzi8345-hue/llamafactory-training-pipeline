const test = require('node:test');
const assert = require('node:assert/strict');
const ui = require('./datagen.js');

test('DPO config carries preference prompts and task mode', () => {
  const cfg = ui.buildConfig({
    finetuneType: 'dpo', taskType: 'qa', mode: 'single', count: '2',
    rejectedPrompt: 'reject', pairJudgePrompt: 'pair',
  });
  assert.equal(cfg.finetune_type, 'dpo');
  assert.equal(cfg.task_type, 'qa');
  assert.equal(cfg.count, 2);
  assert.equal(cfg.rejected_prompt, 'reject');
  assert.equal(cfg.pair_judge_prompt, 'pair');
});

test('multi-document mode maps to qa_multi', () => {
  const cfg = ui.buildConfig({
    finetuneType: 'sft', taskType: 'qa', mode: 'consensus', count: '3',
  });
  assert.equal(cfg.task_type, 'qa_multi');
  assert.equal(cfg.sub_mode, 'consensus');
});

test('generated job label and stage preserve data type', () => {
  const job = {
    job_id: '20260811T000000Z-abcdef', finetune_type: 'dpo',
    task_type: 'fc', accepted: 2, target: 2,
  };
  assert.match(ui.optionLabel(job), /^\[DPO\]/);
  assert.equal(ui.stageForJob(job), 'dpo');
});

test('historical generated jobs default to SFT', () => {
  const job = {job_id: 'old', task_type: 'qa', accepted: 1, target: 1};
  assert.match(ui.optionLabel(job), /^\[SFT\]/);
  assert.equal(ui.stageForJob(job), 'sft');
});

test('prompt keys include DPO-only prompts only for DPO', () => {
  assert.deepEqual(ui.promptKeys('sft', 'fc', 'single'), {
    gen: 'default_fc_gen_prompt', judge: 'default_fc_judge_prompt',
  });
  assert.deepEqual(ui.promptKeys('dpo', 'qa', 'synthesis'), {
    gen: 'default_qa_multi_gen_synthesis',
    judge: 'default_qa_multi_judge_prompt',
    rejected: 'default_qa_dpo_rejected_prompt',
    pairJudge: 'default_dpo_pair_judge_prompt',
  });
});
