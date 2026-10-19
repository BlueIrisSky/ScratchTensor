import assert from 'node:assert/strict';
import test from 'node:test';
import {compileProject, loadFixture, extensionPath} from './compiler.mjs';
import {loadDocumentationExamples} from './documents.mjs';
import {runProject} from './runtime.mjs';

const namespace = 'scratch-tensor-extension';
const json = (result, name) => JSON.parse(result.variables[name]);

function near(actual, expected, epsilon = 1e-6) {
  assert.equal(typeof actual, 'number', `expected a number, got ${typeof actual}`);
  assert(Number.isFinite(actual), `expected finite ${expected}, got ${actual}`);
  assert(Math.abs(actual - expected) <= epsilon, `expected ${expected} ± ${epsilon}, got ${actual}`);
}

async function documentedProgram(heading) {
  const {examples} = await loadDocumentationExamples();
  const matches = examples.filter(example => example.fullProgram &&
    example.document.replaceAll('\\', '/').endsWith('docs/demo.md') && example.heading.includes(heading));
  assert.equal(matches.length, 1, `expected one complete demo under ${heading}`);
  return matches[0].source;
}

test('the documented 250-step Linear training actually converges in the VM', {timeout: 90000}, async () => {
  const compiled = await compileProject({name: 'runtime-doc-training', source: await documentedProgram('完整训练')});
  const result = await runProject(compiled, {timeoutMs: 30000});
  near(result.variables.step, 250, 0);
  near(result.variables.initial_loss, 83 / 3, 1e-4);
  assert(result.variables.final_loss > 0 && result.variables.final_loss < 1e-5,
    `training did not converge: loss=${result.variables.final_loss}`);
  assert(result.variables.final_loss < result.variables.initial_loss / 100000);
  near(json(result, 'learned_W')[0][0], 2, 0.01);
  near(json(result, 'learned_b')[0], 1, 0.01);
  const prediction = json(result, 'prediction');
  assert.equal(prediction.length, 3);
  for (const [index, expected] of [3, 5, 7].entries()) near(prediction[index][0], expected, 0.01);
  for (const method of ['sgdFullBlock', 'backwardBlock', 'clearGradBlock']) {
    assert.equal(result.extensionCalls[`ScratchTensor.${method}`], 250, `${method} training count`);
  }
  assert(result.spoken.some(message => message.startsWith('训练完成，最终 MSE: ')));
  assert.equal(result.greenFlagFinished, true);
});

test('the documented basic example returns the stated product and gradient types', {timeout: 60000}, async () => {
  const compiled = await compileProject({name: 'runtime-doc-basic', source: await documentedProgram('最小乘法')});
  const result = await runProject(compiled);
  near(result.variables.product, 6, 0);
  assert.equal(typeof result.variables.gradient, 'string');
  assert.deepEqual(json(result, 'gradient'), [[3]]);
  assert.equal(result.variables.cleared_gradient, 'null');
  assert(result.spoken.includes('基础验证完成'));
  assert.equal(result.extensionCalls['ScratchTensor.backwardBlock'], 1);

  // Check the prose table as well as independent mathematical expectations: a
  // stale reference value must fail even when the executable example still works.
  const {documents} = await loadDocumentationExamples();
  const text = documents.find(document => document.path === 'docs/demo.md').text;
  const rows = [...text.matchAll(/^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*([^|]+)\|/gmu)];
  for (const name of ['product', 'gradient', 'cleared_gradient']) {
    const row = rows.find(match => match[1] === name);
    assert(row, `docs/demo.md is missing its ${name} reference row`);
    if (row[3].trim() === '数值') near(result.variables[name], Number(row[2]));
    else {
      assert.match(row[3], /字符串/u, `${name} must document its string type`);
      assert.equal(typeof result.variables[name], 'string');
      if (row[3].includes('JSON')) assert.deepEqual(JSON.parse(result.variables[name]), JSON.parse(row[2]));
      else assert.equal(result.variables[name], row[2]);
    }
  }
});

test('graph tracking, accumulation, and visited-node cleanup match the autograd documentation', {timeout: 60000}, async () => {
  const source = await loadFixture('runtime-graph-lifecycle');
  const result = await runProject(await compileProject({name: 'runtime-graph-lifecycle', source}), {allowExtensionErrors: true});
  assert.equal(result.variables.initial, 'null');
  assert.deepEqual(json(result, 'first'), [[9]]);
  assert.deepEqual(json(result, 'reused'), [[9]], 'released graph must not propagate to W again');
  assert.deepEqual(json(result, 'second'), [[12]], 'a new forward graph accumulates leaf gradients');
  assert.deepEqual(json(result, 'disabled'), [[12]], 'disabled tracking must detach new operations');
  assert.deepEqual(json(result, 'old_graph'), [[15]], 'disabling tracking must preserve previously built graphs');
  assert.equal(result.variables.intermediate, '');
  assert.equal(result.extensionErrors.length, 1);
  assert.equal(result.extensionErrors[0].method, 'ScratchTensor.getTensorPropertyBlock');
  assert.match(result.extensionErrors[0].message, /mid.*does not exist/);
  assert.deepEqual(json(result, 'target'), [[18]]);
  assert.deepEqual(json(result, 'unrelated'), [[6]], 'cleanup must not delete unvisited graph nodes');
  assert.deepEqual(json(result, 'parameter'), [[2]]);
  assert.deepEqual(json(result, 'input'), [[3]]);
  assert.equal(result.variables.cleared, 'null');
  assert.equal(result.variables.constant_gradient, 'null');
});

test('same-name creation replaces one registry entry and clears its old gradient', {timeout: 60000}, async () => {
  const source = await loadFixture('runtime-name-replacement');
  const result = await runProject(await compileProject({name: 'runtime-name-replacement', source}));
  const before = json(result, 'before').tensors;
  const after = json(result, 'after').tensors;
  assert.equal(before.length, 1);
  assert.equal(after.length, 1);
  assert.equal(after[0].name, 'W');
  assert.notEqual(after[0].id, before[0].id, 'creation assigns a new tensor identity');
  assert.deepEqual(after[0].data, [[5]]);
  near(result.variables.value, 5, 0);
  assert.equal(result.variables.gradient, 'null');
});

test('a .buildsb alias changes source spelling while preserving runtime extension identity', {timeout: 90000}, async () => {
  const documented = await documentedProgram('最小乘法');
  const original = await compileProject({name: 'runtime-original-namespace', source: documented});
  const aliased = await compileProject({name: 'runtime-aliased-namespace',
    source: documented.replaceAll(`${namespace}.`, 'tensor.'),
    extensionEntries: [`${JSON.stringify(extensionPath)} as "tensor"`]});
  assert.deepEqual(aliased.project.extensions, original.project.extensions);
  assert.deepEqual(Object.keys(aliased.project.extensionURLs), ['ScratchTensor']);
  assert.equal(aliased.project.extensionURLs.ScratchTensor, original.project.extensionURLs.ScratchTensor);
  const originalRun = await runProject(original);
  const aliasRun = await runProject(aliased);
  assert.deepEqual(aliasRun.variables, originalRun.variables);
  assert.deepEqual(aliasRun.extensionCalls, originalRun.extensionCalls);
  assert.deepEqual(aliasRun.registeredExtensions, ['ScratchTensor']);
  assert.equal(aliasRun.extensionCalls['ScratchTensor.binaryOpBlock'], 1);
});

test('equal-shaped trainable inputs own independent gradients during clipping', async () => {
  const source = await loadFixture('runtime-gradient-clipping');
  const result = await runProject(await compileProject({name: 'runtime-gradient-clipping', source}));
  near(json(result, 'first')[0], Math.SQRT1_2);
  near(json(result, 'second')[0], Math.SQRT1_2);
  near(Math.hypot(json(result, 'first')[0], json(result, 'second')[0]), 1);
});

test('command, reporter and procedure failures stop their whole thread while other threads finish', async () => {
  const source = await loadFixture('runtime-error-isolation');
  const result = await runProject(await compileProject({name: 'runtime-error-isolation', source}),
    {continueAfterExtensionErrors: true});
  assert.equal(result.extensionErrors.length, 6);
  assert.deepEqual(result.extensionErrors.map(error => error.method).sort(), [
    'ScratchTensor.binaryOpBlock', 'ScratchTensor.getTensorPropertyBlock',
    'ScratchTensor.getTensorPropertyBlock', 'ScratchTensor.backwardBlock',
    'ScratchTensor.concatBlock', 'ScratchTensor.stackBlock',
  ].sort());
  for (const error of result.extensionErrors) {
    assert.equal(error.threadRetired, true);
    assert(error.message.startsWith(`${error.method}: `));
    assert(result.spoken.includes(error.message), 'the failing sprite must say the opcode-bearing error');
  }
  assert.equal(result.executionErrors.length, 6);
  assert(result.executionErrors.every(error => error.fromExtension));
  for (const name of ['after_command', 'after_reporter', 'after_scalar', 'after_procedure',
    'after_caller', 'after_concat', 'after_stack']) {
    assert.equal(Number(result.variables[name]), 0, `${name} executed after its failure`);
  }
  assert.deepEqual(json(result, 'concat_value'), [99], 'concat must not silently filter absent inputs');
  assert.deepEqual(json(result, 'stack_value'), [88], 'stack must not silently filter absent inputs');
  assert.equal(Number(result.variables.independent_finished), 1);
  assert.equal(Number(result.variables.coordinator_finished), 1);
  assert(result.spoken.includes('其他线程继续运行'));
  assert.equal(result.aborted, false, 'the harness must not stop the whole project');
  assert.equal(result.greenFlagFinished, true);
});

test('operation outputs may reuse temporary names but cannot replace trainable leaf parameters', async () => {
  const source = await loadFixture('runtime-output-protection');
  const result = await runProject(await compileProject({name: 'runtime-output-protection', source}),
    {continueAfterExtensionErrors: true});
  assert.deepEqual(json(result, 'temporary'), [6]);
  assert.deepEqual(json(result, 'parameter'), [2]);
  assert.deepEqual(json(result, 'after'), json(result, 'before'));
  assert.equal(Number(result.variables.after_failure), 0);
  assert.equal(result.extensionErrors.length, 1);
  assert.equal(result.extensionErrors[0].method, 'ScratchTensor.binaryOpBlock');
  assert.match(result.extensionErrors[0].message, /parameter|leaf|trainable/i);
});
